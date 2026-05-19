import os
os.environ.setdefault("DECORD_DUPLICATE_WARNING_THRESHOLD", "1.0")

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm
from thop import profile
from thop import clever_format

from train import VQADataset
from model.qd_model import QD_MODEL


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return {
            "state_dict": ckpt["model"],
            "train_mos_mean": ckpt.get("mos_mean"),
            "train_mos_std": ckpt.get("mos_std"),
            "train_args": ckpt.get("args", {}),
            "is_full_checkpoint": True,
        }
    if isinstance(ckpt, dict):
        return {
            "state_dict": ckpt,
            "train_mos_mean": None,
            "train_mos_std": None,
            "train_args": {},
            "is_full_checkpoint": False,
        }
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")

class ForwardWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rgb, w_art, w_str):
        yhat, _aux = self.model(rgb, w_art, w_str)
        return yhat

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def build_dataset(args, mos_mean, mos_std):
    rows = [(args.video_id, float(args.dummy_mos))]
    dataset = VQADataset(
        rows,
        args.db_path,
        clip_len=args.clip_len,
        size=args.resize,
        win=args.win,
        win_step=args.win_step,
        mos_mean=float(mos_mean),
        mos_std=float(mos_std),
    )
    return dataset

def prepare_single_sample(dataset, device):
    rgb, w_art, w_str, y, vid = dataset[0]

    rgb = rgb.unsqueeze(0).to(device, non_blocking=True)
    w_art = w_art.unsqueeze(0).to(device, non_blocking=True)
    w_str = w_str.unsqueeze(0).to(device, non_blocking=True)
    y = y.unsqueeze(0).to(device, non_blocking=True).float()

    if isinstance(vid, (list, tuple)):
        video_id = vid[0]
    else:
        video_id = vid
    return rgb, w_art, w_str, y, video_id


@torch.no_grad()
def predict_once(
    model,
    rgb,
    w_art,
    w_str,
    *,
    device,
    amp,
    train_mos_mean,
    train_mos_std,
):
    model.eval()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    with autocast(device_type=device_type, enabled=(amp and device_type == "cuda")):
        yhat, _aux = model(rgb, w_art, w_str)

    pred_score = yhat.detach().float().cpu() * float(train_mos_std) + float(train_mos_mean)
    return float(pred_score.squeeze().item())

@torch.no_grad()
def profile_with_thop(model, rgb, w_art, w_str):
    macs, params = profile(model, inputs=(rgb, w_art, w_str), verbose=False)
    flops = 2 * macs
    macs, flops, params = clever_format([macs, flops, params], "%.3f")
    return macs, flops, params

@torch.no_grad()
def benchmark_forward(model, rgb, w_art, w_str, *, device, amp, num_runs=10, warmup=3):
    model.eval()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    for _ in range(max(0, warmup)):
        with autocast(device_type=device_type, enabled=(amp and device_type == "cuda")):
            _ = model(rgb, w_art, w_str)
    if device_type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(int(num_runs)):
        with autocast(device_type=device_type, enabled=(amp and device_type == "cuda")):
            _ = model(rgb, w_art, w_str)
    if device_type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / max(1, int(num_runs))


def run_end_to_end_once(args, model, train_mos_mean, train_mos_std, device, amp):
    start = time.perf_counter()
    dataset = build_dataset(args, train_mos_mean, train_mos_std)
    rgb, w_art, w_str, _y, _video_id = prepare_single_sample(dataset, device)

    pred_score = predict_once(
        model,
        rgb,
        w_art,
        w_str,
        device=device,
        amp=amp,
        train_mos_mean=float(train_mos_mean),
        train_mos_std=float(train_mos_std),
    )
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, pred_score, rgb, w_art, w_str


def parse_args():
    ap = argparse.ArgumentParser(description="Demo-style single-video test for QD_MODEL")
    # for complexity time test:
    ap.add_argument("--ckpt_path", type=str, default="/home/xinyi/Project/FD-VQA/src/checkpoints/lsvq/qd_model.best.pt")
    ap.add_argument("--db_path", type=str, default="/home/xinyi/Project/FD-VQA/test_videos/")
    ap.add_argument("--video_id", type=str, default="SDR_Animal_5ngj")
    # for resolution compelxity test:
    # ap.add_argument("--ckpt_path", type=str, default="/home/xinyi/Project/FD-VQA/src/checkpoints/kvq/qd_model.best.pt")
    # ap.add_argument("--db_path", type=str, default="/home/xinyi/Project/FD-VQA/test_videos/complexity_test/complexity_resolution/")
    # ap.add_argument("--video_id", type=str, default="SDR_Animal_5ngj_540p")

    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--win", type=int, default=6)
    ap.add_argument("--win_step", type=int, default=1)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--dummy_mos", type=float, default=3.0, help="Only used to compute VQADataset, does not affect prediction")
    ap.add_argument("--num_runs", type=int, default=10, help="Average N runs")
    ap.add_argument("--warmup_runs", type=int, default=3)
    ap.add_argument("--skip_profile", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    amp = not bool(args.no_amp)

    print(f"Running on {'GPU' if device.type == 'cuda' else 'CPU'}")

    display_path = str(Path(args.db_path) / args.video_id)
    info = pd.DataFrame([
        {
            "vid": args.video_id,
            "test_video_path": display_path,
        }
    ])
    print(info)

    dataset_preview = build_dataset(args, mos_mean=args.dummy_mos, mos_std=1.0)
    print(f"Dataset loaded. Total videos: {len(dataset_preview)}, Total batches: 1")
    print(f"Loading model from: {args.ckpt_path}")

    ckpt_info = load_checkpoint(Path(args.ckpt_path), device)
    train_mos_mean = ckpt_info["train_mos_mean"]
    train_mos_std = ckpt_info["train_mos_std"]
    if train_mos_mean is None or train_mos_std is None:
        raise ValueError("Checkpoint does not contain mos_mean / mos_std. Please use a full checkpoint.")
    if float(train_mos_std) <= 1e-8:
        raise ValueError("train_mos_std must be > 0")

    model = QD_MODEL(clip_model="openai/clip-vit-base-patch16").to(device)
    model.load_state_dict(ckpt_info["state_dict"], strict=True)
    model.eval()

    run_times = []
    pred_score = None
    rgb = w_art = w_str = None

    for i in range(args.num_runs):
        for _ in tqdm(range(1), desc="Processing Videos"):
            elapsed, pred_score, rgb, w_art, w_str = run_end_to_end_once(
                args, model, train_mos_mean, train_mos_std, device, amp
            )
        run_times.append(elapsed)
        print(f"Run {i + 1} - Time taken: {elapsed:.4f} seconds")

    avg_total_time = sum(run_times) / max(1, len(run_times))
    avg_forward_time = benchmark_forward(
        model,
        rgb,
        w_art,
        w_str,
        device=device,
        amp=amp,
        num_runs=args.num_runs,
        warmup=args.warmup_runs,
    )

    total_params, trainable_params = count_parameters(model)
    macs = flops = params = None
    if not args.skip_profile:
        try:
            macs, flops, params = profile_with_thop(model, rgb, w_art, w_str)
        except Exception as e:
            print(f"[WARN] THOP profiling failed: {e}")

    print(f"Average running time over {args.num_runs} runs: {avg_total_time:.4f} seconds")
    print(f"Predicted Quality Score: {pred_score:.4f}")
    print("\n========== PROFILE SUMMARY ==========")
    print(f"video_id              : {args.video_id}")
    print(f"rgb shape             : {tuple(rgb.shape)}")
    print(f"w_art shape           : {tuple(w_art.shape)}")
    print(f"w_str shape           : {tuple(w_str.shape)}")
    print(f"train mos mean/std    : {float(train_mos_mean):.6f} / {float(train_mos_std):.6f}")
    print(f"predicted score       : {pred_score:.6f}")
    print(f"params total          : {total_params:,} ({total_params / 1e6:.3f} M)")
    print(f"params trainable      : {trainable_params:,} ({trainable_params / 1e6:.3f} M)")
    if params is not None:
        print(f"Params (THOP)         : {params} M")
    if macs is not None:
        print(f"MACs (THOP)           : {macs} G")
    if flops is not None:
        print(f"FLOPs (~2*MACs)       : {flops} G")
    print(f"avg forward time      : {avg_forward_time:.6f} s  (runs={args.num_runs})")
    print(f"avg end-to-end time   : {avg_total_time:.6f} s  (sample prep + forward)")
    print("=====================================")


if __name__ == "__main__":
    main()