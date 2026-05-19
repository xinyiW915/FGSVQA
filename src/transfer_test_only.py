import os
os.environ["DECORD_DUPLICATE_WARNING_THRESHOLD"] = "1.0"
import argparse
import csv
from pathlib import Path
import torch
from torch.amp import autocast
from tqdm import tqdm

from train import VQADataset, com_loss, pearsonr, read_vid_mos_csv, spearmanr
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

def infer_test_scale(rows):
    mos_values = [float(mos) for _vid, mos in rows]
    if not mos_values:
        raise ValueError("Cannot infer test scale from empty rows")

    lo = min(mos_values)
    hi = max(mos_values)

    if 0.0 <= lo and hi <= 1.0:
        return 0.0, 1.0
    if 1.0 <= lo and hi <= 5.0:
        return 1.0, 5.0
    if 0.0 <= lo and hi <= 5.0:
        return 0.0, 5.0
    return 0.0, 100.0

def linear_remap(x, src_min, src_max, dst_min, dst_max):
    src_min = float(src_min)
    src_max = float(src_max)
    dst_min = float(dst_min)
    dst_max = float(dst_max)

    if abs(src_max - src_min) <= 1e-12:
        raise ValueError("Source scale range must be non-zero")

    return (x - src_min) / (src_max - src_min) * (dst_max - dst_min) + dst_min

def save_predictions_csv(save_path, vids, y_true_raw, pred_train_scale, pred_eval_scale):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vid", "y_true_raw", "pred_train_scale", "pred_eval_scale"])
        for vid, y_true, pred_train, pred_eval in zip(
            vids,
            y_true_raw.tolist(),
            pred_train_scale.tolist(),
            pred_eval_scale.tolist(),
            strict=False,
        ):
            writer.writerow([vid, float(y_true), float(pred_train), float(pred_eval)])

    return save_path

@torch.no_grad()
def evaluate_and_collect(
    model,
    loader,
    device,
    *,
    amp=True,
    train_mos_mean,
    train_mos_std,
    train_scale_min,
    train_scale_max,
    test_scale_min,
    test_scale_max,
    desc="",
    show_pbar=True,
    log_interval=10,
):
    model.eval()

    losses = []
    y_all = []
    yhat_all = []
    vids_all = []

    it = loader
    if show_pbar:
        it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    for step, (rgb, w_art, w_str, y, vid) in enumerate(it, start=1):
        rgb = rgb.to(device, non_blocking=True)
        w_art = w_art.to(device, non_blocking=True)
        w_str = w_str.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        with autocast(device_type=device_type, enabled=(amp and device_type == "cuda")):
            yhat, _aux = model(rgb, w_art, w_str)
            loss, _loss_reg, _loss_rank = com_loss(yhat, y)

        losses.append(loss.detach().float().cpu())
        y_all.append(y.detach().float().cpu())
        yhat_all.append(yhat.detach().float().cpu())
        vids_all.extend(list(vid))

        if show_pbar and (step % int(log_interval) == 0 or step == len(loader)):
            avg_loss_so_far = torch.stack(losses).mean().item()
            it.set_postfix({"loss": f"{avg_loss_so_far:.4f}"})

    if y_all:
        y_all = torch.cat(y_all, dim=0)
        yhat_all = torch.cat(yhat_all, dim=0)
    else:
        y_all = torch.empty(0)
        yhat_all = torch.empty(0)

    y_true_raw = y_all * float(train_mos_std) + float(train_mos_mean)
    pred_train_scale = yhat_all * float(train_mos_std) + float(train_mos_mean)
    pred_eval_scale = linear_remap(
        pred_train_scale,
        src_min=float(train_scale_min),
        src_max=float(train_scale_max),
        dst_min=float(test_scale_min),
        dst_max=float(test_scale_max),
    )

    plcc = pearsonr(y_true_raw, pred_eval_scale).item() if y_true_raw.numel() > 1 else 0.0
    srcc = spearmanr(y_true_raw, pred_eval_scale).item() if y_true_raw.numel() > 1 else 0.0
    rmse = (
        torch.sqrt(torch.mean((pred_eval_scale - y_true_raw) ** 2)).item()
        if y_true_raw.numel() > 0
        else 0.0
    )
    avg_loss = torch.stack(losses).mean().item() if losses else 0.0

    return {
        "loss": avg_loss,
        "plcc": plcc,
        "srcc": srcc,
        "rmse": rmse,
        "vids": vids_all,
        "y_true_raw": y_true_raw,
        "pred_train_scale": pred_train_scale,
        "pred_eval_scale": pred_eval_scale,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, default="/home/xinyi/Project/FD-VQA/src/checkpoints/lsvq/qd_model.best.pt")
    ap.add_argument("--csv_path", type=str, default="/home/xinyi/Project/FD-VQA/metadata/KVQ_metadata.csv")
    ap.add_argument("--db_path", type=str, default="/media/xinyi/server/video_dataset/KVQ")

    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--win", type=int, default=6)
    ap.add_argument("--win_step", type=int, default=1)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--train_scale_min", type=float, default=0.0)
    ap.add_argument("--train_scale_max", type=float, default=100.0)
    ap.add_argument("--test_scale_min", type=float, default=1.0)
    ap.add_argument("--test_scale_max", type=float, default=5.0)

    ap.add_argument("--save_pred_csv", type=str, default="/home/xinyi/Project/FD-VQA/src/transfer_test/transfer_test_only_konvid_1k.csv")
    args = ap.parse_args()

    device = torch.device(args.device)
    amp = not bool(args.no_amp)
    ckpt_info = load_checkpoint(Path(args.ckpt_path), device)

    train_mos_mean = ckpt_info["train_mos_mean"]
    train_mos_std = ckpt_info["train_mos_std"]
    if train_mos_mean is None or train_mos_std is None:
        raise ValueError(
            "Prefer loading *.best.pt / *.pt, or pass --train_mos_mean and --train_mos_std manually."
        )
    if float(train_mos_std) <= 1e-8:
        raise ValueError("train_mos_std must be > 0")

    rows = read_vid_mos_csv(args.csv_path)
    if not rows:
        raise ValueError(f"No rows found in csv: {args.csv_path}")

    if args.test_scale_min is None or args.test_scale_max is None:
        inferred_test_scale_min, inferred_test_scale_max = infer_test_scale(rows)
        test_scale_min = inferred_test_scale_min
        test_scale_max = inferred_test_scale_max
    else:
        test_scale_min = float(args.test_scale_min)
        test_scale_max = float(args.test_scale_max)

    dataset = VQADataset(
        rows,
        args.db_path,
        clip_len=args.clip_len,
        size=args.resize,
        win=args.win,
        win_step=args.win_step,
        mos_mean=float(train_mos_mean),
        mos_std=float(train_mos_std),
    )

    pin = str(device).startswith("cuda")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin,
        drop_last=False,
        prefetch_factor=4 if int(args.num_workers) > 0 else None,
    )

    model = QD_MODEL(
        clip_model="openai/clip-vit-base-patch16",
    ).to(device)
    model.load_state_dict(ckpt_info["state_dict"], strict=True)

    print(f"Loaded checkpoint: {args.ckpt_path}")
    print(f"Training normalization: mean={float(train_mos_mean):.6f}, std={float(train_mos_std):.6f}")
    print(
        f"Scale mapping: train=[{float(args.train_scale_min):.3f}, {float(args.train_scale_max):.3f}] -> "
        f"test=[{float(test_scale_min):.3f}, {float(test_scale_max):.3f}]"
    )
    print(f"Test rows: {len(rows)}")

    metrics = evaluate_and_collect(
        model,
        loader,
        device,
        amp=amp,
        train_mos_mean=float(train_mos_mean),
        train_mos_std=float(train_mos_std),
        train_scale_min=float(args.train_scale_min),
        train_scale_max=float(args.train_scale_max),
        test_scale_min=float(test_scale_min),
        test_scale_max=float(test_scale_max),
        desc="Cross-dataset test",
        show_pbar=True,
        log_interval=10,
    )

    print(
        "TEST | "
        f"loss={metrics['loss']:.4f} "
        f"plcc={metrics['plcc']:.4f} "
        f"srcc={metrics['srcc']:.4f} "
        f"rmse={metrics['rmse']:.4f}"
    )

    if args.save_pred_csv:
        save_path = save_predictions_csv(
            args.save_pred_csv,
            metrics["vids"],
            metrics["y_true_raw"],
            metrics["pred_train_scale"],
            metrics["pred_eval_scale"],
        )
        print(f"Saved predictions to: {save_path}")


if __name__ == "__main__":
    main()
