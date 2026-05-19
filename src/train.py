import os
os.environ["DECORD_DUPLICATE_WARNING_THRESHOLD"] = "1.0"
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr as _pearsonr_scipy, spearmanr as _spearmanr_scipy
import torch
import torch.nn.functional as F
import cv2
from decord import VideoReader, cpu
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from module.compute_weight_map import process_video, compute_weight_map
from model.qd_model import QD_MODEL


# ----------------------------
# Data utils
# ----------------------------
def read_vid_mos_csv(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("CSV has no header")
        for r in reader:
            vid = str(r["vid"]).strip()
            mos = float(r["mos"])
            rows.append((vid, mos))
    return rows

def split_rows(rows, seed=42, test_ratio=0.2, val_ratio=0.1):
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(rows))
    rng.shuffle(idx)

    n = len(rows)
    n_test = int(round(n * test_ratio))
    n_train_all = n - n_test  # train+val
    n_val = int(round(n_train_all * val_ratio))    # val from train_all

    val = [rows[i] for i in idx[:n_val]]
    train = [rows[i] for i in idx[n_val:n_train_all]]
    test = [rows[i] for i in idx[n_train_all:]]
    return train, val, test

def split_train_val(rows, seed=42, val_ratio=0.1):
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(rows))
    rng.shuffle(idx)

    n = len(rows)
    n_val = int(round(n * val_ratio))
    val = [rows[i] for i in idx[:n_val]]
    train = [rows[i] for i in idx[n_val:]]
    return train, val

def pearsonr(x, y, eps=1e-12):
    # PLCC (SciPy): returns a torch scalar tensor so call-site ".item()" still works
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    # avoid NaN when constant / too short
    if x.size < 2 or np.std(x) < eps or np.std(y) < eps:
        return torch.tensor(0.0)
    r, _p = _pearsonr_scipy(x, y)
    if np.isnan(r):
        r = 0.0
    return torch.tensor(float(r))

def spearmanr(x, y, eps=1e-12):
    # SRCC (SciPy): handles ties correctly; returns torch scalar tensor
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()

    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size < 2 or np.std(x) < eps or np.std(y) < eps:
        return torch.tensor(0.0)

    r, _p = _spearmanr_scipy(x, y)
    if np.isnan(r):
        r = 0.0
    return torch.tensor(float(r))

# ----------------------------
# Train utils
# ----------------------------
def build_scheduler(optim, args):
    warm = int(args.warmup_epochs)
    total = int(args.epochs)
    warm = max(0, min(warm, total - 1))

    # warmup
    warmup = torch.optim.lr_scheduler.LinearLR(
        optim,
        start_factor=0.1,
        total_iters=warm if warm > 0 else 1,
    )
    # cosine warmup
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=(total - warm) if (total - warm) > 0 else 1,
        eta_min=float(args.min_lr),
    )

    if warm > 0:
        return torch.optim.lr_scheduler.SequentialLR(
            optim,
            schedulers=[warmup, cosine],
            milestones=[warm],
        )
    return cosine

def com_loss(y_pred, y_true, reg_w=0.6, rank_w=1.0, huber_beta=1.0, margin=0.0):
    # 1) Huber / SmoothL1
    if huber_beta is None:
        reg_loss = F.l1_loss(y_pred, y_true, reduction="mean")
    else:
        reg_loss = F.smooth_l1_loss(y_pred, y_true, beta=float(huber_beta), reduction="mean")
    reg_loss = reg_loss * float(reg_w)

    # 2) pairwise hinge rank
    B = y_true.shape[0]
    if B < 2 or float(rank_w) == 0.0:
        rank_loss = y_pred.new_tensor(0.0)
        return reg_loss + rank_loss, reg_loss, rank_loss
    pred_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)  # [B,B]
    true_diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)  # [B,B]
    s = torch.sign(true_diff)                               # -1,0,+1

    # y_true = pair, ignore
    mask = (s != 0).float()
    # hinge: max(0, margin - s*(pred_i - pred_j))
    rank_mat = F.relu(float(margin) - s * pred_diff) * mask
    denom = mask.sum().clamp_min(1.0)
    rank_loss = rank_mat.sum() / denom
    rank_loss = rank_loss * float(rank_w)

    total = reg_loss + rank_loss
    return total, reg_loss, rank_loss

# ----------------------------
# Dataset
# ----------------------------
class VQADataset(torch.utils.data.Dataset):
    """
    Returns per item:
      rgb:  [3, T, H, W] float in [0,1]  (RGB)
      w_art: [1, T, H, W] float in [0,1]
      w_str: [1, T, H, W] float in [0,1]
      y:    scalar float (MOS, optional normalized)
      vid:  str
    """
    def __init__(self, rows, db_path, clip_len, size, win, win_step, mos_mean=None, mos_std=None):
        self.rows = rows
        self.db_path = str(db_path)
        self.clip_len = int(clip_len)
        self.size = int(size)
        self.win = int(win)
        self.win_step = int(win_step)
        self.mos_mean = mos_mean
        self.mos_std = mos_std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        vid, mos = self.rows[int(idx)]
        num_anchors = self.clip_len
        size = self.size
        win = self.win
        win_step = self.win_step

        # get video path
        base_path = Path(self.db_path) / vid
        video_path = None
        for ext in ("mp4", "avi", "mkv"):
            p = Path(str(base_path) + f".{ext}")
            if p.exists():
                video_path = str(p)
                break
        if video_path is None:
            raise FileNotFoundError(f"Cannot find {vid} video")

        try:
            # read video
            vr = VideoReader(video_path, ctx=cpu(0))
            frame_all, w_art_all, w_str_all, anchors_kept = process_video(
                vr,
                size=size, num_anchors=num_anchors, win=win, win_step=win_step,
            )
            frames_np, w_art_np, w_str_np = compute_weight_map(frame_all, w_art_all, w_str_all)
            # print("frames_np:", frames_np.shape, frames_np.dtype)
            # print("w_art_np:", w_art_np.shape, w_art_np.dtype)
            # print("w_str_np:", w_str_np.shape, w_str_np.dtype)
            # print("anchors_kept:", len(anchors_kept), "example:", anchors_kept[0])
        except Exception as e:
            print("\n[DATA ERROR]")
            print("idx:", idx)
            print("vid:", vid)
            raise
        finally:
            # release decord video reader
            try:
                if vr is not None:
                    del vr
            except Exception:
                pass

        # fixed length sampling
        T = self.clip_len

        # frames_sel = [cv2.cvtColor(frames_np[i], cv2.COLOR_BGR2RGB) for i in range(T)] # frame BGR -> RGB: [T,H,W,3] -> [3,T,H,W]
        frames_sel = [frames_np[i] for i in range(T)]  # RGB frames_np
        rgb = torch.from_numpy(np.stack(frames_sel, axis=0)).float()
        rgb = rgb.permute(3, 0, 1, 2).contiguous() / 255.0

        # W_art / W_str: [T,H,W] -> [1,T,H,W]
        w_art = torch.from_numpy(np.stack([w_art_np[i] for i in range(T)], axis=0).astype(np.float32)).unsqueeze(0).float()
        w_str = torch.from_numpy(np.stack([w_str_np[i] for i in range(T)], axis=0).astype(np.float32)).unsqueeze(0).float()

        # MOS
        y = float(mos)
        if self.mos_mean is not None and self.mos_std is not None:
            y = (y - self.mos_mean) / (self.mos_std + 1e-8)
        y = torch.tensor(y).float()
        return rgb, w_art, w_str, y, str(vid)


# ----------------------------
# Train
# ----------------------------
@torch.no_grad()
def _gather_cat(xs):
    if not xs:
        return torch.empty(0)
    return torch.cat(xs, dim=0)

def run_epoch(model, loader, device, *, optim=None, amp=True, mos_mean=None, mos_std=None, desc="", show_pbar=True, log_interval=10):
    is_train = optim is not None
    model.train(is_train)

    scaler = getattr(run_epoch, "_scaler", None)
    if scaler is None:
        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        run_epoch._scaler = GradScaler(device_type, enabled=(amp and device_type == "cuda"))
        scaler = run_epoch._scaler

    losses = []
    y_all = []
    yhat_all = []

    # ---- tqdm progress bar ----
    it = loader
    if show_pbar:
        it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    for step, (rgb, w_art, w_str, y, vid) in enumerate(it, start=1):
        rgb = rgb.to(device, non_blocking=True)  # [B,3,T,H,W]
        w_art = w_art.to(device, non_blocking=True)   # [B,1,T,H,W]
        w_str = w_str.to(device, non_blocking=True)   # [B,1,T,H,W]
        y = y.to(device, non_blocking=True).float()  # [B]

        if is_train:
            optim.zero_grad(set_to_none=True)

        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        if is_train:
            with autocast(device_type=device_type, enabled=(amp and device_type == "cuda")):
                yhat, _aux = model(rgb, w_art, w_str)  # yhat: [B]
                loss, loss_reg, loss_rank = com_loss(yhat, y)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            with torch.inference_mode():
                with autocast(device_type=device_type, enabled=(amp and device_type == "cuda")):
                    yhat, _aux = model(rgb, w_art, w_str)
                    loss, loss_reg, loss_rank = com_loss(yhat, y)

        loss_cpu = loss.detach().float().cpu()
        losses.append(loss_cpu)
        y_all.append(y.detach().float().cpu())
        yhat_all.append(yhat.detach().float().cpu())

        # ---- update bar every log_interval steps ----
        if show_pbar and (step % int(log_interval) == 0 or step == len(loader)):
            avg_loss_so_far = torch.stack(losses).mean().item()
            lrs = None
            if is_train and hasattr(optim, "param_groups") and optim.param_groups:
                lrs = [pg.get("lr", None) for pg in optim.param_groups]

            postfix = {"loss": f"{avg_loss_so_far:.4f}"}
            if lrs is not None:
                postfix["lrs"] = ",".join([f"{x:.2e}" for x in lrs if x is not None])
            it.set_postfix(postfix)

    y_all = _gather_cat(y_all)
    yhat_all = _gather_cat(yhat_all)

    # ---- 反标准化：在 MOS 原尺度上算相关系数 ----
    if mos_mean is not None and mos_std is not None:
        y_all = y_all * mos_std + mos_mean
        yhat_all = yhat_all * mos_std + mos_mean

    plcc = pearsonr(y_all, yhat_all).item() if y_all.numel() > 1 else 0.0
    srcc = spearmanr(y_all, yhat_all).item() if y_all.numel() > 1 else 0.0
    rmse = torch.sqrt(torch.mean((yhat_all - y_all) ** 2)).item() if y_all.numel() > 0 else 0.0

    avg_loss = torch.stack(losses).mean().item() if losses else 0.0
    return avg_loss, plcc, srcc, rmse


def main():
    ap = argparse.ArgumentParser()
    # ----- data -----
    ap.add_argument("--csv_path", default="/home/xinyi/Project/FD-VQA/metadata/LSVQ_TRAIN_metadata.csv")
    ap.add_argument("--db_path", default="/media/xinyi/server/LSVQ/")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--val_ratio", type=float, default=0.1)  # train 80%
    # ----- video processing -----
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--win", type=int, default=6)
    ap.add_argument("--win_step", type=int, default=1)
    # ----- runtime -----
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no_amp", action="store_true")
    # ----- hyperparams -----
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--finetune_lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--clip_unfreeze_blocks", type=int, default=4)
    ap.add_argument("--finetune_last_stage", action="store_true")
    ap.add_argument("--patience", type=int, default=6)
    # ----- save -----
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--save_name", type=str, default="qd_model.pt")

    args = ap.parse_args()
    torch.manual_seed(args.split_seed)
    device = torch.device(args.device)
    amp = not bool(args.no_amp)

    # ----------------------------
    # Load rows and split
    # ----------------------------
    csv_path = Path(args.csv_path)
    if csv_path.name == "LSVQ_TRAIN_metadata.csv":
        # LSVQ official split
        test_csv = csv_path.parent / "LSVQ_TEST_metadata.csv"
        if not test_csv.exists():
            raise FileNotFoundError(f"Cannot find LSVQ test csv: {test_csv}")
        train_all = read_vid_mos_csv(str(csv_path))
        test_rows = read_vid_mos_csv(str(test_csv))
        train_rows, val_rows = split_train_val(
            train_all,
            seed=args.split_seed,
            val_ratio=args.val_ratio,
        )
        print(f"[LSVQ split] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    elif csv_path.name == "KVQ_TRAIN_metadata.csv":
        # KVQ challenge split
        val_csv = csv_path.parent / "KVQ_VAL_metadata.csv"
        test_csv = csv_path.parent / "KVQ_TEST_metadata.csv"
        train_rows = read_vid_mos_csv(str(csv_path))
        val_rows = read_vid_mos_csv(str(val_csv))
        test_rows = read_vid_mos_csv(str(test_csv))
        print(f"[KVQ split] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    else:
        # default split for other datasets
        rows = read_vid_mos_csv(str(csv_path))
        train_rows, val_rows, test_rows = split_rows(
            rows,
            seed=args.split_seed,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
        )
        # print("sizes:", len(rows), len(train_rows), len(val_rows), len(test_rows))
        # print("train first 3:", train_rows[:3])
        # print("val   first 3:", val_rows[:3])
        # print("test  first 3:", test_rows[:3])

    # MOS normalization stats from train split
    mos_train = np.array([mos for _vid, mos in train_rows], dtype=np.float32)
    mos_mean = float(mos_train.mean()) if len(mos_train) else 0.0
    mos_std = float(mos_train.std()) if len(mos_train) else 1.0
    if mos_std <= 1e-8:
        mos_std = 1.0

    # ----------------------------
    # DB and datasets
    # ----------------------------
    ds_train = VQADataset(
        train_rows, args.db_path,
        clip_len=args.clip_len,
        size=args.resize,
        win=args.win,
        win_step=args.win_step,
        mos_mean=mos_mean,
        mos_std=mos_std,
    )
    ds_val = VQADataset(
        val_rows, args.db_path,
        clip_len=args.clip_len,
        size=args.resize,
        win=args.win,
        win_step=args.win_step,
        mos_mean=mos_mean,
        mos_std=mos_std,
    )
    ds_test = VQADataset(
        test_rows, args.db_path,
        clip_len=args.clip_len,
        size=args.resize,
        win=args.win,
        win_step=args.win_step,
        mos_mean=mos_mean,
        mos_std=mos_std,
    )

    pin = str(device).startswith("cuda")

    loader_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        # persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=False,
    )
    loader_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    loader_test = torch.utils.data.DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=False,
    )

    # ----------------------------
    # Model
    # ----------------------------
    model = QD_MODEL(
        clip_model="openai/clip-vit-base-patch16",
    ).to(device)

    # Stage A: freeze CLIP
    model.freeze_clip_all()

    clip_params_all = []
    other_params_all = []
    for name, p in model.named_parameters():
        if name.startswith("encoder."):
            clip_params_all.append(p)
        else:
            other_params_all.append(p)

    param_groups = []
    if other_params_all:
        param_groups.append({"params": other_params_all, "lr": float(args.lr)})
    if clip_params_all:
        param_groups.append({"params": clip_params_all, "lr": float(args.finetune_lr)})

    optim = torch.optim.AdamW(param_groups, weight_decay=float(args.weight_decay))
    scheduler = build_scheduler(optim, args)
    # ----------------------------
    # Train loop
    # ----------------------------
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / args.save_name
    best_path = save_dir / (Path(args.save_name).stem + ".best.pt")
    best_weights_path = save_dir / (Path(args.save_name).stem + ".best_weights.pt")

    best_val_srcc = -1e18
    bad_epochs = 0
    did_unfreeze = False

    for epoch in tqdm(range(1, int(args.epochs) + 1), desc="Epochs", dynamic_ncols=True):

        # Stage B: optional CLIP finetune after warmup
        if (not did_unfreeze) and bool(args.finetune_last_stage) and epoch == (int(args.warmup_epochs) + 1):
            model.unfreeze_clip_last_blocks(n_blocks=int(args.clip_unfreeze_blocks), also_unfreeze_ln=True)
            did_unfreeze = True

            # 不重建 optim, 保留 Adam 状态，只更新 lr
            if hasattr(optim, "param_groups") and len(optim.param_groups) >= 2:
                optim.param_groups[0]["lr"] = float(args.lr)
                optim.param_groups[1]["lr"] = float(args.finetune_lr)

            print(
                f"[Stage B] Unfroze CLIP last {int(args.clip_unfreeze_blocks)} blocks | "
                f"lr={float(args.lr)} finetune_lr={float(args.finetune_lr)}"
            )

        tr_loss, tr_plcc, tr_srcc, tr_rmse = run_epoch(
            model, loader_train, device,
            optim=optim,
            amp=amp,
            mos_mean=mos_mean,
            mos_std=mos_std,
            desc=f"Train e{epoch:03d}",
            show_pbar=True,
            log_interval=10,
        )
        va_loss, va_plcc, va_srcc, va_rmse = run_epoch(
            model, loader_val, device,
            optim=None,
            amp=amp,
            mos_mean=mos_mean,
            mos_std=mos_std,
            desc=f"Val   e{epoch:03d}",
            show_pbar=True,
            log_interval=10,
        )
        scheduler.step()
        print(
            f"epoch {epoch:03d} | "
            f"train: loss={tr_loss:.4f} plcc={tr_plcc:.4f} srcc={tr_srcc:.4f} rmse={tr_rmse:.4f} | "
            f"val: loss={va_loss:.4f} plcc={va_plcc:.4f} srcc={va_srcc:.4f} rmse={va_rmse:.4f}"
        )

        # Save "last" checkpoint every epoch
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "mos_mean": mos_mean,
            "mos_std": mos_std,
            "args": vars(args),
            "best_val_srcc": best_val_srcc,
        }
        torch.save(ckpt, str(save_path))

        # Save best by val SRCC (higher is better)
        if va_srcc > best_val_srcc:
            best_val_srcc = va_srcc
            bad_epochs = 0
            ckpt["best_val_srcc"] = best_val_srcc
            torch.save(ckpt, str(best_path))
            torch.save(model.state_dict(), str(best_weights_path))
            print(
                f"  [best] val_srcc={best_val_srcc:.4f} (val_rmse={va_rmse:.4f}) -> saved "
                f"{best_path} and {best_weights_path}"
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                print(
                    f"[EarlyStop] val_srcc did not improve for {bad_epochs} epochs. "
                    f"Stop at epoch {epoch}."
                )
                break
    # ----------------------------
    # Test (load best)
    # ----------------------------
    if best_weights_path.exists():
        sd = torch.load(str(best_weights_path), map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=True)
        print(f"Loaded best weights: {best_weights_path}")
    elif best_path.exists():
        best = torch.load(str(best_path), map_location=device)
        model.load_state_dict(best["model"], strict=True)
        print(f"Loaded best checkpoint: {best_path} (val_srcc={best.get('best_val_srcc', None)})")

    te_loss, te_plcc, te_srcc, te_rmse = run_epoch(
        model, loader_test, device,
        optim=None,
        amp=amp,
        mos_mean=mos_mean,
        mos_std=mos_std,
    )
    print(f"TEST | loss={te_loss:.4f} plcc={te_plcc:.4f} srcc={te_srcc:.4f} rmse={te_rmse:.4f}")


if __name__ == "__main__":
    main()