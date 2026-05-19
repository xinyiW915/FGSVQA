import argparse
from pathlib import Path
import numpy as np
import torch

from train import (
    read_vid_mos_csv,
    split_rows,
    VQADataset,
    run_epoch,
    build_scheduler,
)
from model.qd_model import QD_MODEL


def load_pretrained_weights(model, pretrained_path, device):
    p = Path(pretrained_path)
    obj = torch.load(str(p), map_location=device, weights_only=True)
    # *.pt: dict checkpoint
    if isinstance(obj, dict) and "model" in obj:
        model.load_state_dict(obj["model"], strict=True)
        return obj
    # best_weights.pt: state_dict
    else:
        model.load_state_dict(obj, strict=True)
        return None

def make_loaders(rows_train, rows_val, rows_test, args, mos_mean, mos_std, device):
    ds_train = VQADataset(
        rows_train, args.db_path,
        clip_len=args.clip_len, size=args.resize, win=args.win, win_step=args.win_step,
        mos_mean=mos_mean, mos_std=mos_std,
    )
    ds_val = VQADataset(
        rows_val, args.db_path,
        clip_len=args.clip_len, size=args.resize, win=args.win, win_step=args.win_step,
        mos_mean=mos_mean, mos_std=mos_std,
    )
    ds_test = VQADataset(
        rows_test, args.db_path,
        clip_len=args.clip_len, size=args.resize, win=args.win, win_step=args.win_step,
        mos_mean=mos_mean, mos_std=mos_std,
    )

    pin = str(device).startswith("cuda")
    loader_train = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=False,
    )
    loader_val = torch.utils.data.DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin, drop_last=False,
    )
    loader_test = torch.utils.data.DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin, drop_last=False,
    )
    return loader_train, loader_val, loader_test


def main():
    ap = argparse.ArgumentParser()
    # ----- mode -----
    ap.add_argument("--mode", choices=["finetune", "test_only"], required=True)
    ap.add_argument("--pretrained", default="/home/xinyi/Project/FD-VQA/src/checkpoints/kvq/qd_model.best.pt", help="pretrain model path")
    # ----- data -----
    ap.add_argument("--csv_path", default="/home/xinyi/Project/FD-VQA/metadata/SHORTS-SDR-DATASET_metadata.csv")
    ap.add_argument("--db_path", default="/media/xinyi/server/video_dataset/shorts-hdr-dataset/sdr/")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--val_ratio", type=float, default=0.1)  # train 80%
    # ----- video processing -----
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--win", type=int, default=6)
    ap.add_argument("--win_step", type=int, default=1)
    # ----- runtime -----
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no_amp", action="store_true")
    # ----- finetune hyperparams -----
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--warmup_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--finetune_lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--clip_unfreeze_blocks", type=int, default=4)
    ap.add_argument("--finetune_last_stage", action="store_true")
    ap.add_argument("--patience", type=int, default=6)
    # ----- save -----
    ap.add_argument("--save_dir", type=str, default="checkpoints_transfer")
    ap.add_argument("--save_name", type=str, default="transfer.pt")

    # ----- MOS normalization control -----
    # finetune: use target train mean/std
    ap.add_argument("--test_only_norm", choices=["none", "use_source_ckpt"], default="use_source_ckpt")

    args = ap.parse_args()
    torch.manual_seed(args.split_seed)
    device = torch.device(args.device)
    amp = not bool(args.no_amp)

    # ----------------------------
    # Load target rows and split
    # ----------------------------
    rows = read_vid_mos_csv(args.csv_path)
    if args.mode == "finetune":
        csv_path = Path(args.csv_path)
        if csv_path.name == "KVQ_TRAIN_metadata.csv":
            # KVQ challenge split
            val_csv = csv_path.parent / "KVQ_VAL_metadata.csv"
            test_csv = csv_path.parent / "KVQ_TEST_metadata.csv"
            train_rows = read_vid_mos_csv(str(csv_path))
            val_rows = read_vid_mos_csv(str(val_csv))
            test_rows = read_vid_mos_csv(str(test_csv))
            print(f"[KVQ split] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
        else:
            train_rows, val_rows, test_rows = split_rows(
                rows, seed=args.split_seed, test_ratio=args.test_ratio, val_ratio=args.val_ratio
            )
        mos_train = np.array([mos for _vid, mos in train_rows], dtype=np.float32)
        mos_mean = float(mos_train.mean()) if len(mos_train) else 0.0
        mos_std = float(mos_train.std()) if len(mos_train) else 1.0
        if mos_std <= 1e-8:
            mos_std = 1.0
    else:
        # test_only
        train_rows, val_rows, test_rows = [], [], rows
        mos_mean, mos_std = None, None

    # ----------------------------
    # Build model, load pretrained
    # ----------------------------
    model = QD_MODEL(
        clip_model="openai/clip-vit-base-patch16",
    ).to(device)

    ckpt = load_pretrained_weights(model, args.pretrained, device)
    print(f"Loaded pretrained: {args.pretrained}")

    # ----------------------------
    # Decide normalization for test_only
    # ----------------------------
    if args.mode == "test_only":
        if args.test_only_norm == "use_source_ckpt" and isinstance(ckpt, dict):
            # 注：这样对 SRCC/PLCC 没影响
            mos_mean = ckpt.get("mos_mean", None)
            mos_std = ckpt.get("mos_std", None)
            if mos_mean is None or mos_std is None:
                mos_mean, mos_std = None, None
                print("[warn] pretrained ckpt has no mos_mean/std, fallback to no normalization.")
            else:
                print(f"test_only uses source mos_mean/std from ckpt: mean={mos_mean:.4f}, std={mos_std:.4f}")
        else:
            print("test_only uses no MOS normalization.")

    # ----------------------------
    # Mode: test_only
    # ----------------------------
    if args.mode == "test_only":
        ds_test = VQADataset(
            test_rows, args.db_path,
            clip_len=args.clip_len, size=args.resize, win=args.win, win_step=args.win_step,
            mos_mean=mos_mean, mos_std=mos_std,
        )
        pin = str(device).startswith("cuda")
        loader_test = torch.utils.data.DataLoader(
            ds_test, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=pin, drop_last=False,
        )
        print("num_test_rows =", len(test_rows))
        print("len(ds_test)  =", len(ds_test))
        print("len(loader_test) =", len(loader_test), "batch_size =", args.batch_size)

        te_loss, te_plcc, te_srcc, te_rmse = run_epoch(
            model, loader_test, device,
            optim=None, amp=amp, mos_mean=mos_mean, mos_std=mos_std,
            desc="TestOnly", show_pbar=True
        )
        print(f"TEST_ONLY | loss={te_loss:.4f} plcc={te_plcc:.4f} srcc={te_srcc:.4f} rmse={te_rmse:.4f}")
        return

    # ----------------------------
    # DataLoaders
    # ----------------------------
    loader_train, loader_val, loader_test = make_loaders(
        train_rows, val_rows, test_rows, args,
        mos_mean=mos_mean, mos_std=mos_std, device=device
    )

    # ----------------------------
    # Mode: finetune
    # ----------------------------
    model.freeze_clip_all()
    did_unfreeze = False

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

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    last_path = save_dir / args.save_name
    best_path = save_dir / (Path(args.save_name).stem + ".best.pt")
    best_weights_path = save_dir / (Path(args.save_name).stem + ".best_weights.pt")

    best_val_srcc = -1e18
    bad_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        if (not did_unfreeze) and bool(args.finetune_last_stage) and epoch == (int(args.warmup_epochs) + 1):
            model.unfreeze_clip_last_blocks(n_blocks=int(args.clip_unfreeze_blocks), also_unfreeze_ln=True)
            did_unfreeze = True
            print(f"[Finetune] Unfroze CLIP last {int(args.clip_unfreeze_blocks)} blocks")

        tr_loss, tr_plcc, tr_srcc, tr_rmse = run_epoch(
            model, loader_train, device,
            optim=optim, amp=amp, mos_mean=mos_mean, mos_std=mos_std,
            desc=f"FT Train e{epoch:03d}", show_pbar=True
        )
        va_loss, va_plcc, va_srcc, va_rmse = run_epoch(
            model, loader_val, device,
            optim=None, amp=amp, mos_mean=mos_mean, mos_std=mos_std,
            desc=f"FT Val   e{epoch:03d}", show_pbar=True
        )
        scheduler.step()

        print(
            f"epoch {epoch:03d} | "
            f"train: loss={tr_loss:.4f} plcc={tr_plcc:.4f} srcc={tr_srcc:.4f} rmse={tr_rmse:.4f} | "
            f"val: loss={va_loss:.4f} plcc={va_plcc:.4f} srcc={va_srcc:.4f} rmse={va_rmse:.4f}"
        )

        ckpt_out = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "mos_mean": mos_mean,
            "mos_std": mos_std,
            "args": vars(args),
            "best_val_srcc": best_val_srcc,
        }
        torch.save(ckpt_out, str(last_path))

        if va_srcc > best_val_srcc:
            best_val_srcc = va_srcc
            bad_epochs = 0
            ckpt_out["best_val_srcc"] = best_val_srcc
            torch.save(ckpt_out, str(best_path))
            torch.save(model.state_dict(), str(best_weights_path))
            print(f"  [best] val_srcc={best_val_srcc:.4f} -> saved {best_weights_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                print(f"[EarlyStop] val_srcc not improved for {bad_epochs} epochs. Stop.")
                break

    # load best and test
    if best_weights_path.exists():
        sd = torch.load(str(best_weights_path), map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=True)
        print(f"Loaded best weights: {best_weights_path}")

    te_loss, te_plcc, te_srcc, te_rmse = run_epoch(
        model, loader_test, device,
        optim=None, amp=amp, mos_mean=mos_mean, mos_std=mos_std,
        desc="FT Test", show_pbar=True
    )
    print(f"FINETUNE TEST | loss={te_loss:.4f} plcc={te_plcc:.4f} srcc={te_srcc:.4f} rmse={te_rmse:.4f}")


if __name__ == "__main__":
    main()