import numpy as np
import csv
from pathlib import Path
import os
os.environ["DECORD_DUPLICATE_WARNING_THRESHOLD"] = "1.0"
from decord import VideoReader, cpu

from module.frequency_dct import compute_twostream_dct
from module.read_frame_decord import (
    sample_frames_uniform,
    collect_needed,
    cache_needed_frames,
    read_window_from_cache,
)

def process_video(
    vr,
    # resize
    size=224,
    # anchors + window
    num_anchors=16,
    win=6,
    win_step=1,
    # DCT / weights
    block=16,
):
    """
    Returns (frame_all, w_art_all, w_str_all, anchors_kept).
    frame_all: anchor frames (RGB uint8, HxWx3)
    w_art_all/w_str_all: maps (float32, HxW)
    anchors_kept: frame indices used per anchor - list[list[int]]
    """
    total_frames = len(vr)
    if total_frames <= 1:
        raise RuntimeError(f"Video too short / invalid frame count: {total_frames}")

    anchor_idxs = sample_frames_uniform(total_frames, num_anchors, win=win, win_step=win_step)
    needed = collect_needed(anchor_idxs, total_frames, win, win_step)
    cache = cache_needed_frames(vr, needed, size)

    frame_all, w_art_all, w_str_all = [], [], []
    anchors_kept = []

    for anchor in anchor_idxs:
        out = read_window_from_cache(cache, anchor, total_frames, win, win_step)
        if out is None:
            continue

        anchor_frame, gray_seq, idxs = out

        w_art, w_str, _dbg = compute_twostream_dct(
            gray_seq,
            block=block,
        )

        frame_all.append(anchor_frame)
        w_art_all.append(w_art.astype(np.float32, copy=False))
        w_str_all.append(w_str.astype(np.float32, copy=False))
        anchors_kept.append([int(x) for x in idxs])

    return frame_all, w_art_all, w_str_all, anchors_kept


def compute_weight_map(frame_all, w_art_all, w_str_all):
    if len(frame_all) == 0:
        raise ValueError("No frames produced.")
    if not (len(frame_all) == len(w_art_all) == len(w_str_all)):
        raise ValueError(
            f"Length mismatch: frames={len(frame_all)}, w_art_all={len(w_art_all)}, w_str_all={len(w_str_all)}"
        )
    frames_np = np.stack(frame_all, axis=0)  # (N,H,W,3) uint8
    w_art_np = np.stack(w_art_all, axis=0)         # (N,H,W) float32
    w_str_np = np.stack(w_str_all, axis=0)         # (N,H,W) float32
    return frames_np, w_art_np, w_str_np

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

def rng_rows(rows, seed=0):
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    train = [rows[i] for i in idx[:]]
    return train

if __name__ == "__main__":
    csv_path = "/home/xinyi/Project/FD-VQA/metadata/TEST_metadata.csv"
    db_path = "/home/xinyi/Project/FD-VQA/test_videos/"
    # video_path = "/home/xinyi/Project/FD-VQA/test_videos/NesAirFortressIn4108.37ByTool23.mp4"

    rows = read_vid_mos_csv(str(csv_path))
    train = rng_rows(rows)
    for i in range(len(train)):
        vid, mos = train[i]
        print(vid, mos)
        # get video path
        base_path = Path(db_path) / vid
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
                size=224,
                num_anchors=16,
                win=6,
                win_step=1,
                block=16,
            )
            frames_np, w_art_np, w_str_np = compute_weight_map(frame_all, w_art_all, w_str_all)
            print("frames_np:", frames_np.shape, frames_np.dtype)
            print("w_art_np:", w_art_np.shape, w_art_np.dtype)
            print("w_str_np:", w_str_np.shape, w_str_np.dtype)
            print("anchors_kept:", len(anchors_kept), "example:", anchors_kept)
        except Exception as e:
            print("\n[DATA ERROR]")
            print("idx:", i)
            print("vid:", vid)
            print("path:", video_path)
            raise
        finally:
            # release decord video reader
            try:
                if vr is not None:
                    del vr
            except Exception:
                pass