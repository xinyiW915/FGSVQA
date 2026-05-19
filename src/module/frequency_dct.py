import os
os.environ["DECORD_DUPLICATE_WARNING_THRESHOLD"] = "1.0"
from decord import VideoReader, cpu
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module.read_frame_decord import sample_frames_uniform, collect_needed, cache_needed_frames, read_window_from_cache

# ----------------------------
# utils
# ----------------------------
def norm01(x, eps=1e-6):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + eps)

# def upsample_grid(grid, out_hw, interp=cv2.INTER_LINEAR):
def upsample_grid(grid, out_hw, interp=cv2.INTER_NEAREST):
    H, W = out_hw
    return cv2.resize(grid.astype(np.float32), (W, H), interpolation=interp)

def gaussian_blur(x, sigma=1.0):
    if sigma <= 0:
        return x
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 3:
        ksize = 3
    return cv2.GaussianBlur(x, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT_101)

# sobel operator
def gradient_mag(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    dst = np.sqrt(gx * gx + gy * gy)
    # plt.imshow(dst, cmap='gray')
    # plt.savefig('/home/xinyi/Project/FD-VQA/test_videos/freq_test_dct_only/Sobel_operator_result.jpg', dpi=300)
    return dst

# Sobel magnitude -> normalize -> threshold -> dilate edge region
def edge_sobel(gray, thr=0.20, dilate_px=2):
    g = gradient_mag(gray).astype(np.float32)
    g = norm01(g)  # normalize
    edge = (g >= thr).astype(np.uint8)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        edge = cv2.dilate(edge, k, iterations=1)
    # plt.imshow(edge, cmap='gray')
    # plt.savefig('/home/xinyi/Project/FD-VQA/test_videos/freq_test_dct_only/edge_result.jpg', dpi=300)
    return edge

# per-block fraction of near edge pixels
def block_edge(edge_mask, block=16):
    h, w = edge_mask.shape
    gh, gw = h // block, w // block
    frac = np.zeros((gh, gw), dtype=np.float32)
    for by in range(gh):
        for bx in range(gw):
            y0, x0 = by * block, bx * block
            frac[by, bx] = float(edge_mask[y0:y0 + block, x0:x0 + block].mean())
    # plt.imshow(frac, cmap='gray')
    # plt.savefig('/home/xinyi/Project/FD-VQA/test_videos/freq_test_dct_only/edge_block.jpg', dpi=300)
    return frac

# per-block mean
def block_mean(img, block=16):
    h, w = img.shape
    gh, gw = h // block, w // block
    x = img[:gh*block, :gw*block]
    x = x.reshape(gh, block, gw, block)      # (block_row, inblock_row, block_col, inblock_col)
    return x.mean(axis=(1, 3)).astype(np.float32)

# discontinuities across block boundaries: |I[:, x] - I[:, x-1]| & |I[y, :] - I[y-1, :]|
def blockiness_boundary_map(gray, block_size=16, blur_sigma=1.0):
    h, w = gray.shape
    v = np.zeros((h, w), dtype=np.float32)
    hmap = np.zeros((h, w), dtype=np.float32)

    for x in range(block_size, w, block_size):
        v[:, x] = np.abs(gray[:, x] - gray[:, x - 1])
    for y in range(block_size, h, block_size):
        hmap[y, :] = np.abs(gray[y, :] - gray[y - 1, :])
    b = v + hmap
    if blur_sigma > 0:
        b = gaussian_blur(b, blur_sigma)
    return b

# ----------------------------
# DCT 16x16 energies
# ----------------------------
def dct_energy_ratios(gray, block=16, low_k=4, mid_k=8, eps=1e-6):
    """
    Return per-block energy ratios for frequency bands.
    low:  [0:low_k, 0:low_k]
    mid:  [0:mid_k, 0:mid_k] - low
    high: remaining
    """
    H, W = gray.shape
    gh, gw = H // block, W // block
    r_low = np.zeros((gh, gw), dtype=np.float32)
    r_mid = np.zeros((gh, gw), dtype=np.float32)
    r_high = np.zeros((gh, gw), dtype=np.float32)

    for by in range(gh):
        for bx in range(gw):
            y0, x0 = by * block, bx * block
            patch = gray[y0:y0 + block, x0:x0 + block].astype(np.float32)

            C = cv2.dct(patch)
            E = C * C
            E_low = E[:low_k, :low_k].sum()  # 4 x 4
            E_mid = E[:mid_k, :mid_k].sum() - E_low  # 8×8
            E_total = E.sum()
            E_high = max(E_total - (E_low + E_mid), 0.0)

            denom = E_total + eps
            r_low[by, bx] = E_low / denom
            r_mid[by, bx] = E_mid / denom
            r_high[by, bx] = E_high / denom
    return r_low, r_mid, r_high

# Temporal map via FFT
def temporal_fft_map(gray_seq, *, block=16, hf_start_bin=2, eps=1e-6):
    # each frame -> (K, gh, gw) block mean grid
    grids = [block_mean(g.astype(np.float32), block=block) for g in gray_seq]
    s = np.stack(grids, axis=0)

    # rFFT along time
    X = np.fft.rfft(s, axis=0)  # (F, gh, gw)
    E = (X.real * X.real + X.imag * X.imag).astype(np.float32)  # power spectrum

    # F = K // 2 + 1
    dc = E[0]
    if E.shape[0] <= 1:
        z = np.zeros_like(dc, dtype=np.float32)
        return z, z
    P = E[1:]  # drop DC -> (F-1, gh, gw)
    non_dc = P.sum(axis=0)

    # changes relative to DC
    motion = non_dc / (dc + eps)

    # flicker: change is in high temporal freqs
    start = max(int(hf_start_bin) - 1, 0)  # index in P
    if start >= P.shape[0]:
        flicker = np.zeros_like(non_dc, dtype=np.float32)
    else:
        hi = P[start:].sum(axis=0)
        flicker = hi / (non_dc + eps)
    return motion.astype(np.float32), flicker.astype(np.float32)

def fuse_temporal_maps(motion_grid, flicker_grid, *, beta=0.5):
    m = norm01(motion_grid)
    f = np.clip(flicker_grid, 0.0, 1.0)
    # boosts where flicker is high
    w = m * ((1.0 - beta) + beta * f)
    return norm01(w)

# ----------------------------
# DCT -> two stream weights
# ----------------------------
def compute_twostream_dct(
    gray_seq,
    *,
    block=16,
):
    K = len(gray_seq)
    gray_anchor = gray_seq[0]
    H, W = gray_anchor.shape

    r_low_stack, r_mid_stack, r_high_stack = [], [], []
    for g in gray_seq:
        r_low, r_mid, r_high = dct_energy_ratios(g, block=block)
        r_low_stack.append(r_low)  # (gh, gw)
        r_mid_stack.append(r_mid)
        r_high_stack.append(r_high)
    r_low_stack = np.stack(r_low_stack, axis=0)  # (K, gh, gw)
    r_mid_stack = np.stack(r_mid_stack, axis=0)
    r_high_stack = np.stack(r_high_stack, axis=0)

    # frequency band (anchor frame)
    anchor_low_grid = r_low_stack[0]  # (gh, gw)
    anchor_mid_grid = r_mid_stack[0]
    anchor_high_grid = r_high_stack[0]

    # Ringing map (anchor)
    edge_mask = edge_sobel(gray_anchor)     # around edges
    edge_frac = block_edge(edge_mask, block=block)
    mh_band = r_mid_stack[0] + r_high_stack[0]  # mid/high frequency energy
    ring_score = np.maximum(mh_band, 0.0)       # score = mid_high - 0 * r_low # alpha = 0
    edge_min_frac = 0.05
    ringing_grid = np.where(edge_frac >= edge_min_frac, edge_frac * ring_score, 0.0).astype(np.float32)
    s = np.percentile(ringing_grid, 99) + 1e-6
    ringing_grid01 = np.clip(ringing_grid / s, 0.0, 1.0)

    # Blur map (anchor): like low-pass filtering
    hf = 0.5 * r_mid_stack[0] + 1.0 * r_high_stack[0]
    blur_raw = np.clip(1.0 - hf, 0.0, 1.0)
    sobel_g = gradient_mag(gray_anchor).astype(np.float32)
    sobel_g_grid = block_mean(sobel_g, block=block)
    sobel_g_grid = norm01(sobel_g_grid)  # soft structure weight
    blur_grid = np.clip(blur_raw * sobel_g_grid, 0.0, 1.0)

    # Blockiness map (anchor): boundary discontinuities
    boundary_pix = blockiness_boundary_map(gray_anchor, block_size=block)
    blockiness_grid = norm01(block_mean(boundary_pix, block=block))

    # Temporal (window): FFT along time
    if K >= 4:
        motion_grid, flick_grid = temporal_fft_map(gray_seq, block=block, hf_start_bin=2)
        temporal_grid = fuse_temporal_maps(motion_grid, flick_grid, beta=0.5)
    elif K == 2:
        E_stack = norm01(r_mid_stack + r_high_stack)
        temporal_grid = norm01(np.abs(E_stack[1] - E_stack[0]))

    # -------------Combine---------------
    w_art = norm01(1.0 * ringing_grid01 + 1.0 * blur_grid + 1.0 * blockiness_grid + 1.0 * temporal_grid)
    w_str = 1.0 - w_art

    debug = {
        # frequency band (anchor)
        "dct_low_grid": anchor_low_grid,
        "dct_mid_grid": anchor_mid_grid,
        "dct_high_grid": anchor_high_grid,
        # ringing (anchor)
        "ringing_grid": ringing_grid01,
        "edge_px": edge_mask,
        # blur (anchor)
        "blur_grid": blur_grid,
        # blockiness (anchor)
        "blockiness_grid": blockiness_grid,
        # temporal (window)
        "temporal_grid": temporal_grid,
    }
    return w_art, w_str, debug

# ----------------------------
# visualization panel
# ----------------------------
def save_panel(out_png, frame_rgb, w_art, w_str, debug):
    fig = plt.figure(figsize=(16, 9), dpi=160)
    def add(ax_i, title, img, cmap=None, vmin=0, vmax=1):
        ax = fig.add_subplot(3, 4, ax_i)
        ax.set_title(title)
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")

    add(1, "Frame_t (anchor)", frame_rgb)
    add(2, "DCT LOW (grid)",  norm01(debug["dct_low_grid"]),  cmap="viridis")
    add(3, "DCT MID (grid)",  norm01(debug["dct_mid_grid"]),  cmap="viridis")
    add(4, "DCT HIGH (grid)", norm01(debug["dct_high_grid"]), cmap="viridis")
    # ringing
    add(5, "EDGE (mask)", debug["edge_px"], cmap="viridis")
    add(6, "RINGING (mid/high, grid)", debug["ringing_grid"], cmap="viridis")
    # blur
    add(7, "BLUR (lowpass, grid)", debug["blur_grid"], cmap="viridis")
    # blockiness
    add(8, "BLOCKINESS (boundary, grid)", debug["blockiness_grid"], cmap="viridis")
    # temporal
    add(9, "TEMPORAL (grid)", debug["temporal_grid"], cmap="viridis")
    # all map
    add(10, "W_art", w_art, cmap="viridis")
    add(11, "W_str", w_str, cmap="viridis")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", default="/home/xinyi/Project/FD-VQA/test_videos/SDR_Animal_5ngj.mp4")
    parser.add_argument("--out_dir", default="/home/xinyi/Project/FD-VQA/test_videos/freq_test_dct_only")
    parser.add_argument("--size", type=int, default=224)

    # fixed-T anchors over whole video (duration-uniform)
    parser.add_argument("--num_anchors", type=int, default=16)
    parser.add_argument("--win", type=int, default=6)
    parser.add_argument("--win_step", type=int, default=1)
    parser.add_argument("--block", type=int, default=16)

    parser.add_argument("--no_panel", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    vr = VideoReader(args.video, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames <= 1:
        raise RuntimeError(f"Video too short / frame count unavailable: total_frames={total_frames}")
    print("total_frames:", total_frames)

    size = args.size
    win = args.win
    win_step = args.win_step
    num_anchors = args.num_anchors

    anchor_idxs = sample_frames_uniform(total_frames, num_anchors, win=win, win_step=win_step)
    needed = collect_needed(anchor_idxs, total_frames, win, win_step)
    print("anchor_idxs:", anchor_idxs)
    cache = cache_needed_frames(vr, needed, size)
    print("cached:", len(cache), "needed:", len(needed))

    frame_all, w_art_all, w_str_all = [], [], []
    anchors_kept = []
    image_idx = 0

    for anchor in tqdm(anchor_idxs, desc="Processing anchors (DCT)"):
        out = read_window_from_cache(cache, anchor, total_frames, win, win_step)
        if out is None:
            continue
        anchor_frame, gray_seq, idxs = out

        w_art, w_str, dbg = compute_twostream_dct(
            gray_seq,
            block=args.block,
        )

        frame_all.append(anchor_frame)
        w_art_all.append(w_art)
        w_str_all.append(w_str)
        anchors_kept.append(idxs)

        image_idx += 1
        if not args.no_panel:
            save_panel(
                os.path.join(args.out_dir, f"anchor_{anchor:03d}_{image_idx:02d}.png"),
                anchor_frame,
                w_art,
                w_str,
                dbg,
            )

    print(f"Done. Outputs saved to: {args.out_dir}")
    print(anchors_kept)
    print(f"total_frames={total_frames}, num_anchors_target={num_anchors}, anchors_produced={len(w_str_all)}")

if __name__ == "__main__":
    main()