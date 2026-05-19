import cv2
import numpy as np
import os
os.environ["DECORD_DUPLICATE_WARNING_THRESHOLD"] = "1.0"
from decord import VideoReader, cpu

def sample_frames_uniform(total_frames, num_anchors, win=8, win_step=1):
    if total_frames <= 0:
        return [0] * num_anchors

    min_win = (win - 1) * win_step + 1
    if total_frames < num_anchors + min_win:
        if total_frames >= num_anchors:
            anchor_idxs = np.linspace(0, total_frames - 1, num_anchors).round().astype(int)
            return anchor_idxs.tolist()
        else:
            anchor_idxs = list(range(total_frames))
            anchor_idxs.extend([total_frames - 1] * (num_anchors - total_frames))
            return anchor_idxs

    max_start = total_frames - min_win
    anchor_idxs = np.linspace(0, max_start, num_anchors).round().astype(int)
    return anchor_idxs.tolist()

# window idx for anchor
def window_indices(anchor, total_frames, win, win_step):
    last = total_frames - 1
    idxs = [anchor + k * win_step for k in range(win)]
    return [i if i < total_frames else last for i in idxs]

# collect idx for all windows
def collect_needed(anchor_idxs, total_frames, win, win_step):
    needed = set()
    for a in anchor_idxs:
        for idx in window_indices(a, total_frames, win, win_step):
            needed.add(idx)
    return needed

# read frame for needed, read video in sequence
def cache_needed_frames(vr, needed, size):
    if not needed:
        return {}
    max_idx = max(needed)
    cache = {}

    last_ok = -1
    for i in range(max_idx + 1):
        try:
            # read next frame
            frame_rgb = vr.next().asnumpy()  # decord frame RGB (H,W,3)
        except StopIteration:
            # video ended early, like cap.read() failed
            print(f"[DEBUG] cap.read() failed at i={i}, last_ok={last_ok}, max_idx={max_idx}")
            print(f"[DEBUG] max_cached={max(cache.keys()) if cache else None}, last_ok={last_ok}")
            break
        except Exception as e:
            # decode error
            print(f"[DEBUG] cap.read() failed at i={i}, last_ok={last_ok}, max_idx={max_idx} | {e}")
            print(f"[DEBUG] max_cached={max(cache.keys()) if cache else None}, last_ok={last_ok}")
            break

        last_ok = i
        if i in needed:
            # frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) # keep same as OpenCV: BGR
            frame = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
            cache[i] = (frame, gray)
    return cache

# each anchor: read window from cache frames
def read_window_from_cache(cache, anchor, total_frames, win, win_step):
    idxs = window_indices(anchor, total_frames, win, win_step)
    # print(f"window idxs: {idxs}")
    if not cache:
        return None

    # fallback = last frame
    last_avail = max(cache.keys())
    fallback = cache[last_avail]

    anchor_frame = None
    gray_seq = []
    for j, idx in enumerate(idxs):
        if idx in cache:
            data = cache[idx]
            fallback = data
        else:
            data = fallback

        frame, gray = data
        if j == 0:              # window [0] is anchor
            anchor_frame = frame
        gray_seq.append(gray)

    if len(gray_seq) == 1:
        gray_seq.append(gray_seq[0])
        idxs.append(idxs[0])
    return anchor_frame, gray_seq, idxs


if __name__ == "__main__":
    video_path = "/home/xinyi/Project/FD-VQA/test_videos/3369925072.mp4"
    print("video:", video_path)

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames <= 1:
        raise RuntimeError(f"Video too short / frame count unavailable: total_frames={total_frames}")
    print("total_frames:", total_frames)

    size = 224
    win = 8
    win_step = 1
    num_anchors = 16

    anchor_idxs = sample_frames_uniform(total_frames, num_anchors, win=win, win_step=win_step)
    needed = collect_needed(anchor_idxs, total_frames, win, win_step)
    print("anchor_idxs:", anchor_idxs)
    print("needed:", needed)

    cache = cache_needed_frames(vr, needed, size)
    print("cached:", len(cache), "needed:", len(needed))

    frames_list, grays_list, processed_list = [], [], []
    for a in anchor_idxs:
        out = read_window_from_cache(cache, a, total_frames, win, win_step)
        if out is None:
            continue
        anchor_frame, gray_seq, idxs = out
        frames_list.append(anchor_frame)
        grays_list.append(gray_seq)
        processed_list.append(idxs)


    print("frames:", len(frames_list))
    print("one frame shape:", frames_list[0].shape)
    print("each window length:", len(grays_list[0]) if grays_list else 0)
    print("window idxs:", processed_list if processed_list else None)