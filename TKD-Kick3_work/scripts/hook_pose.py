# -*- coding: utf-8 -*-
"""
hook_pose.py
Pose extraction for punches (right hook), built on top of extract_pose_mediapipe.py.

extract_pose_mediapipe.py (unchanged) picks the performer and crops the action window
from ANKLE motion, which is right for kicks but crops punches at random. This module
reuses its PoseLandmarker tracking and adds:

- a WRIST-based performer selection + action window (punch view)
- the original ANKLE-based selection + window, reproduced 1:1 (kick view), so a single
  MediaPipe pass can feed both the kick and the punch pipelines at inference time.

Output npz keys match extract_pose_mediapipe.py (kpts, vis, fps, src_video,
invalid_for_scoring, reason, win_s, win_e, ...) plus window_mode / attack_wrist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from extract_pose_mediapipe import (  # noqa: F401  (re-exported for callers)
    LH, LS, LW, RH, RS, RW, VTH, VIDEO_EXTS, Track,
    _ankle_side_indices, _center_of_pose, _choose_kicker_track, _compute_rel_speed_window,
    _match_tracks, _mp_landmarks_to_13, build_landmarker,
)


# ----------------------------
# Tracking (same loop as extract_pose_mediapipe.extract_one_video)
# ----------------------------
def track_video(landmarker, video_path: Path, num_poses: int = 4,
                max_match_dist_ratio: float = 0.15) -> Tuple[List[Track], float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_dist = max_match_dist_ratio * max(W, H)

    tracks: List[Track] = []
    t = 0
    prev_ts = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts = int(round(t * 1000.0 / fps))
        if ts <= prev_ts:
            ts = prev_ts + 1
        prev_ts = ts

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        det_k, det_v, det_c = [], [], []
        if result and result.pose_landmarks:
            for lm33 in result.pose_landmarks[:num_poses]:
                k13, v13 = _mp_landmarks_to_13(lm33, W, H)
                det_k.append(k13)
                det_v.append(v13)
                det_c.append(_center_of_pose(k13, v13).astype(np.float32))

        assign, new_det = _match_tracks(tracks, det_c, max_dist=max_dist)

        for tr in tracks:
            tr.kpts_list.append(np.full((13, 2), np.nan, np.float32))
            tr.vis_list.append(np.zeros((13,), np.float32))
            tr.center_list.append(np.array([np.nan, np.nan], np.float32))
            tr.present_list.append(False)

        for ti, di in assign.items():
            tr = tracks[ti]
            tr.kpts_list[-1] = det_k[di]
            tr.vis_list[-1] = det_v[di]
            tr.center_list[-1] = det_c[di]
            tr.present_list[-1] = True
            tr.last_center = det_c[di]
            tr.n_present += 1

        for di in new_det:
            tr = Track(
                kpts_list=[np.full((13, 2), np.nan, np.float32) for _ in range(t)] + [det_k[di]],
                vis_list=[np.zeros((13,), np.float32) for _ in range(t)] + [det_v[di]],
                center_list=[np.array([np.nan, np.nan], np.float32) for _ in range(t)] + [det_c[di]],
                present_list=[False] * t + [True],
                last_center=det_c[di],
                n_present=1,
            )
            tracks.append(tr)

        t += 1

    cap.release()
    return tracks, fps, W, H, t


def _track_arrays(tr: Track) -> Tuple[np.ndarray, np.ndarray]:
    return np.stack(tr.kpts_list, 0).astype(np.float32), np.stack(tr.vis_list, 0).astype(np.float32)


# ----------------------------
# Punch view: wrist-based window + performer selection
# ----------------------------
def _upper_center(kpts: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Shoulder center (fallback hip center) per frame; NaN where unavailable."""
    T = kpts.shape[0]
    c = np.full((T, 2), np.nan, np.float32)
    ms = (vis[:, LS] > VTH) & (vis[:, RS] > VTH)
    c[ms] = 0.5 * (kpts[ms, LS] + kpts[ms, RS])
    mh = (~ms) & (vis[:, LH] > VTH) & (vis[:, RH] > VTH)
    c[mh] = 0.5 * (kpts[mh, LH] + kpts[mh, RH])
    return c


def _rel_wrist(kpts: np.ndarray, vis: np.ndarray, W_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    c = _upper_center(kpts, vis)
    w = kpts[:, W_idx, :]
    valid = (vis[:, W_idx] > VTH) & np.all(np.isfinite(w), axis=1) & np.all(np.isfinite(c), axis=1)
    rel = np.zeros_like(w)
    rel[valid] = w[valid] - c[valid]
    return rel, valid


def attack_wrist_index(kpts: np.ndarray, vis: np.ndarray) -> int:
    """Wrist with the longer path relative to the shoulder center (RW on ties)."""
    def path_len(j):
        rel, valid = _rel_wrist(kpts, vis, j)
        if rel.shape[0] <= 1:
            return 0.0
        step = np.linalg.norm(np.diff(rel, axis=0), axis=1)
        return float(np.sum(step[valid[1:] & valid[:-1]]))

    return RW if path_len(RW) >= path_len(LW) else LW


def compute_wrist_window(kpts: np.ndarray, vis: np.ndarray, fps: float,
                         min_dur_s: float = 0.6, max_dur_s: float = 2.0,
                         wrist: int | None = None) -> Tuple[int, int, int]:
    """
    Same logic as extract_pose_mediapipe._compute_rel_speed_window, but driven by the
    attacking wrist relative to the shoulder center. Returns (win_s, win_e, wrist_idx).
    wrist: force RW/LW (known label); None = pick the more active wrist.
    """
    T = kpts.shape[0]
    Wi = attack_wrist_index(kpts, vis) if wrist is None else wrist
    if T <= 2:
        return 0, max(0, T - 1), Wi

    rel, valid = _rel_wrist(kpts, vis, Wi)
    sp = np.zeros((T,), np.float32)
    step = np.linalg.norm(np.diff(rel, axis=0), axis=1)
    step[~(valid[1:] & valid[:-1])] = 0.0
    sp[1:] = step

    win = 7
    sp_s = np.convolve(sp, np.ones((win,), np.float32) / win, mode="same") if T >= win else sp

    peak = int(np.argmax(sp_s))
    q85 = float(np.quantile(sp_s, 0.85)) if np.any(sp_s > 0) else 0.0
    thr_hi = max(q85, 2.0)
    thr_lo = max(thr_hi * 0.55, 1.0)

    s = e = peak
    while s - 1 >= 0 and sp_s[s - 1] >= thr_lo:
        s -= 1
    while e + 1 < T and sp_s[e + 1] >= thr_lo:
        e += 1

    min_len = max(8, int(round(min_dur_s * fps)))
    max_len = max(8, int(round(max_dur_s * fps)))

    if e - s + 1 < min_len:
        # grow symmetrically, then slide inward instead of truncating at clip borders
        need = min_len - (e - s + 1)
        s -= need // 2
        e += need - need // 2
        if s < 0:
            e, s = e - s, 0
        if e > T - 1:
            s, e = max(0, s - (e - (T - 1))), T - 1
    if e - s + 1 > max_len:
        s = max(0, peak - max_len // 2)
        e = min(T - 1, s + max_len - 1)
        s = max(0, e - max_len + 1)

    return int(s), int(e), int(Wi)


def choose_puncher_track(tracks: List[Track], fps: float, W: int, H: int,
                         unsure_ratio: float = 0.85, wrist: int | None = None) -> Tuple[int, Dict]:
    """Mirror of _choose_kicker_track with wrist motion instead of ankle motion."""
    if not tracks:
        return 0, {"invalid_for_scoring": 1, "reason": "no_pose", "scores": []}

    raw = []
    for tr in tracks:
        k, v = _track_arrays(tr)
        s, e, Wi = compute_wrist_window(k, v, fps, wrist=wrist)
        seg = slice(s, e + 1)
        rel, valid = _rel_wrist(k[seg], v[seg], Wi)
        step = np.linalg.norm(np.diff(rel, axis=0), axis=1) if rel.shape[0] > 1 else np.zeros((0,))
        motion = float(np.sum(step[valid[1:] & valid[:-1]])) if step.size else 0.0
        cov = float(np.mean(v[seg, Wi] > VTH)) if e >= s else 0.0

        c = _upper_center(k[seg], v[seg])
        mc = np.all(np.isfinite(c), axis=1)
        if np.any(mc):
            d = float(np.mean(np.linalg.norm(c[mc] - np.array([W * 0.5, H * 0.5]), axis=1)))
            center_score = 1.0 - min(1.0, d / (0.5 * max(W, H)))
        else:
            center_score = 0.0
        raw.append((motion, cov, center_score))

    motions = np.array([r[0] for r in raw], np.float32)
    mnorm = motions / (float(motions.max()) + 1e-6)
    scores = [0.55 * float(mnorm[i]) + 0.35 * raw[i][1] + 0.10 * raw[i][2] for i in range(len(raw))]

    order = np.argsort(scores)[::-1]
    best = int(order[0])
    best_score = float(scores[best])
    second_score = float(scores[int(order[1])]) if len(order) >= 2 else 0.0

    invalid, reason = 0, ""
    if len(order) >= 2 and best_score > 0 and second_score >= best_score * unsure_ratio:
        invalid = 1
        reason = f"multiple people moving, performer uncertain (best={best_score:.3f}, second={second_score:.3f})"

    return best, {"best_score": best_score, "second_score": second_score, "scores": scores,
                  "invalid_for_scoring": invalid, "reason": reason}


# ----------------------------
# Views
# ----------------------------
def _empty_view(video_path: Path, fps: float, mode: str) -> Dict:
    return dict(kpts=np.zeros((0, 13, 2), np.float32), vis=np.zeros((0, 13), np.float32),
                fps=np.float32(fps), src_video=str(video_path), invalid_for_scoring=np.int32(1),
                reason="no_pose_or_too_short", window_mode=mode)


def punch_view(tracks: List[Track], fps: float, W: int, H: int, n_frames: int, video_path: Path,
               unsure_ratio: float = 0.85, wrist: int | None = None) -> Dict:
    if n_frames <= 1 or not tracks:
        return _empty_view(video_path, fps, "wrist")

    best, dbg = choose_puncher_track(tracks, fps, W, H, unsure_ratio=unsure_ratio, wrist=wrist)
    kpts, vis = _track_arrays(tracks[best])
    s, e, Wi = compute_wrist_window(kpts, vis, fps, wrist=wrist)
    k_win, v_win = kpts[s:e + 1], vis[s:e + 1]
    return dict(
        kpts=k_win, vis=v_win, fps=np.float32(fps), src_video=str(video_path),
        invalid_for_scoring=np.int32(dbg["invalid_for_scoring"]), reason=str(dbg["reason"]),
        win_s=np.int32(s), win_e=np.int32(e), frames_total=np.int32(n_frames),
        cov_best=np.float32(np.mean(v_win[:, Wi] > VTH) if len(v_win) else 0.0),
        best_track=np.int32(best), best_score=np.float32(dbg["best_score"]),
        second_score=np.float32(dbg["second_score"]), scores=np.array(dbg["scores"], np.float32),
        window_mode="wrist", attack_wrist="R" if Wi == RW else "L",
    )


def kick_view(tracks: List[Track], fps: float, W: int, H: int, n_frames: int, video_path: Path,
              unsure_ratio: float = 0.85) -> Dict:
    """Identical post-processing to extract_pose_mediapipe.extract_one_video (ankle window)."""
    if n_frames <= 1 or not tracks:
        return _empty_view(video_path, fps, "ankle")

    best, dbg = _choose_kicker_track(tracks, fps=fps, W=W, H=H, unsure_ratio=unsure_ratio)
    kpts, vis = _track_arrays(tracks[best])
    s, e, _ = _compute_rel_speed_window(kpts, vis, fps=fps)
    k_win, v_win = kpts[s:e + 1], vis[s:e + 1]
    A, *_ = _ankle_side_indices(k_win, v_win)
    return dict(
        kpts=k_win, vis=v_win, fps=np.float32(fps), src_video=str(video_path),
        invalid_for_scoring=np.int32(int(dbg.get("invalid_for_scoring", 0))),
        reason=str(dbg.get("reason", "")),
        win_s=np.int32(s), win_e=np.int32(e), frames_total=np.int32(n_frames),
        cov_best=np.float32(np.mean(v_win[:, A] > VTH) if len(v_win) else 0.0),
        miss_best=np.float32(np.sum(v_win[:, A] <= VTH) / max(1.0, fps)),
        best_track=np.int32(best), best_score=np.float32(dbg.get("best_score", 0.0)),
        second_score=np.float32(dbg.get("second_score", 0.0)),
        scores=np.array(dbg.get("scores", []), np.float32), window_mode="ankle",
    )


def extract_views(task_path: Path, video_path: Path, num_poses: int = 4,
                  unsure_ratio: float = 0.85, wrist: int | None = None) -> Tuple[Dict, Dict]:
    """One MediaPipe pass -> (kick_view, punch_view). wrist: see compute_wrist_window."""
    landmarker = build_landmarker(task_path, num_poses)
    try:
        tracks, fps, W, H, n = track_video(landmarker, video_path, num_poses=num_poses)
    finally:
        landmarker.close()
    return (kick_view(tracks, fps, W, H, n, video_path, unsure_ratio),
            punch_view(tracks, fps, W, H, n, video_path, unsure_ratio, wrist=wrist))


def save_view(view: Dict, out_npz: Path) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **view)
