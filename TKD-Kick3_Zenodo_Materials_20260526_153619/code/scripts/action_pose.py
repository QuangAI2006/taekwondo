# scripts/action_pose.py
"""
Trích xuất pose dùng chung cho MỌI động tác (đá, đấm, ... và động tác train thêm).

1. extract_pose(video)   : MediaPipe PoseLandmarker đa người + ghép track (giống extract_pose_mediapipe.py),
                           giữ NGUYÊN video (không cắt), lưu thêm tọa độ 3D (world). Chọn người làm động tác
                           theo chuyển động của cổ tay + cổ chân (hoặc chỉ cổ tay / cổ chân).
2. action_window(...)    : cắt đoạn đưa vào model nhận dạng.
   - "ankle" : đúng hàm _compute_rel_speed_window của tác giả (cổ chân so với hông) -> tương thích 100%
               với checkpoint TKD-Kick3 (mặc định khi checkpoint không ghi window_mode).
   - "motion": đoạn chuyển động mạnh nhất của cả cơ thể (4 đầu chi) — không cần biết trước là đá hay
               đấm; mặc định cho model tự train (train_actions.py) có nhiều loại động tác.
   - "wrist" : thuật toán của tác giả áp cho cổ tay so với vai.
   - "none"  : lấy cả video.

Định dạng 13 khớp: HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA (như extract_pose_mediapipe.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from extract_pose_mediapipe import (
    LA, LH, LS, LW, MP33_TO_13, RA, RH, RS, RW, VTH,
    _center_of_pose, _compute_rel_speed_window, _match_tracks, build_landmarker,
)

WINDOW_MODES = ("ankle", "wrist", "motion", "none")


@dataclass
class _Track:
    kpts: List[np.ndarray] = field(default_factory=list)   # (13,2) pixel
    vis: List[np.ndarray] = field(default_factory=list)    # (13,)
    world: List[np.ndarray] = field(default_factory=list)  # (13,3) mét, gốc ở hông
    last_center: Optional[np.ndarray] = None
    n_present: int = 0

    def pad(self, n: int) -> None:
        for _ in range(n):
            self.kpts.append(np.full((13, 2), np.nan, np.float32))
            self.vis.append(np.zeros((13,), np.float32))
            self.world.append(np.full((13, 3), np.nan, np.float32))


def _to_13(landmarks, world_landmarks, w: int, h: int):
    k = np.full((13, 2), np.nan, np.float32)
    v = np.zeros((13,), np.float32)
    wd = np.full((13, 3), np.nan, np.float32)
    for j13, j33 in MP33_TO_13.items():
        lm = landmarks[j33]
        k[j13] = (lm.x * w, lm.y * h)
        if getattr(lm, "visibility", None) is not None:
            v[j13] = float(lm.visibility)
        elif getattr(lm, "presence", None) is not None:
            v[j13] = float(lm.presence)
        else:
            v[j13] = 1.0
        if world_landmarks is not None:
            wl = world_landmarks[j33]
            wd[j13] = (wl.x, wl.y, wl.z)
    return k, v, wd


def _effector_motion(k: np.ndarray, v: np.ndarray, effectors: Tuple[int, int], anchor: Tuple[int, int]) -> float:
    """Tổng quãng đường của 2 đầu chi so với điểm neo (hông/vai), chia chiều dài thân."""
    anc = 0.5 * (k[:, anchor[0]] + k[:, anchor[1]])
    torso = np.linalg.norm(0.5 * (k[:, LS] + k[:, RS]) - 0.5 * (k[:, LH] + k[:, RH]), axis=1)
    L = float(np.nanmedian(torso)) if np.any(np.isfinite(torso)) else 1.0
    tot = 0.0
    for e in effectors:
        rel = k[:, e] - anc
        ok = (v[:, e] > VTH) & np.all(np.isfinite(rel), axis=1)
        if ok.sum() >= 2:
            tot += float(np.nansum(np.linalg.norm(np.diff(rel[ok], axis=0), axis=1)))
    return tot / max(L, 1e-6)


def extract_pose(video_path: Path, task_path: Path, num_poses: int = 4, select: str = "limbs",
                 max_match_dist_ratio: float = 0.15, on_frame: Optional[Callable] = None,
                 start_s: float = 0.0, end_s: Optional[float] = None) -> Dict:
    """
    select: "limbs" (cổ tay + cổ chân), "wrists", "ankles" — tiêu chí chọn người làm động tác.
    on_frame(t, frame_bgr, people_kpts, n_frames): gọi sau mỗi frame (cho cửa sổ pipeline).
    start_s / end_s: chỉ lấy đoạn [start_s, end_s) của video (giây); frame 0 của kết quả = start_s.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int(round(start_s * fps)) if start_s > 0 else 0
    if f0 > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    limit = None if end_s is None else max(0, int(round(end_s * fps)) - f0)
    n_frames = max(0, n_frames - f0) if limit is None else min(limit, max(0, n_frames - f0))
    max_dist = max_match_dist_ratio * max(W, H)

    landmarker = build_landmarker(task_path, num_poses)
    tracks: List[_Track] = []
    t, prev_ts = 0, -1
    try:
        while limit is None or t < limit:
            ok, frame = cap.read()
            if not ok:
                break
            ts = max(int(round(t * 1000.0 / fps)), prev_ts + 1)
            prev_ts = ts
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(img, ts)

            dets = []
            if res and res.pose_landmarks:
                for i, lm in enumerate(res.pose_landmarks[:num_poses]):
                    wl = res.pose_world_landmarks[i] if res.pose_world_landmarks else None
                    k, v, wd = _to_13(lm, wl, W, H)
                    dets.append((k, v, wd, _center_of_pose(k, v)))

            assign, new_det = _match_tracks(tracks, [d[3] for d in dets], max_dist=max_dist)
            for tr in tracks:
                tr.pad(1)
            for ti, di in assign.items():
                k, v, wd, c = dets[di]
                tr = tracks[ti]
                tr.kpts[-1], tr.vis[-1], tr.world[-1] = k, v, wd
                tr.last_center = c
                tr.n_present += 1
            for di in new_det:
                k, v, wd, c = dets[di]
                tr = _Track()
                tr.pad(t)
                tr.kpts.append(k)
                tr.vis.append(v)
                tr.world.append(wd)
                tr.last_center, tr.n_present = c, 1
                tracks.append(tr)
            if on_frame is not None:
                on_frame(t, frame, [d[0] for d in dets], n_frames)
            t += 1
    finally:
        cap.release()
        landmarker.close()

    empty = {"kpts": np.zeros((0, 13, 2), np.float32), "vis": np.zeros((0, 13), np.float32),
             "world": np.zeros((0, 13, 3), np.float32), "fps": np.float32(fps), "width": W, "height": H,
             "n_tracks": 0, "multi_person": 0, "invalid_for_scoring": 1}
    if not tracks:
        return empty

    scores = []
    for tr in tracks:
        k, v = np.stack(tr.kpts), np.stack(tr.vis)
        m = 0.0
        if select in ("limbs", "wrists"):
            m += _effector_motion(k, v, (LW, RW), (LS, RS))
        if select in ("limbs", "ankles"):
            m += _effector_motion(k, v, (LA, RA), (LH, RH))
        scores.append(m * (tr.n_present / max(1, t)))
    order = np.argsort(scores)[::-1]
    best = tracks[int(order[0])]
    unsure = len(order) >= 2 and scores[int(order[1])] >= 0.85 * scores[int(order[0])] > 0

    return {
        "kpts": np.stack(best.kpts).astype(np.float32),
        "vis": np.stack(best.vis).astype(np.float32),
        "world": np.stack(best.world).astype(np.float32),
        "fps": np.float32(fps), "width": W, "height": H,
        "n_tracks": len(tracks), "multi_person": int(len(tracks) > 1),
        "invalid_for_scoring": int(unsure),
    }


# ---------- cắt đoạn cho model nhận dạng ----------
def _wrist_as_ankle(kpts: np.ndarray, vis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Đổi chỗ cổ tay->cổ chân, vai->hông để dùng lại nguyên thuật toán cửa sổ của tác giả cho tay."""
    k, v = kpts.copy(), vis.copy()
    for dst, src in ((LA, LW), (RA, RW), (LH, LS), (RH, RS)):
        k[:, dst], v[:, dst] = kpts[:, src], vis[:, src]
    return k, v


def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.zeros_like(x)
    t = np.arange(len(x))
    return np.interp(t, t[ok], x[ok])


def motion_energy(kpts: np.ndarray, vis: np.ndarray, fps: float) -> np.ndarray:
    """Tốc độ lớn nhất của 4 đầu chi (cổ tay so với vai, cổ chân so với hông) mỗi frame, đơn vị thân/giây."""
    T = kpts.shape[0]
    torso = np.linalg.norm(0.5 * (kpts[:, LS] + kpts[:, RS]) - 0.5 * (kpts[:, LH] + kpts[:, RH]), axis=1)
    L = float(np.nanmedian(torso)) if np.any(np.isfinite(torso)) else 1.0
    energy = np.zeros(T, np.float64)
    for effs, anchor in (((LW, RW), (LS, RS)), ((LA, RA), (LH, RH))):
        anc = 0.5 * (kpts[:, anchor[0]] + kpts[:, anchor[1]])
        for e in effs:
            rel = (kpts[:, e] - anc) / max(L, 1e-6)
            rel[vis[:, e] <= VTH] = np.nan  # frame bị che -> nội suy, không coi là đứng yên
            rel = np.stack([_interp_nan_1d(rel[:, c]) for c in range(2)], axis=1)
            sp = np.linalg.norm(np.diff(rel, axis=0), axis=1) * fps
            energy = np.maximum(energy, np.r_[sp[:1], sp] if sp.size else np.zeros(T))
    if T >= 5:
        energy = np.convolve(energy, np.ones(5) / 5, mode="same")
    return energy


def motion_window(kpts: np.ndarray, vis: np.ndarray, fps: float, rel_thr: float = 0.08,
                  pad_s: float = 0.4, min_s: float = 0.6, max_s: float = 3.0) -> Tuple[int, int]:
    """Đoạn chuyển động chính của CẢ cơ thể (không cần biết là đá hay đấm) + đệm 0.4s mỗi đầu.
    Ngưỡng 8% đỉnh: giữ ~70% đoạn cắt của tác giả với đá (cả pha nâng gối, thu chân) và trọn cú đấm."""
    T = kpts.shape[0]
    en = motion_energy(kpts, vis, fps)
    peak = int(np.argmax(en))
    thr = rel_thr * float(en[peak])
    s = e = peak
    while s > 0 and en[s - 1] >= thr:
        s -= 1
    while e < T - 1 and en[e + 1] >= thr:
        e += 1
    pad = int(round(pad_s * fps))
    s, e = max(0, s - pad), min(T - 1, e + pad)
    min_len, max_len = int(round(min_s * fps)), int(round(max_s * fps))
    if e - s + 1 < min_len:
        need = min_len - (e - s + 1)
        s, e = max(0, s - need // 2), min(T - 1, e + need - need // 2)
    if e - s + 1 > max_len:
        s = max(0, min(peak - max_len // 2, T - max_len))
        e = min(T - 1, s + max_len - 1)
    return int(s), int(e)


def action_window(kpts: np.ndarray, vis: np.ndarray, fps: float, mode: str = "motion") -> Tuple[int, int, str]:
    """Trả về (s, e, mode) — đoạn [s, e] (gồm cả e) đưa vào model nhận dạng."""
    T = kpts.shape[0]
    if T == 0:
        return 0, -1, mode
    if mode == "none":
        return 0, T - 1, mode
    if mode == "motion":
        s, e = motion_window(kpts, vis, fps)
    elif mode == "ankle":
        s, e, _ = _compute_rel_speed_window(kpts, vis, fps)
    elif mode == "wrist":
        k, v = _wrist_as_ankle(kpts, vis)
        s, e, _ = _compute_rel_speed_window(k, v, fps)
    else:
        raise ValueError(f"window mode phải thuộc {WINDOW_MODES}, nhận '{mode}'")
    return int(s), int(e), mode
