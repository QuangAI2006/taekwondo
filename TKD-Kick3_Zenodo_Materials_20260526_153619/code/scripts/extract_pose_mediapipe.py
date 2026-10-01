# -*- coding: utf-8 -*-
"""
extract_pose_mediapipe.py (PoseLandmarker 多人版：按“踢腿运动强度”选踢腿者)

新增：
- --exclude_classes（默认 side）：抽骨架时跳过指定类别目录（如 side），用于你“只保留 3 类”。

功能：
- 递归扫描 data/videos/<split> 下的视频
- PoseLandmarker(VIDEO) 多人检测（num_poses）
- 通过“脚踝相对髋部的运动强度（速度/轨迹长度）+ 可见度”选出踢腿者
- 自动裁剪到“踢腿窗口”（rel_speed_window），避免踢完走回去拿手机那段污染
- 不确定是谁在踢（两人运动强度接近）时：invalid_for_scoring=1（后续训练/打分可剔除）

输出：
- data/sequences/<split>/<class>/<video>.npz
  包含：kpts(T,13,2), vis(T,13), fps, src_video, invalid_for_scoring, win_s, win_e, cov_best, miss_best, debug...
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


# ====== 路径 ======
BASE_DIR = Path(__file__).resolve().parents[1]
VID_ROOT = BASE_DIR / "data" / "videos"
SEQ_ROOT = BASE_DIR / "data" / "sequences"

# ====== 13点骨架索引（和训练/打分保持一致）======
HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(13)
VTH = 0.2  # 关键点可见性阈值（<VTH 当作缺失）

# MediaPipe 33点 -> 13点映射（2D像素坐标）
# 0:nose, 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips, 25/26 knees, 27/28 ankles
MP33_TO_13 = {
    HEAD: 0,
    LS: 11, RS: 12,
    LE: 13, RE: 14,
    LW: 15, RW: 16,
    LH: 23, RH: 24,
    LK: 25, RK: 26,
    LA: 27, RA: 28,
}

VIDEO_EXTS = [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV"]


@dataclass
class Track:
    """简单人体轨迹：按髋部中心做关联"""
    kpts_list: List[np.ndarray]     # 每帧 (13,2) float32，缺失为 nan
    vis_list: List[np.ndarray]      # 每帧 (13,)  float32
    center_list: List[np.ndarray]   # 每帧 (2,)   float32
    present_list: List[bool]        # 每帧 是否有匹配到检测
    last_center: Optional[np.ndarray] = None
    n_present: int = 0


def _resolve_task_path(task: str) -> Path:
    p = Path(task)
    if p.is_file():
        return p
    cand = [
        BASE_DIR / task,
        BASE_DIR / "models" / task,
        BASE_DIR / "scripts" / task,
        Path.cwd() / task,
    ]
    for c in cand:
        if c.is_file():
            return c
    raise FileNotFoundError(f"找不到 task 模型文件：{task}（可放项目根目录或 models/ 下）")


def _parse_exclude_classes(s: str) -> Set[str]:
    s = (s or "").strip()
    if not s:
        return set()
    return set([x.strip() for x in s.split(",") if x.strip()])


def _get_class_name(video_path: Path, split: str) -> Optional[str]:
    """
    假设结构：data/videos/<split>/<class>/<video>.mp4
    返回 class（比如 front/roundhouse/side/axe）
    """
    try:
        rel = video_path.relative_to(VID_ROOT / split)
    except Exception:
        return None
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return None


def _list_videos(split: str, exclude_classes: Set[str]) -> List[Path]:
    root = VID_ROOT / split
    if not root.exists():
        return []
    vids = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in VIDEO_EXTS:
            continue
        cls = _get_class_name(p, split)
        if cls is not None and cls in exclude_classes:
            continue
        vids.append(p)
    return sorted(vids)


def _video_to_out_npz(video_path: Path, split: str) -> Path:
    rel = video_path.relative_to(VID_ROOT / split)
    return (SEQ_ROOT / split / rel).with_suffix(".npz")


def _mp_landmarks_to_13(landmarks33, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    landmarks33: list[NormalizedLandmark] 长度33
    返回：
      kpts (13,2) 像素坐标
      vis  (13,)  visibility (若没有则用 presence 兜底；再兜底 1.0)
    """
    k = np.full((13, 2), np.nan, np.float32)
    v = np.zeros((13,), np.float32)
    for j13, j33 in MP33_TO_13.items():
        lm = landmarks33[j33]
        x = float(lm.x) * w
        y = float(lm.y) * h
        k[j13, 0] = x
        k[j13, 1] = y
        if hasattr(lm, "visibility") and lm.visibility is not None:
            vv = float(lm.visibility)
        elif hasattr(lm, "presence") and lm.presence is not None:
            vv = float(lm.presence)
        else:
            vv = 1.0
        v[j13] = float(vv)
    return k, v


def _center_of_pose(k13: np.ndarray, v13: np.ndarray) -> np.ndarray:
    """用髋中心优先；否则肩中心；否则全身均值"""
    if v13[LH] > VTH and v13[RH] > VTH:
        return 0.5 * (k13[LH] + k13[RH])
    if v13[LS] > VTH and v13[RS] > VTH:
        return 0.5 * (k13[LS] + k13[RS])
    m = v13 > VTH
    if np.any(m):
        return np.nanmean(k13[m], axis=0).astype(np.float32)
    return np.array([np.nan, np.nan], np.float32)


def _match_tracks(tracks: List[Track], det_centers: List[np.ndarray], max_dist: float) -> Tuple[Dict[int, int], List[int]]:
    if not tracks:
        return {}, list(range(len(det_centers)))

    Tn = len(tracks)
    Dn = len(det_centers)
    if Dn == 0:
        return {}, []

    dist = np.full((Tn, Dn), 1e9, np.float32)
    for ti, tr in enumerate(tracks):
        if tr.last_center is None or np.any(~np.isfinite(tr.last_center)):
            continue
        for di, c in enumerate(det_centers):
            if np.any(~np.isfinite(c)):
                continue
            dist[ti, di] = float(np.linalg.norm(tr.last_center - c))

    pairs = []
    for ti in range(Tn):
        for di in range(Dn):
            pairs.append((dist[ti, di], ti, di))
    pairs.sort(key=lambda x: x[0])

    assign: Dict[int, int] = {}
    used_det = set()
    used_tr = set()
    for d, ti, di in pairs:
        if d > max_dist:
            break
        if ti in used_tr or di in used_det:
            continue
        assign[ti] = di
        used_tr.add(ti)
        used_det.add(di)

    new_det = [di for di in range(Dn) if di not in used_det]
    return assign, new_det


def _ankle_side_indices(kpts: np.ndarray, vis: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    def path_len(p):
        if p.shape[0] <= 1:
            return 0.0
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        return float(np.nansum(d))

    la = kpts[:, LA, :]
    ra = kpts[:, RA, :]
    return (RA, RK, RH, LA, LK, LH) if path_len(ra) >= path_len(la) else (LA, LK, LH, RA, RK, RH)


def _compute_rel_speed_window(kpts: np.ndarray, vis: np.ndarray, fps: float,
                             min_dur_s: float = 0.6, max_dur_s: float = 2.0) -> Tuple[int, int, np.ndarray]:
    T = kpts.shape[0]
    if T <= 2:
        return 0, max(0, T - 1), np.zeros((T,), np.float32)

    A, _, _, _, _, _ = _ankle_side_indices(kpts, vis)

    hip = np.full((T, 2), np.nan, np.float32)
    mh = (vis[:, LH] > VTH) & (vis[:, RH] > VTH)
    hip[mh] = 0.5 * (kpts[mh, LH] + kpts[mh, RH])
    ms = (~mh) & (vis[:, LS] > VTH) & (vis[:, RS] > VTH)
    hip[ms] = 0.5 * (kpts[ms, LS] + kpts[ms, RS])
    m = np.any(np.isfinite(hip), axis=1)
    if not np.any(m):
        hip[:] = 0.0

    ank = kpts[:, A, :].copy()
    valid = (vis[:, A] > VTH) & np.all(np.isfinite(ank), axis=1) & np.all(np.isfinite(hip), axis=1)
    rel = np.zeros((T, 2), np.float32)
    rel[valid] = ank[valid] - hip[valid]

    sp = np.zeros((T,), np.float32)
    d = np.linalg.norm(np.diff(rel, axis=0), axis=1)
    sp[1:] = d.astype(np.float32)

    win = 7
    if T >= win:
        kernel = np.ones((win,), np.float32) / win
        sp_s = np.convolve(sp, kernel, mode="same")
    else:
        sp_s = sp

    peak = int(np.argmax(sp_s))
    q85 = float(np.quantile(sp_s, 0.85)) if np.any(sp_s > 0) else 0.0
    thr_hi = max(q85, 2.0)
    thr_lo = max(thr_hi * 0.55, 1.0)

    s = peak
    e = peak
    while s - 1 >= 0 and sp_s[s - 1] >= thr_lo:
        s -= 1
    while e + 1 < T and sp_s[e + 1] >= thr_lo:
        e += 1

    min_len = int(round(min_dur_s * fps))
    max_len = int(round(max_dur_s * fps))
    if max_len < 8:
        max_len = 8
    if min_len < 8:
        min_len = 8

    cur_len = e - s + 1
    if cur_len < min_len:
        need = min_len - cur_len
        left = need // 2
        right = need - left
        s = max(0, s - left)
        e = min(T - 1, e + right)
    if (e - s + 1) > max_len:
        half = max_len // 2
        s = max(0, peak - half)
        e = min(T - 1, s + max_len - 1)
        s = max(0, e - max_len + 1)

    return int(s), int(e), sp_s.astype(np.float32)


def _track_score_for_kicker(kpts: np.ndarray, vis: np.ndarray, fps: float, W: int, H: int) -> Tuple[float, Dict]:
    T = kpts.shape[0]
    if T <= 2:
        return -1.0, {"reason": "too_short"}

    s, e, _ = _compute_rel_speed_window(kpts, vis, fps)
    seg = slice(s, e + 1)

    A, _, _, _, _, _ = _ankle_side_indices(kpts, vis)

    cov = float(np.mean(vis[seg, A] > VTH)) if (e >= s) else 0.0

    hip = 0.5 * (kpts[:, LH] + kpts[:, RH])
    mhip = (vis[:, LH] > VTH) & (vis[:, RH] > VTH)
    hip[~mhip] = np.nan

    ank = kpts[:, A, :].copy()
    m = (vis[:, A] > VTH) & np.all(np.isfinite(ank), axis=1) & np.all(np.isfinite(hip), axis=1)
    rel = np.zeros_like(ank)
    rel[m] = ank[m] - hip[m]
    rel_seg = rel[seg]

    if rel_seg.shape[0] <= 1:
        motion = 0.0
    else:
        motion = float(np.nansum(np.linalg.norm(np.diff(rel_seg, axis=0), axis=1)))

    cx = W * 0.5
    cy = H * 0.5
    hip_seg = hip[seg]
    mhip2 = np.all(np.isfinite(hip_seg), axis=1)
    if np.any(mhip2):
        d = np.linalg.norm(hip_seg[mhip2] - np.array([cx, cy]), axis=1)
        d = float(np.mean(d))
        center_score = 1.0 - min(1.0, d / (0.5 * max(W, H)))
    else:
        center_score = 0.0

    miss_s = float(np.sum(vis[seg, A] <= VTH) / max(1.0, fps))

    info = {
        "win_s": int(s),
        "win_e": int(e),
        "cov": cov,
        "motion": motion,
        "center_score": float(center_score),
        "miss_s": miss_s,
        "ankle_idx": int(A),
    }
    return 0.0, info


def _choose_kicker_track(tracks: List[Track], fps: float, W: int, H: int,
                         unsure_ratio: float = 0.85) -> Tuple[int, Dict]:
    raw = []
    for i, tr in enumerate(tracks):
        k = np.stack(tr.kpts_list, axis=0)
        v = np.stack(tr.vis_list, axis=0)
        _, info = _track_score_for_kicker(k, v, fps, W, H)
        info["track_idx"] = i
        info["present"] = int(tr.n_present)
        raw.append(info)

    if not raw:
        return 0, {"invalid_for_scoring": 1, "reason": "no_pose"}

    motions = np.array([r["motion"] for r in raw], np.float32)
    mmax = float(np.max(motions)) if motions.size else 0.0
    mnorm = motions / (mmax + 1e-6)

    scores = []
    for i, r in enumerate(raw):
        cov = float(r["cov"])
        cs = float(r["center_score"])
        s = 0.55 * float(mnorm[i]) + 0.35 * cov + 0.10 * cs
        scores.append(float(s))

    order = np.argsort(scores)[::-1]
    best = int(order[0])
    best_score = float(scores[best])
    second_score = float(scores[int(order[1])]) if len(order) >= 2 else 0.0

    invalid = 0
    reason = ""
    if len(order) >= 2 and best_score > 0:
        if second_score >= best_score * unsure_ratio:
            invalid = 1
            reason = f"多人入镜踢腿者不确定（best={best_score:.3f}, second={second_score:.3f}）"

    dbg = {
        "best_track": best,
        "best_score": best_score,
        "second_score": second_score,
        "scores": scores,
        "tracks": raw,
        "invalid_for_scoring": int(invalid),
        "reason": reason,
    }
    return best, dbg


def extract_one_video(landmarker: vision.PoseLandmarker, video_path: Path, out_npz: Path,
                      num_poses: int = 4,
                      max_match_dist_ratio: float = 0.15,
                      unsure_ratio: float = 0.85,
                      skip_existing: bool = False,
                      verbose: bool = True):
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and out_npz.exists():
        if verbose:
            print(f"[SKIP] {video_path.name} -> {out_npz.relative_to(SEQ_ROOT)}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracks: List[Track] = []
    t = 0
    prev_ts = -1
    max_dist = max_match_dist_ratio * max(W, H)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts = int(round(t * 1000.0 / fps))
        if ts <= prev_ts:
            ts = prev_ts + 1
        prev_ts = ts

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, ts)

        det_k, det_v, det_c = [], [], []
        if result and result.pose_landmarks:
            for lm33 in result.pose_landmarks[:num_poses]:
                k13, v13 = _mp_landmarks_to_13(lm33, W, H)
                c = _center_of_pose(k13, v13)
                det_k.append(k13)
                det_v.append(v13)
                det_c.append(c.astype(np.float32))

        assign, new_det = _match_tracks(tracks, det_c, max_dist=max_dist)

        for tr in tracks:
            tr.kpts_list.append(np.full((13, 2), np.nan, np.float32))
            tr.vis_list.append(np.zeros((13,), np.float32))
            tr.center_list.append(np.array([np.nan, np.nan], np.float32))
            tr.present_list.append(False)

        for ti, di in assign.items():
            tr = tracks[ti]
            tr.kpts_list[-1] = det_k[di].astype(np.float32)
            tr.vis_list[-1] = det_v[di].astype(np.float32)
            tr.center_list[-1] = det_c[di].astype(np.float32)
            tr.present_list[-1] = True
            tr.last_center = det_c[di].astype(np.float32)
            tr.n_present += 1

        for di in new_det:
            tr = Track(kpts_list=[], vis_list=[], center_list=[], present_list=[])
            for _ in range(t):
                tr.kpts_list.append(np.full((13, 2), np.nan, np.float32))
                tr.vis_list.append(np.zeros((13,), np.float32))
                tr.center_list.append(np.array([np.nan, np.nan], np.float32))
                tr.present_list.append(False)

            tr.kpts_list.append(det_k[di].astype(np.float32))
            tr.vis_list.append(det_v[di].astype(np.float32))
            tr.center_list.append(det_c[di].astype(np.float32))
            tr.present_list.append(True)
            tr.last_center = det_c[di].astype(np.float32)
            tr.n_present = 1
            tracks.append(tr)

        t += 1

    cap.release()

    if t <= 1 or not tracks:
        np.savez_compressed(
            out_npz,
            kpts=np.zeros((0, 13, 2), np.float32),
            vis=np.zeros((0, 13), np.float32),
            fps=np.float32(fps),
            src_video=str(video_path),
            invalid_for_scoring=np.int32(1),
            reason="no_pose_or_too_short",
        )
        if verbose:
            print(f"[ERR] {video_path.name} -> {out_npz.relative_to(SEQ_ROOT)} | no_pose_or_too_short")
        return

    best_idx, dbg = _choose_kicker_track(tracks, fps=fps, W=W, H=H, unsure_ratio=unsure_ratio)
    tr = tracks[best_idx]
    kpts = np.stack(tr.kpts_list, axis=0).astype(np.float32)
    vis = np.stack(tr.vis_list, axis=0).astype(np.float32)

    win_s, win_e, _ = _compute_rel_speed_window(kpts, vis, fps=fps)
    k_win = kpts[win_s:win_e + 1]
    v_win = vis[win_s:win_e + 1]

    A, *_ = _ankle_side_indices(k_win, v_win)
    cov_best = float(np.mean(v_win[:, A] > VTH)) if k_win.shape[0] > 0 else 0.0
    miss_best = float(np.sum(v_win[:, A] <= VTH) / max(1.0, fps))

    invalid_for_scoring = int(dbg.get("invalid_for_scoring", 0))
    reason = str(dbg.get("reason", ""))

    np.savez_compressed(
        out_npz,
        kpts=k_win.astype(np.float32),
        vis=v_win.astype(np.float32),
        fps=np.float32(fps),
        src_video=str(video_path),
        invalid_for_scoring=np.int32(invalid_for_scoring),
        reason=reason,
        win_s=np.int32(win_s),
        win_e=np.int32(win_e),
        cov_best=np.float32(cov_best),
        miss_best=np.float32(miss_best),
        best_track=np.int32(best_idx),
        best_score=np.float32(dbg.get("best_score", 0.0)),
        second_score=np.float32(dbg.get("second_score", 0.0)),
        scores=np.array(dbg.get("scores", []), np.float32),
    )

    if verbose:
        print(f"[OK] {video_path.name} -> {out_npz.relative_to(SEQ_ROOT)} | "
              f"cov_best={cov_best:.2f} miss_best={miss_best:.2f}s | "
              f"win=rel_speed_window pre≈{win_s/max(1.0,fps):.1f}s dur≈{(win_e-win_s+1)/max(1.0,fps):.1f}s"
              + (f" | INVALID: {reason}" if invalid_for_scoring else ""))


def build_landmarker(task_path: Path, num_poses: int,
                     min_det: float = 0.5, min_pres: float = 0.5, min_track: float = 0.5):
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_path)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=int(num_poses),
        min_pose_detection_confidence=float(min_det),
        min_pose_presence_confidence=float(min_pres),
        min_tracking_confidence=float(min_track),
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--task", type=str, default="pose_landmarker_full.task", help="PoseLandmarker .task 文件路径或文件名")
    ap.add_argument("--num_poses", type=int, default=4, help="最多检测多少个人")
    ap.add_argument("--skip_existing", action="store_true", help="若 npz 已存在则跳过")
    ap.add_argument("--unsure_ratio", type=float, default=0.85, help="第二名分数>=第一名*该比例 -> 判不确定(invalid)")
    ap.add_argument("--max_match_dist_ratio", type=float, default=0.15, help="轨迹关联最大距离比例（相对 max(W,H)）")
    ap.add_argument("--min_det", type=float, default=0.5)
    ap.add_argument("--min_pres", type=float, default=0.5)
    ap.add_argument("--min_track", type=float, default=0.5)

    # 新增：默认排除 side
    ap.add_argument("--exclude_classes", type=str, default="side",
                    help="逗号分隔：要跳过的类别目录（默认 side）。为空则不过滤。例：--exclude_classes side")

    args = ap.parse_args()

    exclude_classes = _parse_exclude_classes(args.exclude_classes)

    task_path = _resolve_task_path(args.task)
    videos = _list_videos(args.split, exclude_classes=exclude_classes)
    print(f"[INFO] videos={len(videos)} | num_poses={args.num_poses} | task={task_path.name} | exclude={sorted(list(exclude_classes))}")

    for vp in videos:
        out_npz = _video_to_out_npz(vp, args.split)
        landmarker = build_landmarker(task_path, args.num_poses, args.min_det, args.min_pres, args.min_track)
        try:
            extract_one_video(
                landmarker, vp, out_npz,
                num_poses=args.num_poses,
                max_match_dist_ratio=args.max_match_dist_ratio,
                unsure_ratio=args.unsure_ratio,
                skip_existing=args.skip_existing,
                verbose=True
            )
        except Exception as e:
            print(f"[ERR] {vp} | {e}")
        finally:
            landmarker.close()
    print("[DONE]")


if __name__ == "__main__":
    main()
