# -*- coding: utf-8 -*-
"""
visualize_video.py
Render what the pipeline "sees" on top of an input video, to explain how the algorithm works.

Output video layout:
  Left  : input video
          - bbox of every detected person (green = chosen kicker, grey = others) + kicker-selection score
          - MediaPipe 33-point skeleton (thin) and the 13-point skeleton used by the model
            (dot color = visibility: green high -> red low, hollow = below threshold / treated as missing)
          - kicking-leg highlight and ankle trail
          - frames outside the detected kick window are darkened (the model never sees them)
          - phase from the scoring rules: preparation / chamber-extension / peak / recovery
  Right : 1) MediaPipe 3D world landmarks (rotating view; NOT used by the model, which is 2D only)
          2) Model input: hip-centered, body-scale-normalized 13-point skeleton
          3) Ankle-speed curve used to cut the kick window (thresholds, window, phases, cursor)
          4) Final recognition probabilities and score

Does NOT touch data/sequences. Outputs go to outputs/predictions/<video_stem>/:
  visualization.mp4, peak.png (frame at maximum extension)

Arm mode (--limb arm): for punch / elbow videos. The action window and phases are computed from the
wrist speed relative to the shoulder center instead of the ankle. There is no punch/elbow model or
scoring rule yet, so recognition and scoring are skipped and raw arm measurements are shown instead.
Performer selection (multi-person) is still the leg-motion rule from extract_pose_mediapipe.

Usage:
    python scripts/visualize_video.py path\\to\\video.mp4
    python scripts/visualize_video.py path\\to\\punch.mp4 --limb arm
    python scripts/visualize_video.py path\\to\\folder_of_videos
    python scripts/visualize_video.py video.mp4 --kick front
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from extract_pose_mediapipe import (  # noqa: E402
    HEAD, LA, LE, LH, LK, LS, LW, RA, RE, RH, RK, RS, RW, VTH,
    Track, _center_of_pose, _choose_kicker_track, _compute_rel_speed_window,
    _match_tracks, _mp_landmarks_to_13, build_landmarker,
)
from eval_kick3_baselines import normalize_kpts  # noqa: E402
from predict_video import (  # noqa: E402
    DEFAULT_CKPTS, DEFAULT_TASK, OUT_ROOT, ensure_task, list_videos, load_models, recognize,
)
from scoring_api import _kick_side_indices, _split_phases, score_one_video  # noqa: E402

EDGES_13 = [
    (HEAD, LS), (HEAD, RS), (LS, LE), (LE, LW), (RS, RE), (RE, RW), (LS, RS),
    (LS, LH), (RS, RH), (LH, RH), (LH, LK), (LK, LA), (RH, RK), (RK, RA),
]
POSE_CONNECTIONS = sorted(mp.solutions.pose.POSE_CONNECTIONS)
J13_TO_33 = {HEAD: 0, LS: 11, RS: 12, LE: 13, RE: 14, LW: 15, RW: 16,
             LH: 23, RH: 24, LK: 25, RK: 26, LA: 27, RA: 28}

HO = 720             # output height
PW = 480             # right column width
PANEL_H = (250, 200, 140, 130)
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, GREY, RED, WHITE = (0, 220, 0), (160, 160, 160), (60, 60, 255), (240, 240, 240)
LEFT_C, RIGHT_C = (255, 170, 60), (60, 170, 255)  # BGR: left side blue, right side orange
BG = (28, 28, 28)


# ----------------------------------------------------------------------------
# Pass 1: detection + tracking (same logic as extract_pose_mediapipe.extract_one_video,
# but keeps every detection, the 33-point landmarks and the 3D world landmarks)
# ----------------------------------------------------------------------------
def detect_and_track(video: Path, task_path: Path, num_poses: int, max_match_dist_ratio: float = 0.15):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_dist = max_match_dist_ratio * max(W, H)

    landmarker = build_landmarker(task_path, num_poses)
    tracks: list[Track] = []
    frames: list[list[dict]] = []
    t, prev_ts = 0, -1
    try:
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

            dets = []
            if result and result.pose_landmarks:
                worlds = result.pose_world_landmarks or []
                for i, lm33 in enumerate(result.pose_landmarks[:num_poses]):
                    k13, v13 = _mp_landmarks_to_13(lm33, W, H)
                    p33 = np.array([[l.x * W, l.y * H, 1.0 if l.visibility is None else l.visibility]
                                    for l in lm33], np.float32)
                    w33 = np.array([[l.x, l.y, l.z] for l in worlds[i]], np.float32) if i < len(worlds) else None
                    dets.append({"k13": k13, "v13": v13, "p33": p33, "w33": w33,
                                 "center": _center_of_pose(k13, v13).astype(np.float32)})

            assign, new_det = _match_tracks(tracks, [d["center"] for d in dets], max_dist=max_dist)
            for tr in tracks:
                tr.kpts_list.append(np.full((13, 2), np.nan, np.float32))
                tr.vis_list.append(np.zeros((13,), np.float32))
                tr.center_list.append(np.array([np.nan, np.nan], np.float32))
                tr.present_list.append(False)
            for ti, di in assign.items():
                tr = tracks[ti]
                tr.kpts_list[-1] = dets[di]["k13"]
                tr.vis_list[-1] = dets[di]["v13"]
                tr.center_list[-1] = dets[di]["center"]
                tr.present_list[-1] = True
                tr.last_center = dets[di]["center"]
                tr.n_present += 1
                dets[di]["track"] = ti
            for di in new_det:
                tr = Track(kpts_list=[np.full((13, 2), np.nan, np.float32) for _ in range(t)],
                           vis_list=[np.zeros((13,), np.float32) for _ in range(t)],
                           center_list=[np.array([np.nan, np.nan], np.float32) for _ in range(t)],
                           present_list=[False] * t)
                tr.kpts_list.append(dets[di]["k13"])
                tr.vis_list.append(dets[di]["v13"])
                tr.center_list.append(dets[di]["center"])
                tr.present_list.append(True)
                tr.last_center = dets[di]["center"]
                tr.n_present = 1
                tracks.append(tr)
                dets[di]["track"] = len(tracks) - 1

            frames.append(dets)
            t += 1
    finally:
        cap.release()
        landmarker.close()
    return fps, W, H, tracks, frames


def analyse(tracks, fps, W, H, models, kick):
    best, dbg = _choose_kicker_track(tracks, fps=fps, W=W, H=H, unsure_ratio=0.85)
    kpts = np.stack(tracks[best].kpts_list).astype(np.float32)
    vis = np.stack(tracks[best].vis_list).astype(np.float32)
    win_s, win_e, sp_s = _compute_rel_speed_window(kpts, vis, fps=fps)
    k_win, v_win = kpts[win_s:win_e + 1], vis[win_s:win_e + 1]

    # same thresholds as _compute_rel_speed_window
    q85 = float(np.quantile(sp_s, 0.85)) if np.any(sp_s > 0) else 0.0
    thr_hi = max(q85, 2.0)
    thr_lo = max(thr_hi * 0.55, 1.0)

    probs = None
    if kick:
        kick_type = kick
    else:
        kick_type, probs = recognize(models, k_win, v_win)
    acc4, expr6, total10, details = score_one_video({"kpts": k_win, "vis": v_win, "fps": fps}, kick_type)
    raw = details["raw"]

    xy = k_win.copy()
    xy[v_win < 0.2] = np.nan
    with np.errstate(all="ignore"):
        A = _kick_side_indices(xy)[0]

    # panel scale: fit the whole normalized window (side views give a tiny body width -> large values)
    norm = normalize_kpts(k_win, v_win, tau=0.5)
    norm_sc = _norm_panel_scale(norm)

    return {
        "best": best, "scores": dbg.get("scores", [0.0] * len(tracks)),
        "invalid": int(dbg.get("invalid_for_scoring", 0)),
        "kpts": kpts, "vis": vis, "win_s": win_s, "win_e": win_e,
        "sp_s": sp_s, "thr_hi": thr_hi, "thr_lo": thr_lo,
        "norm": norm,
        "norm_sc": norm_sc,
        "kick_type": kick_type, "probs": probs,
        "score": (acc4, expr6, total10), "top_issues": details["top_issues"],
        "t0": win_s + int(raw["t0"]), "tp": win_s + int(raw["t_peak"]), "t1": win_s + int(raw["t1"]),
        "A": A,
        "role": "KICKER", "win_name": "kick window",
        "limb_desc": "Kick-window detection: ankle speed rel. to hip",
        "peak_desc": "PEAK: max ankle-hip distance",
        "phase_names": ("preparation", "chamber / extension", "recovery"),
        "measures": None,
    }


def _norm_panel_scale(norm: np.ndarray) -> float:
    ext = np.nanmax(np.abs(norm), axis=(0, 1)) if np.isfinite(norm).any() else np.array([1.0, 1.0])
    ext = np.maximum(np.nan_to_num(ext, nan=1.0), 1e-3)
    return float(min(30.0, 0.42 * PW / ext[0], 0.30 * PANEL_H[1] / ext[1]))


def _cut_window(sp_s: np.ndarray, fps: float, min_dur_s: float = 0.6, max_dur_s: float = 2.0):
    """Same window rule as extract_pose_mediapipe._compute_rel_speed_window, on any speed curve."""
    T = sp_s.shape[0]
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
        need = min_len - (e - s + 1)
        s = max(0, s - need // 2)
        e = min(T - 1, e + need - need // 2)
    if e - s + 1 > max_len:
        s = max(0, peak - max_len // 2)
        e = min(T - 1, s + max_len - 1)
        s = max(0, e - max_len + 1)
    return int(s), int(e), thr_hi, thr_lo


def _angle(a, b, c) -> float:
    """angle ABC in degrees"""
    v1, v2 = a - b, c - b
    n = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if not np.isfinite(n) or n < 1e-6:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def analyse_arm(tracks, fps, W, H, side: str):
    best, dbg = _choose_kicker_track(tracks, fps=fps, W=W, H=H, unsure_ratio=0.85)
    kpts = np.stack(tracks[best].kpts_list).astype(np.float32)
    vis = np.stack(tracks[best].vis_list).astype(np.float32)
    T = kpts.shape[0]

    xy = kpts.copy()
    xy[vis <= VTH] = np.nan
    sh = 0.5 * (xy[:, LS] + xy[:, RS])
    hip = 0.5 * (xy[:, LH] + xy[:, RH])

    def path_len(j):
        r = xy[:, j] - sh
        return float(np.nansum(np.linalg.norm(np.diff(r, axis=0), axis=1)))

    if side == "right":
        A = RW
    elif side == "left":
        A = LW
    else:
        A = RW if path_len(RW) >= path_len(LW) else LW

    # wrist speed relative to shoulder center (invalid frames -> 0, as in the ankle version)
    rel = np.nan_to_num(xy[:, A] - sh, nan=0.0)
    sp = np.zeros((T,), np.float32)
    sp[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=1)
    sp_s = np.convolve(sp, np.ones(7, np.float32) / 7, mode="same") if T >= 7 else sp
    win_s, win_e, thr_hi, thr_lo = _cut_window(sp_s, fps)

    with np.errstate(all="ignore"):
        t0, tp, t1 = _split_phases(xy[win_s:win_e + 1], A, sh[win_s:win_e + 1])
        torso = float(np.nanmedian(np.linalg.norm(sh - hip, axis=1)))
    tp_abs = win_s + tp
    S, E = A - 4, A - 2
    measures = {
        "arm": "right" if A == RW else "left",
        "elbow_deg": _angle(xy[tp_abs, S], xy[tp_abs, E], xy[tp_abs, A]),
        "wrist_above_shoulder": float((sh[tp_abs, 1] - xy[tp_abs, A, 1]) / torso) if torso > 0 else float("nan"),
        "peak_speed": float(sp_s[win_s:win_e + 1].max() * fps / torso) if torso > 0 else float("nan"),
        "wrist_vis": float(np.mean(vis[win_s:win_e + 1, A] > VTH)),
        "elbow_vis": float(np.mean(vis[win_s:win_e + 1, E] > VTH)),
    }

    norm = normalize_kpts(kpts[win_s:win_e + 1], vis[win_s:win_e + 1], tau=0.5)
    return {
        "best": best, "scores": dbg.get("scores", [0.0] * len(tracks)),
        "invalid": int(dbg.get("invalid_for_scoring", 0)),
        "kpts": kpts, "vis": vis, "win_s": win_s, "win_e": win_e,
        "sp_s": sp_s, "thr_hi": thr_hi, "thr_lo": thr_lo,
        "norm": norm, "norm_sc": _norm_panel_scale(norm),
        "kick_type": "arm strike", "probs": None, "score": None, "top_issues": [],
        "t0": win_s + t0, "tp": tp_abs, "t1": win_s + t1,
        "A": A,
        "role": "PERFORMER", "win_name": "strike window",
        "limb_desc": "Strike-window detection: wrist speed rel. to shoulders",
        "peak_desc": "PEAK: max wrist-shoulder distance",
        "phase_names": ("guard", "extension", "retraction"),
        "measures": measures,
    }


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
def put(img, text, org, color=WHITE, scale=0.45, thick=1, bg=True):
    (w, h), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg:
        cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def vis_color(v: float):
    v = float(np.clip(v, 0.0, 1.0))
    return (0, int(255 * v), int(255 * (1.0 - v)))


def ok_pt(p) -> bool:
    return bool(np.all(np.isfinite(p)))


def ipt(p, s=1.0, off=(0.0, 0.0)):
    return int(round(p[0] * s + off[0])), int(round(p[1] * s + off[1]))


def side_color(j33: int):
    if j33 <= 10:
        return WHITE
    return LEFT_C if j33 % 2 == 1 else RIGHT_C


def phase_label(t, a) -> tuple[str, tuple]:
    wn, (p0, p1, p2) = a["win_name"], a["phase_names"]
    if not (a["win_s"] <= t <= a["win_e"]):
        return f"OUTSIDE {wn} - ignored", GREY
    if abs(t - a["tp"]) <= 1:
        return a["peak_desc"], RED
    if t < a["t0"]:
        return f"{wn}: {p0}", WHITE
    if t < a["tp"]:
        return f"{wn}: {p1}", (0, 200, 255)
    if t <= a["t1"]:
        return f"{wn}: {p2}", (255, 200, 0)
    return f"{wn}: after {p2}", WHITE


# ----------------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------------
def render_main(frame, t, dets, a, W, H, fps, T):
    s = HO / H
    img = cv2.resize(frame, (int(round(W * s)) // 2 * 2, HO))
    in_win = a["win_s"] <= t <= a["win_e"]
    if not in_win:
        img = (img * 0.45).astype(np.uint8)

    A = a["A"]
    kick_leg = {(A - 4, A - 2), (A - 2, A)}  # active limb: hip-knee-ankle or shoulder-elbow-wrist

    for d in dets:
        is_k = d.get("track") == a["best"]
        p33 = d["p33"]
        m = p33[:, 2] > VTH
        if m.any():
            x0, y0 = p33[m, :2].min(0)
            x1, y1 = p33[m, :2].max(0)
            pad = 0.06 * max(x1 - x0, y1 - y0)
            col = GREEN if is_k else GREY
            cv2.rectangle(img, ipt((x0 - pad, y0 - pad), s), ipt((x1 + pad, y1 + pad), s), col, 2 if is_k else 1)
            ti = d.get("track", -1)
            sc = a["scores"][ti] if 0 <= ti < len(a["scores"]) else 0.0
            lx, ly = ipt((x0 - pad, y0 - pad), s)
            ly = ly - 4 if ly > 70 else max(ly, 50) + 18  # keep label below the top banner
            put(img, f"{a['role'] + ' ' if is_k else ''}track#{ti} select={sc:.2f}", (max(lx, 4), ly), col, 0.45)

        if is_k:
            for i, j in POSE_CONNECTIONS:  # full MediaPipe 33-point skeleton, thin
                if p33[i, 2] > VTH and p33[j, 2] > VTH:
                    cv2.line(img, ipt(p33[i], s), ipt(p33[j], s), (200, 200, 200), 1, cv2.LINE_AA)
            k, v = d["k13"], d["v13"]
            for i, j in EDGES_13:  # 13-point skeleton used by the model
                if ok_pt(k[i]) and ok_pt(k[j]):
                    leg = (i, j) in kick_leg or (j, i) in kick_leg
                    cv2.line(img, ipt(k[i], s), ipt(k[j], s), RED if leg else GREEN, 4 if leg else 2, cv2.LINE_AA)
            for j in range(13):
                if ok_pt(k[j]):
                    cv2.circle(img, ipt(k[j], s), 6, vis_color(v[j]), -1 if v[j] > VTH else 2, cv2.LINE_AA)
        else:
            k = d["k13"]
            for i, j in EDGES_13:
                if ok_pt(k[i]) and ok_pt(k[j]):
                    cv2.line(img, ipt(k[i], s), ipt(k[j], s), GREY, 1, cv2.LINE_AA)

    # active ankle / wrist trail (last ~0.8 s)
    n = max(2, int(0.8 * fps))
    pts = [(tt, a["kpts"][tt, A]) for tt in range(max(0, t - n), t + 1)
           if a["vis"][tt, A] > VTH and ok_pt(a["kpts"][tt, A])]
    for (t_a, p_a), (_, p_b) in zip(pts, pts[1:]):
        f = 1.0 - (t - t_a) / n
        cv2.line(img, ipt(p_a, s), ipt(p_b, s), (0, int(120 + 135 * f), 255), max(1, int(4 * f)), cv2.LINE_AA)

    cv2.rectangle(img, (0, 0), (img.shape[1], 50), (0, 0, 0), -1)
    put(img, f"frame {t}/{T - 1}  t={t / fps:.2f}s", (8, 18), WHITE, 0.5, bg=False)
    label, col = phase_label(t, a)
    put(img, label, (8, 42), col, 0.55, 2 if col == RED else 1, bg=False)
    put(img, "dot color = visibility (green high, red low, hollow = missing)", (8, HO - 10), WHITE, 0.4)
    return img


def render_3d(w33, t, fps, A):
    h = PANEL_H[0]
    img = np.full((h, PW, 3), BG, np.uint8)
    put(img, "3D world landmarks (MediaPipe, meters)", (8, 18), WHITE, 0.45, bg=False)
    put(img, "not used by model - model is 2D only", (8, 36), GREY, 0.4, bg=False)
    if w33 is None:
        put(img, "no performer detected in this frame", (PW // 2 - 140, h // 2), GREY, 0.45, bg=False)
        return img

    yaw = 0.9 * math.sin(2 * math.pi * t / (fps * 8.0))  # slow side-to-side rotation
    pitch = 0.25
    cy_, sy_, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    sc, cx, cy = 100.0, PW / 2, h * 0.5

    def proj(p):
        x = p[0] * cy_ + p[2] * sy_
        z = -p[0] * sy_ + p[2] * cy_
        y = p[1] * cp - z * sp
        return int(round(cx + x * sc)), int(round(cy + y * sc))

    foot_y = float(np.max(w33[27:33, 1]))
    for g in np.arange(-1.0, 1.01, 0.5):
        cv2.line(img, proj((g, foot_y, -1.0)), proj((g, foot_y, 1.0)), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.line(img, proj((-1.0, foot_y, g)), proj((1.0, foot_y, g)), (70, 70, 70), 1, cv2.LINE_AA)

    kick33 = {J13_TO_33[A - 4], J13_TO_33[A - 2], J13_TO_33[A]}
    for i, j in POSE_CONNECTIONS:
        leg = i in kick33 and j in kick33
        cv2.line(img, proj(w33[i]), proj(w33[j]), RED if leg else side_color(max(i, j)), 3 if leg else 2, cv2.LINE_AA)
    for i in range(33):
        cv2.circle(img, proj(w33[i]), 2, WHITE, -1, cv2.LINE_AA)
    put(img, "blue = left side, orange = right, red = active limb", (8, h - 8), GREY, 0.38, bg=False)
    return img


def render_norm(t, a):
    h = PANEL_H[1]
    img = np.full((h, PW, 3), BG, np.uint8)
    put(img, "Model input: hip-centered, scaled by body width", (8, 18), WHITE, 0.45, bg=False)
    ws, we = a["win_s"], a["win_e"]
    if not (ws <= t <= we):
        put(img, f"frame outside {a['win_name']} -> not analysed", (40, h // 2), GREY, 0.45, bg=False)
        return img

    norm, A = a["norm"], a["A"]
    i = t - ws
    Tw = norm.shape[0]
    sc, off = a["norm_sc"], (PW / 2, h * 0.5)
    cv2.line(img, (int(off[0]) - 8, int(off[1])), (int(off[0]) + 8, int(off[1])), GREY, 1)
    cv2.line(img, (int(off[0]), int(off[1]) - 8), (int(off[0]), int(off[1]) + 8), GREY, 1)

    trail = [ipt(p, sc, off) for p in norm[:, A] if ok_pt(p)]
    if len(trail) >= 2:
        cv2.polylines(img, [np.array(trail, np.int32)], False, (90, 90, 160), 1, cv2.LINE_AA)
    p = norm[i]
    for u, v in EDGES_13:
        if ok_pt(p[u]) and ok_pt(p[v]):
            leg = {u, v} <= {A - 4, A - 2, A}
            cv2.line(img, ipt(p[u], sc, off), ipt(p[v], sc, off), RED if leg else GREEN, 2, cv2.LINE_AA)
    for j in range(13):
        if ok_pt(p[j]):
            cv2.circle(img, ipt(p[j], sc, off), 3, WHITE, -1, cv2.LINE_AA)
    r = int(round(i * 95 / max(1, Tw - 1)))
    put(img, f"1 body-width unit = {sc:.0f}px", (PW - 150, 36), GREY, 0.38, bg=False)
    put(img, f"window frame {i + 1}/{Tw} -> resampled step {r + 1}/96", (8, h - 26), GREY, 0.4, bg=False)
    put(img, "input = 13 joints x (x,y) + velocity = 52 values/step", (8, h - 8), GREY, 0.4, bg=False)
    return img


def build_speed_bg(a, T):
    h = PANEL_H[2]
    img = np.full((h, PW, 3), BG, np.uint8)
    put(img, a["limb_desc"], (8, 16), WHITE, 0.45, bg=False)
    x0, x1, y0, y1 = 10, PW - 10, 26, h - 12
    sp = a["sp_s"]
    top = max(float(sp.max()) if sp.size else 1.0, a["thr_hi"]) * 1.1

    def X(t):
        return int(round(x0 + (x1 - x0) * t / max(1, T - 1)))

    def Y(v):
        return int(round(y1 - (y1 - y0) * v / top))

    cv2.rectangle(img, (X(a["win_s"]), y0), (X(a["win_e"]), y1), (45, 70, 45), -1)
    cv2.line(img, (x0, Y(a["thr_hi"])), (x1, Y(a["thr_hi"])), (0, 140, 255), 1)
    cv2.line(img, (x0, Y(a["thr_lo"])), (x1, Y(a["thr_lo"])), (0, 200, 255), 1)
    put(img, "hi thr", (x1 - 50, Y(a["thr_hi"]) - 3), (0, 140, 255), 0.35, bg=False)
    put(img, "lo thr", (x1 - 50, Y(a["thr_lo"]) + 11), (0, 200, 255), 0.35, bg=False)
    pts = np.array([(X(t), Y(float(v))) for t, v in enumerate(sp)], np.int32)
    if len(pts) >= 2:
        cv2.polylines(img, [pts], False, WHITE, 1, cv2.LINE_AA)
    for row, (key, lab, col) in enumerate((("t0", "start", (0, 200, 255)), ("tp", "peak", RED),
                                           ("t1", "end", (255, 200, 0)))):
        x = X(a[key])
        cv2.line(img, (x, y0), (x, y1), col, 1)
        put(img, lab, (x + 3, y0 + 10 + 12 * row), col, 0.35, bg=False)
    put(img, f"green = {a['win_name']} (only part analysed)", (8, h - 1), GREY, 0.35, bg=False)
    return img, X, (y0, y1)


def render_speed(bg, X, yr, t):
    img = bg.copy()
    cv2.line(img, (X(t), yr[0]), (X(t), yr[1]), (255, 0, 255), 2)
    return img


def build_result_panel(a, n_people):
    h = PANEL_H[3]
    img = np.full((h, PW, 3), BG, np.uint8)
    if a["score"] is None:
        m = a["measures"]
        put(img, f"ARM STRIKE ({m['arm']} arm) - no punch/elbow model yet", (8, 20), (0, 200, 255), 0.5, 1, bg=False)
        put(img, f"at peak: elbow angle {m['elbow_deg']:.0f} deg, wrist {m['wrist_above_shoulder']:+.2f} torso above shoulder",
            (8, 44), WHITE, 0.4, bg=False)
        put(img, f"peak wrist speed {m['peak_speed']:.1f} torso-lengths/s", (8, 64), WHITE, 0.4, bg=False)
        qc = GREEN if min(m["wrist_vis"], m["elbow_vis"]) >= 0.8 else RED
        put(img, f"tracking in window: wrist visible {m['wrist_vis']:.0%}, elbow visible {m['elbow_vis']:.0%}",
            (8, 84), qc, 0.4, bg=False)
        put(img, "raw measurements only - not a score", (8, h - 26), GREY, 0.38, bg=False)
        if a["invalid"]:
            put(img, "WARNING: performer uncertain (two people scored similarly)", (8, h - 8), RED, 0.4, bg=False)
        return img
    acc4, expr6, total10 = a["score"]
    put(img, f"Kick: {a['kick_type'].upper()}", (8, 20), GREEN, 0.6, 2, bg=False)
    put(img, f"Score {total10:.2f}/10  (acc {acc4:.2f}/4, expr {expr6:.2f}/6)", (170, 20), WHITE, 0.45, bg=False)
    if a["probs"]:
        for k, (c, p) in enumerate(a["probs"].items()):
            y = 44 + k * 22
            put(img, c, (8, y), WHITE, 0.45, bg=False)
            cv2.rectangle(img, (110, y - 12), (110 + int(300 * p), y + 2),
                          GREEN if c == a["kick_type"] else GREY, -1)
            put(img, f"{p:.2f}", (420, y), WHITE, 0.45, bg=False)
    else:
        put(img, "kick type given by user (--kick), recognition skipped", (8, 48), GREY, 0.42, bg=False)
    msg = f"{n_people} person track(s); kicker = most leg motion + visibility + centered"
    put(img, msg, (8, h - 26), GREY, 0.38, bg=False)
    if a["invalid"]:
        put(img, "WARNING: kicker uncertain (two people scored similarly)", (8, h - 8), RED, 0.4, bg=False)
    return img


# ----------------------------------------------------------------------------
def open_writer(path: Path, fps: float, size):
    for code in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, size)
        if w.isOpened():
            return w
    raise RuntimeError("cannot open a video writer (tried avc1, mp4v)")


def visualize(video: Path, task_path: Path, num_poses: int, models, kick: str | None,
              limb: str = "leg", side: str = "auto") -> None:
    print(f"[1/3] detecting people + pose: {video.name}")
    fps, W, H, tracks, frames = detect_and_track(video, task_path, num_poses)
    T = len(frames)
    if T <= 1 or not tracks:
        print("  [ERR] no pose detected or video too short")
        return

    if limb == "arm":
        print("[2/3] performer selection, strike window, arm measurements")
        a = analyse_arm(tracks, fps, W, H, side)
    else:
        print("[2/3] kicker selection, kick window, recognition, scoring")
        a = analyse(tracks, fps, W, H, models, kick)

    out_dir = OUT_ROOT / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "visualization.mp4"
    speed_bg, X, yr = build_speed_bg(a, T)
    result_img = build_result_panel(a, len(tracks))

    print(f"[3/3] rendering {T} frames")
    cap = cv2.VideoCapture(str(video))
    writer = None
    try:
        for t in range(T):
            ok, frame = cap.read()
            if not ok:
                break
            kdet = next((d for d in frames[t] if d.get("track") == a["best"]), None)
            main = render_main(frame, t, frames[t], a, W, H, fps, T)
            right = np.vstack([
                render_3d(kdet["w33"] if kdet else None, t, fps, a["A"]),
                render_norm(t, a),
                render_speed(speed_bg, X, yr, t),
                result_img,
            ])
            canvas = np.hstack([main, right])
            if writer is None:
                writer = open_writer(out_path, fps, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
            if t == a["tp"]:
                cv2.imwrite(str(out_dir / "peak.png"), canvas)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    warn = "  [WARN] performer uncertain" if a["invalid"] else ""
    if a["score"] is None:
        m = a["measures"]
        print(f"  arm={m['arm']}  window={a['win_s']}-{a['win_e']}  peak_frame={a['tp']}  "
              f"elbow={m['elbow_deg']:.0f}deg  wrist_visible={m['wrist_vis']:.0%}  people={len(tracks)}{warn}")
    else:
        acc4, expr6, total10 = a["score"]
        print(f"  kick={a['kick_type']}  score={total10:.2f}/10  window={a['win_s']}-{a['win_e']}  "
              f"people={len(tracks)}{warn}")
    print(f"  saved: {out_path}")
    print(f"  saved: {out_dir / 'peak.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize pose, kicker selection, kick window, model input and score.")
    ap.add_argument("input", type=str, help="video file or folder of videos")
    ap.add_argument("--kick", type=str, default=None, choices=["front", "roundhouse", "axe"],
                    help="known kick type: skip recognition")
    ap.add_argument("--ckpt", type=str, nargs="+", default=[str(p) for p in DEFAULT_CKPTS])
    ap.add_argument("--task", type=str, default=str(DEFAULT_TASK))
    ap.add_argument("--num_poses", type=int, default=4)
    ap.add_argument("--limb", type=str, default="leg", choices=["leg", "arm"],
                    help="leg = kicks (default, original pipeline); arm = punch / elbow (measurements only)")
    ap.add_argument("--side", type=str, default="auto", choices=["auto", "left", "right"],
                    help="arm mode: which arm strikes (auto = the wrist that moves most)")
    args = ap.parse_args()

    videos = list_videos(Path(args.input))
    if not videos:
        sys.exit(f"[ERR] no video found at {args.input}")
    task_path = ensure_task(Path(args.task))
    models = None if (args.kick or args.limb == "arm") else load_models([Path(p) for p in args.ckpt])
    for v in videos:
        try:
            visualize(v, task_path, args.num_poses, models, args.kick, args.limb, args.side)
        except Exception as e:
            print(f"[ERR] {v} | {e}")


if __name__ == "__main__":
    main()
