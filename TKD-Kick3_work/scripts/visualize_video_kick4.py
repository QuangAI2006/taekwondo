# -*- coding: utf-8 -*-
"""
visualize_video_kick4.py
Render model inference on top of a video with the 4-class model
(front / roundhouse / axe / hook_right) and the matching scoring rules.

Same layout as visualize_video.py (whose drawing code is imported, not modified):
  Left  : input video with bbox of every person (green = performer), MediaPipe 33-point
          skeleton, 13-point model skeleton (dot color = visibility), active limb + trail,
          frames outside the action window darkened, phase label
  Right : 1) MediaPipe 3D world landmarks (rotating)
          2) model input (normalized 13-point skeleton)
          3) speed curve used to cut the action window (ankle for kicks, wrist for hook)
          4) recognition probabilities, score /10 and main deductions (Unicode text)

Recognition = predict_video_kick4.recognize (kick view vs punch view).
Scoring     = scoring_api (kicks) / hook_scoring (hook_right).

Outputs: outputs/predictions_kick4/<video_stem>/{visualization.mp4, peak.png, result.json}

Usage:
    python scripts/visualize_video_kick4.py path\\to\\video.mp4
    python scripts/visualize_video_kick4.py path\\to\\folder
    python scripts/visualize_video_kick4.py video.mp4 --kick hook_right   (skip recognition)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from extract_pose_mediapipe import _compute_rel_speed_window  # noqa: E402
from eval_kick3_baselines import normalize_kpts  # noqa: E402
from hook_pose import RW, _rel_wrist, kick_view, punch_view  # noqa: E402
from hook_scoring import compute_raw, score_hook  # noqa: E402
from predict_video import ensure_task, list_videos, load_models  # noqa: E402
from predict_video_kick4 import CLASS_CHOICES, DEFAULT_CKPTS, DEFAULT_TASK, HOOK, OUT_ROOT, recognize  # noqa: E402
from scoring_api import _kick_side_indices, score_one_video  # noqa: E402
from visualize_video import (  # noqa: E402
    BG, GREEN, GREY, PANEL_H, PW, RED, WHITE, _norm_panel_scale, build_speed_bg, detect_and_track,
    open_writer, render_3d, render_main, render_norm, render_speed,
)

WARN_VI = {
    "left wrist moved more than right wrist: check that this is a RIGHT hook":
        "tay trái chuyển động nhiều hơn tay phải - kiểm tra đúng móc phải",
}
FONT_DIR = Path("C:/Windows/Fonts")
_FONTS: dict = {}


def _font(size: int, cjk: bool) -> ImageFont.ImageFont:
    key = (size, cjk)
    if key not in _FONTS:
        names = ["msyh.ttc", "segoeui.ttf", "arial.ttf"] if cjk else ["segoeui.ttf", "arial.ttf", "msyh.ttc"]
        for n in names:
            try:
                _FONTS[key] = ImageFont.truetype(str(FONT_DIR / n), size)
                break
            except OSError:
                continue
        else:
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]


def draw_texts(img: np.ndarray, texts: list[tuple[str, tuple[int, int], tuple, int]]) -> np.ndarray:
    """Draw Unicode text (Vietnamese / Chinese) with PIL. texts: (text, (x, y_top), BGR color, px size)."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for text, (x, y), bgr, size in texts:
        cjk = bool(re.search(r"[\u3400-\u9fff]", text))
        d.text((x, y), text, font=_font(size, cjk), fill=(bgr[2], bgr[1], bgr[0]))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def _thresholds(sp_s: np.ndarray) -> tuple[float, float]:
    """Same thresholds as the window cutters in extract_pose_mediapipe / hook_pose."""
    q85 = float(np.quantile(sp_s, 0.85)) if np.any(sp_s > 0) else 0.0
    thr_hi = max(q85, 2.0)
    return thr_hi, max(thr_hi * 0.55, 1.0)


def _wrist_speed(kpts: np.ndarray, vis: np.ndarray, Wi: int) -> np.ndarray:
    """Same speed curve as hook_pose.compute_wrist_window."""
    rel, valid = _rel_wrist(kpts, vis, Wi)
    T = kpts.shape[0]
    sp = np.zeros((T,), np.float32)
    if T > 1:
        step = np.linalg.norm(np.diff(rel, axis=0), axis=1)
        step[~(valid[1:] & valid[:-1])] = 0.0
        sp[1:] = step
    return np.convolve(sp, np.ones(7, np.float32) / 7, mode="same") if T >= 7 else sp


def _track(tracks, i):
    return (np.stack(tracks[i].kpts_list).astype(np.float32), np.stack(tracks[i].vis_list).astype(np.float32))


def analyse_kick4(video: Path, tracks, fps, W, H, T, models, kick: str | None) -> dict:
    kv = kick_view(tracks, fps, W, H, T, video)
    pv = punch_view(tracks, fps, W, H, T, video)

    probs_all = None
    if kick:
        label = kick
    else:
        label, probs_all = recognize(models, kv, pv)

    if label == HOOK:
        if pv.get("attack_wrist") != "R":  # label says right hook -> analyse the right wrist
            pv = punch_view(tracks, fps, W, H, T, video, wrist=RW)
        best = int(pv["best_track"])
        kpts, vis = _track(tracks, best)
        ws, we = int(pv["win_s"]), int(pv["win_e"])
        sp_s = _wrist_speed(kpts, vis, RW)
        acc4, expr6, total10, details = score_hook(pv)
        raw = compute_raw(pv["kpts"], pv["vis"], float(pv["fps"]))
        tp = ws + raw.t_impact
        # phases from wrist travel away from the guard position
        rel, _ = _rel_wrist(kpts[ws:we + 1], vis[ws:we + 1], RW)
        disp = np.linalg.norm(rel - np.median(rel[:3], axis=0), axis=1)
        dmax = max(float(disp.max()), 1e-6)
        pre = np.where(disp[:raw.t_impact + 1] < 0.2 * dmax)[0]
        post = np.where(disp[raw.t_impact:] < 0.5 * dmax)[0]
        t0 = ws + (int(pre[-1]) if len(pre) else 0)
        t1 = ws + raw.t_impact + (int(post[0]) if len(post) else we - ws - raw.t_impact)
        A = RW
        view, probs = pv, (probs_all or {}).get("punch_view")
        extra = dict(role="PERFORMER", win_name="strike window",
                     limb_desc="Strike-window detection: wrist speed rel. to shoulders",
                     peak_desc="IMPACT: max wrist travel from guard",
                     phase_names=("guard", "hook swing", "retraction"))
        warnings = details["warnings"]
    else:
        best = int(kv["best_track"])
        kpts, vis = _track(tracks, best)
        ws, we, sp_s = _compute_rel_speed_window(kpts, vis, fps=fps)
        acc4, expr6, total10, details = score_one_video(kv, label)
        raw = details["raw"]
        t0, tp, t1 = ws + int(raw["t0"]), ws + int(raw["t_peak"]), ws + int(raw["t1"])
        xy = kv["kpts"].copy()
        xy[kv["vis"] < 0.2] = np.nan
        with np.errstate(all="ignore"):
            A = _kick_side_indices(xy)[0]
        view, probs = kv, (probs_all or {}).get("kick_view")
        extra = dict(role="KICKER", win_name="kick window",
                     limb_desc="Kick-window detection: ankle speed rel. to hip",
                     peak_desc="PEAK: max ankle-hip distance",
                     phase_names=("preparation", "chamber / extension", "recovery"))
        warnings = []

    thr_hi, thr_lo = _thresholds(sp_s)
    norm = normalize_kpts(view["kpts"][..., :2], view["vis"], tau=0.5)
    # performer-selection scores of the view that was used, per track
    scores = [float(s) for s in view.get("scores", [])] or [0.0] * len(tracks)
    return {
        "best": best, "scores": scores, "invalid": int(view["invalid_for_scoring"]),
        "kpts": kpts, "vis": vis, "win_s": ws, "win_e": we,
        "sp_s": sp_s, "thr_hi": thr_hi, "thr_lo": thr_lo,
        "norm": norm, "norm_sc": _norm_panel_scale(norm),
        "kick_type": label, "probs": probs, "probs_all": probs_all,
        "score": (acc4, expr6, total10), "top_issues": details["top_issues"], "warnings": warnings,
        "t0": t0, "tp": tp, "t1": t1, "A": A, "measures": None, **extra,
    }


# ----------------------------------------------------------------------------
# Result panel (Unicode)
# ----------------------------------------------------------------------------
def build_result_panel(a: dict, n_people: int) -> np.ndarray:
    h = PANEL_H[3]
    img = np.full((h, PW, 3), BG, np.uint8)
    acc4, expr6, total10 = a["score"]
    texts = [(a["kick_type"].upper(), (8, 4), GREEN, 18),
             (f"{total10:.2f}/10   (chính xác {acc4:.2f}/4, biểu hiện {expr6:.2f}/6)", (170, 7), WHITE, 14)]

    if a["probs"]:
        for k, (c, p) in enumerate(a["probs"].items()):
            x, y = 8 + (k % 2) * 236, 32 + (k // 2) * 18
            cv2.rectangle(img, (x + 92, y + 4), (x + 92 + int(100 * p), y + 14),
                          GREEN if c == a["kick_type"] else GREY, -1)
            texts += [(c, (x, y), WHITE, 13), (f"{p:.2f}", (x + 198, y), WHITE, 13)]
    else:
        texts.append(("loại động tác do người dùng chỉ định (--kick)", (8, 36), GREY, 13))

    for k, it in enumerate(a["top_issues"][:2]):
        texts.append((f"-{it['deduct']:.2f}  {it['label']}", (8, 70 + k * 17), (0, 200, 255), 13))

    if a["warnings"] or a["invalid"]:
        msg = "CẢNH BÁO: " + ("không chắc ai là người thực hiện" if a["invalid"] else WARN_VI.get(a["warnings"][0], a["warnings"][0]))
        texts.append((msg[:80], (8, h - 22), RED, 12))
    else:
        texts.append((f"{n_people} người được theo dõi; bbox xanh = người thực hiện", (8, h - 22), GREY, 12))
    return draw_texts(img, texts)


# ----------------------------------------------------------------------------
def visualize(video: Path, task_path: Path, num_poses: int, models, kick: str | None) -> dict | None:
    print(f"[1/3] detecting people + pose: {video.name}")
    fps, W, H, tracks, frames = detect_and_track(video, task_path, num_poses)
    T = len(frames)
    if T <= 1 or not tracks:
        print("  [ERR] no pose detected or video too short")
        return None

    print("[2/3] recognition (4 classes) + scoring")
    a = analyse_kick4(video, tracks, fps, W, H, T, models, kick)

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
            pdet = next((d for d in frames[t] if d.get("track") == a["best"]), None)
            main = render_main(frame, t, frames[t], a, W, H, fps, T)
            right = np.vstack([
                render_3d(pdet["w33"] if pdet else None, t, fps, a["A"]),
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

    acc4, expr6, total10 = a["score"]
    result = {
        "video": str(video), "kick_type": a["kick_type"], "kick_source": "user" if kick else "model",
        "probabilities": a["probs_all"], "window": [a["win_s"], a["win_e"]], "impact_frame": int(a["tp"]),
        "score": {"accuracy_4": round(acc4, 2), "expression_6": round(expr6, 2), "total_10": round(total10, 2)},
        "top_issues": a["top_issues"], "warnings": a["warnings"], "invalid_for_scoring": a["invalid"],
    }
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    warn = "  [WARN] " + "; ".join(a["warnings"]) if a["warnings"] else ""
    print(f"  technique={a['kick_type']}  score={total10:.2f}/10  window={a['win_s']}-{a['win_e']}  "
          f"people={len(tracks)}{warn}")
    print(f"  saved: {out_path}\n  saved: {out_dir / 'peak.png'}\n  saved: {out_dir / 'result.json'}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize 4-class recognition (kicks + right hook) and scoring.")
    ap.add_argument("input", type=str, help="video file or folder of videos")
    ap.add_argument("--kick", type=str, default=None, choices=CLASS_CHOICES, help="known technique: skip recognition")
    ap.add_argument("--ckpt", type=str, nargs="+", default=[str(p) for p in DEFAULT_CKPTS])
    ap.add_argument("--task", type=str, default=str(DEFAULT_TASK))
    ap.add_argument("--num_poses", type=int, default=4)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    videos = list_videos(Path(args.input))
    if not videos:
        sys.exit(f"[ERR] no video found at {args.input}")
    task_path = ensure_task(Path(args.task))
    models = None if args.kick else load_models([Path(p) for p in args.ckpt])
    for v in videos:
        try:
            visualize(v, task_path, args.num_poses, models, args.kick)
        except Exception as e:
            print(f"[ERR] {v} | {e}")


if __name__ == "__main__":
    main()
