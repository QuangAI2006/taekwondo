# -*- coding: utf-8 -*-
"""
predict_video_kick4.py
Recognize and score videos with the 4-class model (front / roundhouse / axe / hook_right).
The original predict_video.py (3 kicks) is left unchanged and still works as before.

Pipeline per video:
  1. One MediaPipe pass (hook_pose.extract_views) -> two views of the same track:
       kick view  : ankle-based window, identical to extract_pose_mediapipe.py
       punch view : wrist-based window
  2. Recognition (ensemble of kick4 checkpoints, uniform alignment):
       p_kick = best kick-class probability on the kick view
       p_hook = hook_right probability on the punch view
     -> hook_right if p_hook > p_kick, else the best kick class
  3. Scoring: kicks -> scoring_api.score_one_video (kick view)
              hook  -> hook_scoring.score_hook   (punch view)

Outputs: outputs/predictions_kick4/<video_stem>/{result.json, pose_kick.npz, pose_punch.npz}

Usage:
    python scripts/predict_video_kick4.py path\\to\\video.mp4
    python scripts/predict_video_kick4.py path\\to\\folder
    python scripts/predict_video_kick4.py video.mp4 --kick hook_right   (skip recognition)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from eval_kick3_baselines import build_feat_52, forward_one, normalize_kpts, resample_kpts  # noqa: E402
from hook_pose import RW, extract_views, save_view  # noqa: E402
from hook_scoring import score_hook  # noqa: E402
from predict_video import ensure_task, list_videos, load_models  # noqa: E402
from scoring_api import score_one_video  # noqa: E402

CKPT_DIR = ROOT / "models" / "checkpoints" / "kick4_hook"
DEFAULT_CKPTS = [CKPT_DIR / f"kick4_lstm_uniform_seed{s}.pt" for s in (0, 1, 2)]
DEFAULT_TASK = ROOT / "models" / "pose_landmarker_full.task"
OUT_ROOT = ROOT / "outputs" / "predictions_kick4"
HOOK = "hook_right"
CLASS_CHOICES = ["front", "roundhouse", "axe", HOOK]


def ensemble_prob(models, kpts: np.ndarray, vis: np.ndarray) -> np.ndarray:
    kpts_n = normalize_kpts(kpts[..., :2], vis, tau=0.5)
    probs = []
    for model, _, T_out in models:
        kpts_a, _ = resample_kpts(kpts_n, vis, T_out)
        probs.append(forward_one(model, build_feat_52(kpts_a), device="cpu"))
    return np.mean(probs, axis=0)


def recognize(models, kick: dict, punch: dict) -> tuple[str, dict]:
    classes = models[0][1]
    if any(m[1] != classes for m in models):
        raise ValueError("checkpoints disagree on class order")
    if HOOK not in classes:
        raise ValueError(f"checkpoints have no '{HOOK}' class: {classes} (use predict_video.py for 3-class models)")

    pk = ensemble_prob(models, kick["kpts"], kick["vis"]) if len(kick["kpts"]) >= 2 else np.zeros(len(classes))
    pp = ensemble_prob(models, punch["kpts"], punch["vis"]) if len(punch["kpts"]) >= 2 else np.zeros(len(classes))
    h = classes.index(HOOK)
    kick_ids = [i for i in range(len(classes)) if i != h]
    best_kick = max(kick_ids, key=lambda i: pk[i])

    label = HOOK if pp[h] > pk[best_kick] else classes[best_kick]
    info = {
        "kick_view": {c: float(p) for c, p in zip(classes, pk)},
        "punch_view": {c: float(p) for c, p in zip(classes, pp)},
        "decision": f"p_hook(punch view)={pp[h]:.3f} vs p_{classes[best_kick]}(kick view)={pk[best_kick]:.3f}",
    }
    return label, info


def process(video: Path, task: Path, num_poses: int, models, kick: str | None) -> dict:
    out_dir = OUT_ROOT / video.stem
    kv, pv = extract_views(task, video, num_poses=num_poses)
    save_view(kv, out_dir / "pose_kick.npz")
    save_view(pv, out_dir / "pose_punch.npz")

    result = {"video": str(video), "frames_total": int(pv.get("frames_total", len(pv["kpts"])))}
    if len(kv["kpts"]) < 2 and len(pv["kpts"]) < 2:
        result["error"] = "no pose detected or video too short"
        return result

    if kick:
        result["kick_type"], result["kick_source"] = kick, "user"
    else:
        result["kick_type"], result["probabilities"] = recognize(models, kv, pv)
        result["kick_source"] = "model"

    view = pv if result["kick_type"] == HOOK else kv
    result["view"] = "punch" if view is pv else "kick"
    result["frames_used"] = int(len(view["kpts"]))
    result["invalid_for_scoring"] = int(view["invalid_for_scoring"])
    result["invalid_reason"] = str(view.get("reason", ""))

    if result["kick_type"] == HOOK:
        # score the right wrist even if another wrist moved more (label says right hook)
        if pv.get("attack_wrist") != "R":
            _, pv_r = extract_views(task, video, num_poses=num_poses, wrist=RW)
            view = pv_r
        acc4, expr6, total10, details = score_hook(view)
        result["warnings"] = details["warnings"]
    else:
        acc4, expr6, total10, details = score_one_video(view, result["kick_type"])

    result["score"] = {"accuracy_4": round(acc4, 2), "expression_6": round(expr6, 2), "total_10": round(total10, 2)}
    result["top_issues"] = details["top_issues"]
    result["tips"] = details["tips"]
    result["items"] = details["acc_items"] + details["expr_items"]

    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def print_result(r: dict) -> None:
    print("=" * 60)
    print(f"Video: {r['video']}")
    if "error" in r:
        print(f"  [ERR] {r['error']}")
        return
    if r["kick_source"] == "model":
        p = r["probabilities"]
        view = p["punch_view"] if r["kick_type"] == HOOK else p["kick_view"]
        print(f"  Technique : {r['kick_type']}   (" + "  ".join(f"{c}={v:.2f}" for c, v in view.items()) + ")")
        print(f"              {p['decision']}")
    else:
        print(f"  Technique : {r['kick_type']}   (given by user)")
    s = r["score"]
    print(f"  Score     : {s['total_10']:.2f}/10  (accuracy {s['accuracy_4']:.2f}/4, expression {s['expression_6']:.2f}/6)")
    if r["invalid_for_scoring"]:
        print(f"  [WARN] performer selection uncertain ({r['invalid_reason']}); treat score with caution")
    for w in r.get("warnings", []):
        print(f"  [WARN] {w}")
    if r["top_issues"]:
        print("  Main deductions:")
        for it in r["top_issues"]:
            print(f"    - [{it['code']}] {it['label']}: -{it['deduct']:.2f}/{it['maxi']:.1f}  {it['reason']}")
            print(f"        tip: {it['tip']}")
    print(f"  Saved     : {OUT_ROOT / Path(r['video']).stem / 'result.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Recognize and score Taekwondo kicks + right hook punch.")
    ap.add_argument("input", type=str, help="video file or folder of videos")
    ap.add_argument("--kick", type=str, default=None, choices=CLASS_CHOICES,
                    help="known technique: skip recognition and score as this type")
    ap.add_argument("--ckpt", type=str, nargs="+", default=[str(p) for p in DEFAULT_CKPTS],
                    help="kick4 checkpoint(s); probabilities are averaged")
    ap.add_argument("--task", type=str, default=str(DEFAULT_TASK), help="PoseLandmarker .task file")
    ap.add_argument("--num_poses", type=int, default=4)
    args = ap.parse_args()

    # Windows consoles default to cp1252; tips are Vietnamese / Chinese
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    videos = list_videos(Path(args.input))
    if not videos:
        sys.exit(f"[ERR] no video found at {args.input}")

    task = ensure_task(Path(args.task))
    models = None if args.kick else load_models([Path(p) for p in args.ckpt])
    print(f"[INFO] videos={len(videos)}  recognition="
          + (f"off (--kick {args.kick})" if args.kick else f"{len(models)} kick4 checkpoint(s)"))
    for v in videos:
        try:
            print_result(process(v, task, args.num_poses, models, args.kick))
        except Exception as e:
            print(f"[ERR] {v} | {e}")


if __name__ == "__main__":
    main()
