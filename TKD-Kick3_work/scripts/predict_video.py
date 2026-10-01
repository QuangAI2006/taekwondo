# -*- coding: utf-8 -*-
"""
predict_video.py
Predict the kick type of unlabeled videos and score them (single-video inference).

Pipeline per video:
  1. MediaPipe PoseLandmarker -> 13-joint sequence (same as extract_pose_mediapipe.py)
  2. Recognition with an ensemble of checkpoints (default: 3 LSTM uniform seeds)
  3. Rule-based scoring (scoring_api.score_one_video), Accuracy 4 + Expression 6

Does NOT touch data/sequences. Outputs go to outputs/predictions/<video_stem>/.

Usage:
    python scripts/predict_video.py path\\to\\video.mp4
    python scripts/predict_video.py path\\to\\folder_of_videos
    python scripts/predict_video.py video.mp4 --kick front      (skip recognition, score as front)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import List

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from extract_pose_mediapipe import VIDEO_EXTS, build_landmarker, extract_one_video  # noqa: E402
from eval_kick3_baselines import (  # noqa: E402
    DEFAULT_CLASSES, build_feat_52, build_model, forward_one, normalize_kpts, resample_kpts,
)
from scoring_api import score_one_video  # noqa: E402

TASK_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
DEFAULT_TASK = ROOT / "models" / "pose_landmarker_full.task"
CKPT_DIR = ROOT / "models" / "checkpoints" / "sample_level"
DEFAULT_CKPTS = [CKPT_DIR / f"sample_lstm_uniform_seed{s}.pt" for s in (0, 1, 2)]
OUT_ROOT = ROOT / "outputs" / "predictions"


def ensure_task(task_path: Path) -> Path:
    if task_path.is_file():
        return task_path
    print(f"[INFO] downloading MediaPipe model -> {task_path}")
    task_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TASK_URL, str(task_path))
    return task_path


def list_videos(inp: Path) -> List[Path]:
    if inp.is_file():
        return [inp]
    return sorted(p for p in inp.rglob("*") if p.is_file() and p.suffix in VIDEO_EXTS)


def load_models(ckpt_paths: List[Path]):
    models = []
    for p in ckpt_paths:
        ckpt = torch.load(str(p), map_location="cpu")
        hparams = ckpt.get("hparams", {})
        classes = ckpt.get("classes", DEFAULT_CLASSES)
        model = build_model(ckpt.get("model_type", ""), hparams, num_classes=len(classes))
        model.load_state_dict(ckpt.get("model", ckpt), strict=True)
        model.eval()
        models.append((model, classes, int(hparams.get("T", 96))))
    return models


def recognize(models, kpts: np.ndarray, vis: np.ndarray) -> tuple[str, dict]:
    kpts_n = normalize_kpts(kpts, vis, tau=0.5)
    classes = models[0][1]
    probs = []
    for model, cls, T_out in models:
        if cls != classes:
            raise ValueError(f"checkpoints disagree on class order: {cls} vs {classes}")
        kpts_a, _ = resample_kpts(kpts_n, vis, T_out)
        probs.append(forward_one(model, build_feat_52(kpts_a), device="cpu"))
    prob = np.mean(probs, axis=0)
    return classes[int(np.argmax(prob))], {c: float(p) for c, p in zip(classes, prob)}


def process(video: Path, landmarker_args: dict, models, kick: str | None) -> dict:
    out_dir = OUT_ROOT / video.stem
    out_npz = out_dir / "pose.npz"
    landmarker = build_landmarker(**landmarker_args)
    try:
        extract_one_video(landmarker, video, out_npz, verbose=False)
    finally:
        landmarker.close()

    z = np.load(str(out_npz), allow_pickle=True)
    kpts = z["kpts"].astype(np.float32)
    vis = z["vis"].astype(np.float32)
    result = {
        "video": str(video),
        "frames_used": int(kpts.shape[0]),
        "invalid_for_scoring": int(z["invalid_for_scoring"]),
        "invalid_reason": str(z["reason"]) if "reason" in z.files else "",
    }
    if kpts.shape[0] < 2:
        result["error"] = "no pose detected or video too short"
        return result

    if kick:
        result["kick_type"] = kick
        result["kick_source"] = "user"
    else:
        result["kick_type"], result["probabilities"] = recognize(models, kpts, vis)
        result["kick_source"] = "model"

    acc4, expr6, total10, details = score_one_video(z, result["kick_type"])
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
        probs = "  ".join(f"{c}={p:.2f}" for c, p in r["probabilities"].items())
        print(f"  Kick type : {r['kick_type']}   ({probs})")
    else:
        print(f"  Kick type : {r['kick_type']}   (given by user)")
    s = r["score"]
    print(f"  Score     : {s['total_10']:.2f}/10  (accuracy {s['accuracy_4']:.2f}/4, expression {s['expression_6']:.2f}/6)")
    if r["invalid_for_scoring"]:
        print(f"  [WARN] kicker selection uncertain ({r['invalid_reason']}); treat score with caution")
    if r["top_issues"]:
        print("  Main deductions:")
        for it in r["top_issues"]:
            print(f"    - [{it['code']}] {it['label']}: -{it['deduct']:.2f}/{it['maxi']:.1f}  {it['reason']}")
            print(f"        tip: {it['tip']}")
    print(f"  Saved     : {OUT_ROOT / Path(r['video']).stem / 'result.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Recognize and score Taekwondo kick videos.")
    ap.add_argument("input", type=str, help="video file or folder of videos")
    ap.add_argument("--kick", type=str, default=None, choices=["front", "roundhouse", "axe"],
                    help="known kick type: skip recognition and score as this type")
    ap.add_argument("--ckpt", type=str, nargs="+", default=[str(p) for p in DEFAULT_CKPTS],
                    help="checkpoint(s) for recognition; probabilities are averaged")
    ap.add_argument("--task", type=str, default=str(DEFAULT_TASK), help="PoseLandmarker .task file")
    ap.add_argument("--num_poses", type=int, default=4)
    args = ap.parse_args()

    videos = list_videos(Path(args.input))
    if not videos:
        sys.exit(f"[ERR] no video found at {args.input} (supported: {', '.join(sorted(set(e.lower() for e in VIDEO_EXTS)))})")

    landmarker_args = {"task_path": ensure_task(Path(args.task)), "num_poses": args.num_poses}
    models = None if args.kick else load_models([Path(p) for p in args.ckpt])

    print(f"[INFO] videos={len(videos)}  recognition={'off (--kick ' + args.kick + ')' if args.kick else f'{len(models)} checkpoint(s)'}")
    for v in videos:
        try:
            print_result(process(v, landmarker_args, models, args.kick))
        except Exception as e:
            print(f"[ERR] {v} | {e}")


if __name__ == "__main__":
    main()
