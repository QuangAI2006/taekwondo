# -*- coding: utf-8 -*-
"""
hook_build_dataset.py
Build the right-hook (hook_right) pose dataset from pre-cut clips.

- Input : a folder of clips (default: <workspace>/clips, i.e. E:/martial arts/clips)
- Output: data/sequences_hook/<split>/hook_right/<clip>.npz   (wrist-window punch view)
          outputs/hook/split_manifest.csv
- The existing kick data under data/sequences is NOT touched.

Split is sample-level (clip-level) and deterministic (--seed). Participant identifiers
are not available, so this is NOT a participant-independent split.

Usage:
    python scripts/hook_build_dataset.py
    python scripts/hook_build_dataset.py --clips_dir "E:/martial arts/clips" --skip_existing
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from hook_pose import LW, RW, VIDEO_EXTS, extract_views, save_view

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIPS = ROOT.parents[1] / "clips"
DEFAULT_OUT = ROOT / "data" / "sequences_hook"
MANIFEST = ROOT / "outputs" / "hook" / "split_manifest.csv"


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def safe_stem(p: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", p.stem).strip("_")


def assign_splits(n: int, val_ratio: float, test_ratio: float, seed: int) -> list[str]:
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * test_ratio)))
    order = np.random.default_rng(seed).permutation(n)
    splits = ["train"] * n
    for i in order[:n_val]:
        splits[i] = "val"
    for i in order[n_val:n_val + n_test]:
        splits[i] = "test"
    return splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips_dir", type=str, default=str(DEFAULT_CLIPS))
    ap.add_argument("--out_root", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--class_name", type=str, default="hook_right")
    ap.add_argument("--val_ratio", type=float, default=1 / 6)
    ap.add_argument("--test_ratio", type=float, default=1 / 6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", type=str, default=str(ROOT / "models" / "pose_landmarker_full.task"))
    ap.add_argument("--num_poses", type=int, default=4)
    ap.add_argument("--wrist", type=str, default="R", choices=["R", "L", "auto"],
                    help="wrist that drives the action window (label is known: hook_right -> R)")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    clips = sorted((p for p in Path(args.clips_dir).iterdir() if p.is_file() and p.suffix in VIDEO_EXTS),
                   key=natural_key)
    if not clips:
        raise SystemExit(f"[ERR] no clips found in {args.clips_dir}")

    splits = assign_splits(len(clips), args.val_ratio, args.test_ratio, args.seed)
    out_root = Path(args.out_root)
    print(f"[INFO] clips={len(clips)} train={splits.count('train')} val={splits.count('val')} "
          f"test={splits.count('test')} -> {out_root}")

    rows = []
    for clip, split in zip(clips, splits):
        out_npz = out_root / split / args.class_name / f"{safe_stem(clip)}.npz"
        if args.skip_existing and out_npz.exists():
            view = dict(np.load(out_npz, allow_pickle=True))
            status = "skipped_existing"
        else:
            wrist = {"R": RW, "L": LW, "auto": None}[args.wrist]
            _, view = extract_views(Path(args.task), clip, num_poses=args.num_poses, wrist=wrist)
            save_view(view, out_npz)
            status = "extracted"

        T = int(view["kpts"].shape[0])
        row = {
            "clip": clip.name, "split": split, "class": args.class_name,
            "npz": str(out_npz.relative_to(ROOT)), "frames_total": int(view.get("frames_total", T)),
            "win_s": int(view.get("win_s", 0)), "win_e": int(view.get("win_e", max(0, T - 1))), "frames_used": T,
            "attack_wrist": str(view.get("attack_wrist", "")),
            "invalid_for_scoring": int(view["invalid_for_scoring"]), "reason": str(view.get("reason", "")),
            "status": status,
        }
        rows.append(row)
        print(f"[{split:5s}] {clip.name} -> {row['npz']} | win={row['win_s']}-{row['win_e']}/{row['frames_total']}"
              f" | wrist={row['attack_wrist']}" + (f" | INVALID: {row['reason']}" if row["invalid_for_scoring"] else ""))

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[SAVE] {MANIFEST}")


if __name__ == "__main__":
    main()
