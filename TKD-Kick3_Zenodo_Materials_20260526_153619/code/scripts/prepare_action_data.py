# scripts/prepare_action_data.py
"""
Chuẩn bị dữ liệu train model nhận dạng cho các động tác MỚI (đấm, chỏ, ... hoặc thêm đòn đá).

Cấu trúc video đầu vào (1 video = 1 lần thực hiện động tác, nên cắt ngắn 1–3 giây):
  A) <videos>/<động tác>/*.mp4                -> tự chia train/val theo --val_ratio
  B) <videos>/train/<động tác>/*.mp4 và <videos>/val/<động tác>/*.mp4  -> giữ nguyên cách chia
Tên thư mục = tên động tác. Đặt trùng tên luật chấm (hook, straight, front, roundhouse, axe, side, ...)
để khi chạy infer_action.py động tác được chấm điểm tự động.

Đầu ra (đọc được bằng train_actions.py):
  <out>/train/<động tác>/<video>.npz, <out>/val/<động tác>/<video>.npz   (kpts, vis, world, fps đã cắt đoạn)
  <out>/meta.json   (window_mode, danh sách động tác, số mẫu)

Cắt đoạn bằng action_pose.action_window(mode) — ĐÚNG hàm infer_action.py dùng lúc chạy, nên dữ liệu train
và dữ liệu lúc chạy luôn nhất quán.

Ví dụ:
  python code/scripts/prepare_action_data.py --videos "E:/martial arts/my_videos" --out data/my_sequences
  python code/scripts/prepare_action_data.py --videos "E:/martial arts/my_videos" --out data/my_sequences --include_public_kicks
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
PKG_ROOT = SCRIPTS_DIR.parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from action_pose import WINDOW_MODES, action_window, extract_pose  # noqa: E402

TASK_PATH = PKG_ROOT / "models" / "pose_landmarker_full.task"
PUBLIC_KICKS = PKG_ROOT / "data" / "processed_pose_sequences"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def list_videos(root: Path, val_ratio: float, seed: int):
    """-> list of (split, class, path)."""
    items = []
    if (root / "train").is_dir():
        for split in ("train", "val"):
            for cdir in sorted(p for p in (root / split).iterdir() if p.is_dir()) if (root / split).is_dir() else []:
                items += [(split, cdir.name, v) for v in sorted(cdir.iterdir()) if v.suffix.lower() in VIDEO_EXTS]
        return items
    rng = random.Random(seed)
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        vids = sorted(v for v in cdir.iterdir() if v.suffix.lower() in VIDEO_EXTS)
        rng.shuffle(vids)
        n_val = max(1, int(round(val_ratio * len(vids)))) if len(vids) >= 2 else 0
        items += [("val", cdir.name, v) for v in vids[:n_val]] + [("train", cdir.name, v) for v in vids[n_val:]]
    return items


def save_seq(path: Path, kpts, vis, world, fps, s, e, mode, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(kpts=kpts[s:e + 1].astype(np.float32), vis=vis[s:e + 1].astype(np.float32),
                fps=np.float32(fps), win_s=np.int32(s), win_e=np.int32(e), window_mode=mode, **extra)
    if world is not None:
        data["world"] = world[s:e + 1].astype(np.float32)
    np.savez_compressed(path, **data)


def main() -> None:
    ap = argparse.ArgumentParser(description="Video theo thư mục động tác -> chuỗi pose để train")
    ap.add_argument("--videos", required=True, help="thư mục video (cấu trúc A hoặc B, xem đầu file)")
    ap.add_argument("--out", required=True, help="thư mục đầu ra")
    ap.add_argument("--window", choices=[m for m in WINDOW_MODES], default="motion",
                    help="cách cắt đoạn (mặc định motion — hợp cho nhiều loại động tác)")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include_public_kicks", action="store_true",
                    help="gộp 765 mẫu đá public (front/roundhouse/axe) của TKD-Kick3 vào train/val")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    counts = defaultdict(Counter)
    skipped = []

    for split, cls, video in list_videos(Path(args.videos), args.val_ratio, args.seed):
        dst = out / split / cls / (video.stem + ".npz")
        if args.skip_existing and dst.exists():
            counts[split][cls] += 1
            continue
        try:
            pose = extract_pose(video, TASK_PATH, num_poses=4, select="limbs")
        except Exception as ex:
            skipped.append((video.name, f"lỗi: {ex}"))
            continue
        k, v, w, fps = pose["kpts"], pose["vis"], pose["world"], float(pose["fps"])
        if k.shape[0] < 8 or not np.any(np.isfinite(k)):
            skipped.append((video.name, "không phát hiện người / quá ngắn"))
            continue
        s, e, mode = action_window(k, v, fps, args.window)
        save_seq(dst, k, v, w, fps, s, e, mode, n_tracks=np.int32(pose["n_tracks"]),
                 invalid_for_scoring=np.int32(pose["invalid_for_scoring"]))
        counts[split][cls] += 1
        flag = "  [nhiều người]" if pose["invalid_for_scoring"] else ""
        print(f"[OK] {split}/{cls}/{video.name}: {k.shape[0]} frame -> đoạn f{s}-f{e}{flag}")

    if args.include_public_kicks:
        for split in ("train", "val"):
            for f in sorted((PUBLIC_KICKS / split).rglob("*.npz")):
                cls = f.parent.name
                z = np.load(f, allow_pickle=True)
                k = z["kpts"] if "kpts" in z.files else z["keypoints"]
                v = z["vis"] if "vis" in z.files else z["visibility"]
                fps = float(z["fps"])
                # dữ liệu public đã được tác giả cắt theo cổ chân; cắt thêm bằng cùng hàm như lúc chạy
                s, e, mode = action_window(k, v, fps, args.window)
                save_seq(out / split / cls / ("public_" + f.name), k, v, None, fps, s, e, mode)
                counts[split][cls] += 1
        print(f"[OK] đã gộp dữ liệu đá public từ {PUBLIC_KICKS}")

    classes = sorted(set(counts["train"]) | set(counts["val"]))
    meta = {"window_mode": args.window, "classes": classes,
            "counts": {sp: dict(counts[sp]) for sp in ("train", "val")},
            "include_public_kicks": args.include_public_kicks, "skipped": skipped}
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Xong: {out} (cắt đoạn: {args.window}) ===")
    print(f"{'động tác':<14s}{'train':>7s}{'val':>6s}")
    for c in classes:
        print(f"{c:<14s}{counts['train'][c]:>7d}{counts['val'][c]:>6d}")
    for name, why in skipped:
        print(f"[BỎ QUA] {name}: {why}")
    few = [c for c in classes if counts["train"][c] < 20]
    if few:
        print(f"[!] Ít mẫu train (<20): {few} — model khó học tốt các động tác này.")


if __name__ == "__main__":
    main()
