# -*- coding: utf-8 -*-
import csv
import shutil
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
SEQ_ROOT = BASE_DIR / "data" / "sequences"
WRONG_CSV = BASE_DIR / "data" / "vis_out" / "val_confusion" / "wrong_val.csv"
OUT_DIR = BASE_DIR / "data" / "vis_out" / "val_confusion" / "wrong_src_videos"

def main():
    if not WRONG_CSV.exists():
        raise FileNotFoundError(f"找不到 {WRONG_CSV}，请先运行 eval_val_confusion.py --outT ...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    with open(WRONG_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            npz_rel = row["npz_rel"]
            true_c = row["true"]
            pred_c = row["pred"]

            npz_path = SEQ_ROOT / npz_rel
            if not npz_path.exists():
                continue

            d = np.load(npz_path, allow_pickle=True)
            src = str(d.get("src_video", ""))
            if not src:
                continue

            srcp = Path(src)
            if not srcp.exists():
                # src_video 偶尔是相对路径时，尝试补全
                alt = BASE_DIR / src
                if alt.exists():
                    srcp = alt
                else:
                    continue

            dst_name = f"{true_c}_TO_{pred_c}__{srcp.name}"
            dst = OUT_DIR / dst_name
            try:
                shutil.copy2(srcp, dst)
                copied += 1
            except Exception:
                pass

    print(f"[OK] copied wrong videos -> {OUT_DIR} (count={copied})")

if __name__ == "__main__":
    main()
