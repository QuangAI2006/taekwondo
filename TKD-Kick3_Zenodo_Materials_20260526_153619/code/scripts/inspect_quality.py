# -*- coding: utf-8 -*-
"""
inspect_quality.py (FULL OVERWRITE)

功能：
1) 扫描 data/sequences/{split} 下的 npz，导出 quality CSV
2) 随机抽取 N 个 invalid_for_scoring=1 的样本：
   - 只渲染动作窗口（win_start~win_end），前后可加缓冲
   - 用 PIL 绘制中文提示（避免 cv2.putText 变 ????）
   - 输出到 data/vis_out/quality/{split}/invalid_videos/

用法：
python scripts/inspect_quality.py --split val --csv
python scripts/inspect_quality.py --split val --csv --render_invalid 20
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import csv
import os

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parents[1]
SEQ_ROOT = BASE_DIR / "data" / "sequences"
VIS_ROOT = BASE_DIR / "data" / "vis_out" / "quality"

# 13点索引
HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(13)
LEFT_IDXS  = [LS, LE, LW, LH, LK, LA]
RIGHT_IDXS = [RS, RE, RW, RH, RK, RA]

EDGES = [
    (HEAD, LS), (HEAD, RS),
    (LS, LE), (LE, LW),
    (RS, RE), (RE, RW),
    (LS, LH), (RS, RH),
    (LH, LK), (LK, LA),
    (RH, RK), (RK, RA),
    (LH, RH),
]


def draw_skeleton(img, k, v):
    c_left  = (0, 220, 0)
    c_right = (0, 0, 255)
    c_edge  = (255, 255, 255)

    def col(i):
        if i in LEFT_IDXS:
            return c_left
        if i in RIGHT_IDXS:
            return c_right
        return c_edge

    for i in range(13):
        if v[i] > 0.1:
            x, y = int(k[i, 0]), int(k[i, 1])
            cv2.circle(img, (x, y), 3, col(i), -1)

    for i, j in EDGES:
        if v[i] > 0.1 and v[j] > 0.1:
            xi, yi = int(k[i, 0]), int(k[i, 1])
            xj, yj = int(k[j, 0]), int(k[j, 1])
            cv2.line(img, (xi, yi), (xj, yj), c_edge, 1)

    tags = {LH: "LH", RH: "RH", LK: "LK", RK: "RK", LA: "LA", RA: "RA"}
    for i, t in tags.items():
        if v[i] > 0.1:
            x, y = int(k[i, 0]), int(k[i, 1])
            cv2.putText(img, t, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col(i), 2, cv2.LINE_AA)


def _pick_cn_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Windows 常见中文字体候选。优先用微软雅黑。
    """
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",    # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",  # 微软雅黑粗体
        r"C:\Windows\Fonts\simhei.ttf",  # 黑体
        r"C:\Windows\Fonts\simsun.ttc",  # 宋体
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, font_size)
    # 找不到就退化，可能仍不支持中文，但至少不崩
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """
    按像素宽度自动换行（中文按字符切分）。
    """
    text = text.replace("\r", "")
    paragraphs = text.split("\n")
    lines: list[str] = []

    for para in paragraphs:
        if not para:
            lines.append("")
            continue

        cur = ""
        for ch in para:
            nxt = cur + ch
            bbox = draw.textbbox((0, 0), nxt, font=font)
            w = bbox[2] - bbox[0]
            if w <= max_width:
                cur = nxt
            else:
                if cur:
                    lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)

    return lines


def put_text_cn_block(
    frame_bgr: np.ndarray,
    text: str,
    org: tuple[int, int] = (16, 16),
    font_size: int = 24,
    color_bgr: tuple[int, int, int] = (30, 255, 30),
    bg_bgr: tuple[int, int, int] = (0, 0, 0),
    bg_alpha: float = 0.45,
    max_width_ratio: float = 0.92,
    line_spacing: int = 6,
) -> np.ndarray:
    """
    用 PIL 在 OpenCV(BGR) 帧上绘制中文多行文本，并加半透明背景提升可读性。
    """
    h, w = frame_bgr.shape[:2]
    x0, y0 = org
    max_w = int(w * max_width_ratio)

    # BGR -> RGB -> PIL
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font = _pick_cn_font(font_size)

    # 自动换行
    lines = _wrap_text(draw, text, font, max_width=max_w)

    # 计算文本块大小
    line_heights = []
    line_widths = []
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    block_w = (max(line_widths) if line_widths else 0)
    block_h = sum(line_heights) + line_spacing * max(0, len(lines) - 1)

    # 背景框坐标
    pad = 10
    bx1, by1 = x0 - pad, y0 - pad
    bx2, by2 = x0 + block_w + pad, y0 + block_h + pad
    bx1, by1 = max(0, bx1), max(0, by1)
    bx2, by2 = min(w, bx2), min(h, by2)

    # 画半透明背景（用 overlay）
    overlay = pil_img.copy()
    overlay_draw = ImageDraw.Draw(overlay)
    # PIL RGB
    bg_rgb = (int(bg_bgr[2]), int(bg_bgr[1]), int(bg_bgr[0]))
    overlay_draw.rectangle([bx1, by1, bx2, by2], fill=bg_rgb)
    pil_img = Image.blend(pil_img, overlay, alpha=float(bg_alpha))
    draw = ImageDraw.Draw(pil_img)

    # 写文字
    rgb_color = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    yy = y0
    for ln, lh in zip(lines, line_heights):
        draw.text((x0, yy), ln, font=font, fill=rgb_color)
        yy += lh + line_spacing

    # PIL -> OpenCV
    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return out_bgr


def render_invalid(npz_path: Path, out_dir: Path, pad_sec: float = 0.3):
    """
    只渲染动作窗口（win_start~win_end），并用中文叠字。
    """
    data = np.load(npz_path, allow_pickle=True)
    kpts = data["kpts"]
    vis  = data["vis"]
    src  = str(data.get("src_video", ""))

    if not src:
        print(f"[WARN] npz 无 src_video：{npz_path}")
        return

    src_path = Path(src)
    if not src_path.exists():
        print(f"[WARN] src_video 不存在：{src_path}")
        return

    fps = float(data.get("fps", 25.0))
    if not fps or fps <= 1e-6:
        fps = 25.0

    # 读取窗口
    win_start = int(data.get("win_start", 0))
    win_end   = int(data.get("win_end", kpts.shape[0]))
    reason    = str(data.get("win_reason", "full"))

    pad = int(pad_sec * fps)
    s = max(0, win_start - pad)
    e = min(int(kpts.shape[0]), win_end + pad)
    if e <= s + 5:
        s, e = 0, int(kpts.shape[0])

    hint = str(data.get("reshoot_hint", "请重拍：确保踢击动作阶段脚踝全程可见"))
    cov_best = float(data.get("cov_ankle_best", 0.0))
    miss_best = float(data.get("max_missing_run_sec_best", 0.0))
    pre_roll = float(data.get("pre_roll_sec_est", 0.0))

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        print(f"[WARN] 无法打开视频：{src_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (src_path.stem + f"_invalid_{reason}_s{s}_e{e}.mp4")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # 跳到窗口起点
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)

    t = s
    while t < e:
        ok, frame = cap.read()
        if not ok:
            break

        draw_skeleton(frame, kpts[t], vis[t])

        text = (
            f"窗口：{reason}  帧：{t}/{e}  "
            f"pre_roll≈{pre_roll:.1f}s\n"
            f"cov_ankle_best={cov_best:.2f}  max_missing_run_sec_best={miss_best:.2f}s\n"
            f"{hint}"
        )
        frame = put_text_cn_block(frame, text, org=(16, 16), font_size=24, color_bgr=(30, 255, 30))

        writer.write(frame)
        t += 1

    cap.release()
    writer.release()
    print(f"[OK] render -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--csv", action="store_true", help="导出 CSV 到 data/vis_out/quality/{split}/quality.csv")
    ap.add_argument("--render_invalid", type=int, default=0, help="随机抽取 N 个 invalid 样本渲染窗口视频（0=不渲染）")
    ap.add_argument("--pad_sec", type=float, default=0.3, help="渲染窗口前后额外缓冲秒数（默认0.3）")
    args = ap.parse_args()

    seq_dir = SEQ_ROOT / args.split
    npzs = sorted(seq_dir.rglob("*.npz"))
    if not npzs:
        print(f"[WARN] 没找到 npz：{seq_dir}")
        return

    rows = []
    invalids = []
    for p in npzs:
        d = np.load(p, allow_pickle=True)
        row = {
            "npz": str(p.relative_to(SEQ_ROOT)),
            "src_video": str(d.get("src_video", "")),
            "cov_LA": float(d.get("cov_LA", 0.0)),
            "cov_RA": float(d.get("cov_RA", 0.0)),
            "cov_ankle_best": float(d.get("cov_ankle_best", 0.0)),
            "max_missing_run_sec_best": float(d.get("max_missing_run_sec_best", 0.0)),
            "invalid_for_scoring": int(d.get("invalid_for_scoring", 0)),
            "win_start": int(d.get("win_start", 0)),
            "win_end": int(d.get("win_end", 0)),
            "win_reason": str(d.get("win_reason", "")),
            "pre_roll_sec_est": float(d.get("pre_roll_sec_est", 0.0)),
            "reshoot_hint": str(d.get("reshoot_hint", "")),
        }
        rows.append(row)
        if row["invalid_for_scoring"] == 1:
            invalids.append(p)

    out_dir = VIS_ROOT / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        csv_path = out_dir / "quality.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] CSV -> {csv_path}")

    if args.render_invalid > 0 and invalids:
        k = min(args.render_invalid, len(invalids))
        picks = random.sample(invalids, k)
        vid_dir = out_dir / "invalid_videos"
        for p in picks:
            render_invalid(p, vid_dir, pad_sec=float(args.pad_sec))

    print(f"[INFO] total={len(rows)} invalid={len(invalids)} ({len(invalids)/max(1,len(rows)):.1%})")


if __name__ == "__main__":
    main()
