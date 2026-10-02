# scripts/score_segments.py
"""
Chấm điểm các ĐOẠN của video dài (vd video bài giảng: mỗi bài tập ~30 s, người tập lặp động tác nhiều lần).

  đoạn [start, end] -> MediaPipe (action_pose.extract_pose) -> tách từng lần thực hiện (rep)
                    -> luật chấm của động tác (action_scorers + scorers_*.py) cho TỪNG rep -> CSV + tổng hợp

File đoạn (CSV, UTF-8): video,start,end,action[,label][,guard][,side]
  - video : đường dẫn (tương đối với --root)
  - start/end : giây hoặc m:ss
  - action : tên luật chấm (front, roundhouse, back, high_block, ...); xem --list
  - label : tên bài hiển thị (vd "Đá sau (phải)")
  - guard/side : tùy chọn cho luật đấm

Ví dụ:
  python code/scripts/score_segments.py --segments code/tutorial_segments.csv --root "E:/martial arts" \
         --out outputs/tutorial_scores --pose_cache outputs/tutorial_scores/pose_cache --workers 6
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
PKG_ROOT = SCRIPTS_DIR.parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from action_pose import _interp_nan_1d  # noqa: E402
from action_scorers import SCORERS, canonical, score_action  # noqa: E402
from extract_pose_mediapipe import VTH  # noqa: E402
from infer_action import TASK_PATH, load_plugins  # noqa: E402

HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(13)

# tách rep theo chuyển động của đầu chi nào (mặc định: cả tay và chân)
REP_EFFECTORS = {"legs": ((LA, RA), (LH, RH)), "arms": ((LW, RW), (LS, RS))}


def parse_time(s: str) -> float:
    s = str(s).strip()
    if ":" in s:
        m, sec = s.split(":", 1)
        return 60.0 * float(m) + float(sec)
    return float(s)


def load_segments(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("video") and not r["video"].lstrip().startswith("#")]
    for r in rows:
        r["start_s"], r["end_s"] = parse_time(r["start"]), parse_time(r["end"])
        r["action"] = canonical(r["action"])
        r.setdefault("label", r["action"])
    return rows


# ---------- pose ----------
def _cache_name(video: Path, s: float, e: float) -> str:
    return f"{video.stem[:60]}__{s:.1f}_{e:.1f}.npz".replace(" ", "_").replace("#", "")


def _cut_h264(video: Path, s: float, e: float, out: Path) -> None:
    """Cắt đoạn [s, e] sang H.264 bằng ffmpeg (giải mã phần cứng nếu có). OpenCV giải mã AV1 bằng CPU rất
    chậm (~15 frame/s), nên với video AV1 (vd tải từ YouTube) cắt trước rồi mới chạy MediaPipe."""
    import subprocess
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-hwaccel", "auto", "-ss", f"{s:.3f}",
           "-i", str(video), "-t", f"{e - s:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-an", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not out.is_file():
        raise RuntimeError("ffmpeg lỗi: " + r.stderr.strip()[-300:])


def get_pose(video: Path, s: float, e: float, cache_dir: Optional[Path], transcode: bool = False) -> Dict:
    from action_pose import extract_pose
    cp = cache_dir / _cache_name(video, s, e) if cache_dir else None
    if cp is not None and cp.is_file():
        d = np.load(cp)
        return {k: d[k] for k in d.files}
    if cp is not None and cache_dir.is_dir():  # dùng lại pose đã lưu của một đoạn dài hơn chứa [s, e]
        stem = _cache_name(video, 0.0, 0.0).rsplit("__", 1)[0]
        for f in sorted(cache_dir.glob(stem + "__*.npz")):
            try:
                s0, e0 = (float(x) for x in f.stem.rsplit("__", 1)[1].split("_"))
            except ValueError:
                continue
            if s0 <= s + 1e-6 and e0 >= e - 1e-6:
                d = np.load(f)
                pose = {k: d[k] for k in d.files}
                fps = float(pose["fps"])
                a, b = int(round((s - s0) * fps)), int(round((e - s0) * fps))
                for k in ("kpts", "vis", "world"):
                    pose[k] = pose[k][a:b]
                return pose
    if transcode:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "seg.mp4"
            _cut_h264(video, s, e, tmp)
            pose = extract_pose(tmp, TASK_PATH, num_poses=4, select="limbs")
    else:
        pose = extract_pose(video, TASK_PATH, num_poses=4, select="limbs", start_s=s, end_s=e)
    if cp is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cp, **pose)
    return pose


# ---------- tách rep ----------
def rep_energy(kpts: np.ndarray, vis: np.ndarray, fps: float, use: Tuple[str, ...] = ("arms", "legs")) -> np.ndarray:
    """Tốc độ lớn nhất của các đầu chi (so với vai/hông), đơn vị chiều dài thân/giây, làm mượt 0.3 s."""
    T = kpts.shape[0]
    torso = np.linalg.norm(0.5 * (kpts[:, LS] + kpts[:, RS]) - 0.5 * (kpts[:, LH] + kpts[:, RH]), axis=1)
    L = float(np.nanmedian(torso)) if np.any(np.isfinite(torso)) else 1.0
    en = np.zeros(T)
    for name in use:
        effs, anchor = REP_EFFECTORS[name]
        anc = 0.5 * (kpts[:, anchor[0]] + kpts[:, anchor[1]])
        for e in effs:
            rel = (kpts[:, e] - anc) / max(L, 1e-6)
            rel[vis[:, e] <= VTH] = np.nan
            rel = np.stack([_interp_nan_1d(rel[:, c]) for c in range(2)], axis=1)
            sp = np.linalg.norm(np.diff(rel, axis=0), axis=1) * fps
            en = np.maximum(en, np.r_[sp[:1], sp] if sp.size else np.zeros(T))
    k = max(3, int(round(0.3 * fps)) | 1)
    return np.convolve(en, np.ones(k) / k, mode="same") if T >= k else en


def ankle_signal(kpts: np.ndarray, vis: np.ndarray, fps: float) -> np.ndarray:
    """Chênh lệch độ cao 2 cổ chân (× chiều dài thân, 2D), làm mượt 0.2 s — mỗi cú đá là 1 đỉnh; khi bước/nhảy
    hai cổ chân lên xuống cùng nhau nên tín hiệu gần như không đổi. (Không dùng khoảng cách cổ chân–hông: chân
    trụ đứng thẳng cũng xa hông ngang chân đá duỗi thẳng.)"""
    T = kpts.shape[0]
    torso = np.linalg.norm(0.5 * (kpts[:, LS] + kpts[:, RS]) - 0.5 * (kpts[:, LH] + kpts[:, RH]), axis=1)
    L = float(np.nanmedian(torso)) if np.any(np.isfinite(torso)) else 1.0
    yL, yR = kpts[:, LA, 1].astype(float), kpts[:, RA, 1].astype(float)
    yL[vis[:, LA] <= VTH] = np.nan
    yR[vis[:, RA] <= VTH] = np.nan
    sig = np.abs(_interp_nan_1d(yL) - _interp_nan_1d(yR)) / max(L, 1e-6)
    k = max(3, int(round(0.2 * fps)) | 1)
    return np.convolve(sig, np.ones(k) / k, mode="same") if T >= k else sig


def rep_config(action: str, fn) -> Dict:
    """Cách tách rep cho từng luật. Trên hàm chấm có thể đặt:
    rep_signal = "ankle" | rep_signal_fn(pose, action, **opts) -> tín hiệu (cao = đang ở điểm đến của động tác)
    | rep_effectors (đầu chi cho tín hiệu năng lượng) | rep_cfg (ghi đè min_gap_s, pre_s, post_s, ...)."""
    cfg = {"signal": "energy", "use": ("arms", "legs"), "min_gap_s": 0.7, "pre_s": 1.0, "post_s": 1.2}
    if getattr(fn, "rep_signal", None) == "ankle" or action in ("front", "roundhouse", "side", "axe"):
        cfg.update(signal="ankle", min_gap_s=0.8)
    elif action == "hook":
        cfg.update(use=("arms",))
    elif action == "straight":  # đấm thẳng trái – phải xen kẽ
        from hand_scoring import alternating_side, alternating_signal
        cfg.update(signal=alternating_signal, min_gap_s=0.5, pre_s=0.8, post_s=1.0, clip=False,
                   side_fn=alternating_side)
    if getattr(fn, "rep_effectors", None):
        cfg["use"] = fn.rep_effectors
    if getattr(fn, "rep_signal_fn", None):
        cfg["signal"] = fn.rep_signal_fn
    cfg.update(getattr(fn, "rep_cfg", {}))
    return cfg


def split_reps(pose: Dict, action: str = "", signal="energy", use: Tuple[str, ...] = ("arms", "legs"),
               min_gap_s: float = 0.7, pre_s: float = 1.0, post_s: float = 1.2,
               opts: Optional[Dict] = None, clip: bool = True) -> List[Tuple[int, int, int]]:
    """
    Mỗi đỉnh của tín hiệu = 1 rep.
      signal="energy": tốc độ đầu chi, đỉnh cao >= 35% phân vị 95;
      signal="ankle" hoặc hàm (pose, action) -> tín hiệu vị trí: đỉnh vượt trung vị + 40% (p95 - trung vị).
    Các đỉnh cách nhau >= min_gap_s (giữ đỉnh cao hơn). Rep i = [đỉnh - pre_s, đỉnh + post_s], cắt tại điểm
    thấp nhất của tín hiệu giữa 2 đỉnh kề nhau để mỗi đoạn chỉ chứa 1 rep (clip=False: không cắt — dùng cho
    động tác liên tục như đấm xen kẽ, khi luật chấm được chỉ định sẵn tay ra đòn). Trả về [(s, e, peak), ...].
    """
    kpts, vis, fps = pose["kpts"], pose["vis"], float(pose["fps"])
    T = kpts.shape[0]
    if T < 8:
        return []
    if signal == "energy":
        sig = rep_energy(kpts, vis, fps, use)
        thr = 0.35 * float(np.percentile(sig, 95))
    else:
        sig = ankle_signal(kpts, vis, fps) if signal == "ankle" else np.asarray(signal(pose, action, **(opts or {})), float)
        med = float(np.median(sig))
        thr = med + 0.4 * (float(np.percentile(sig, 95)) - med)
    if not np.all(np.isfinite(sig)) or np.ptp(sig) <= 0:
        return []
    cand = [t for t in range(1, T - 1) if sig[t] >= thr and sig[t] >= sig[t - 1] and sig[t] > sig[t + 1]]
    gap = int(round(min_gap_s * fps))
    peaks: List[int] = []
    for t in sorted(cand, key=lambda i: -sig[i]):  # giữ đỉnh cao, bỏ đỉnh phụ quá gần
        if all(abs(t - p) >= gap for p in peaks):
            peaks.append(t)
    peaks.sort()
    if clip:
        bounds = [0] + [p + int(np.argmin(sig[p:q + 1])) for p, q in zip(peaks[:-1], peaks[1:])] + [T - 1]
    else:
        bounds = [0] * len(peaks) + [T - 1]
    pre, post = int(round(pre_s * fps)), int(round(post_s * fps))
    reps = []
    for i, p in enumerate(peaks):
        s = max(bounds[i], p - pre)
        e = min(bounds[i + 1] if clip else T - 1, p + post)
        if e - s + 1 >= 8:
            reps.append((int(s), int(e), int(p)))
    return reps


def _numeric(raw: Dict, prefix: str = "") -> Dict:
    """Chỉ số thô dạng số (để kiểm tra / hiệu chỉnh ngưỡng); dict lồng nhau (đòn phối hợp) được làm phẳng."""
    out = {}
    for k, v in (raw or {}).items():
        if isinstance(v, dict):
            out.update(_numeric(v, f"{prefix}{k}."))
        elif isinstance(v, (bool, np.bool_)):
            out[prefix + k] = int(v)
        elif isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
            out[prefix + k] = float(v)
    return out


def _slice(pose: Dict, s: int, e: int) -> Dict:
    out = dict(pose)
    for k in ("kpts", "vis", "world"):
        out[k] = pose[k][s:e + 1]
    return out


# ---------- chấm 1 đoạn ----------
def score_segment(seg: Dict, root: Path, cache_dir: Optional[Path], transcode: bool = False) -> Dict:
    load_plugins()
    video = (root / seg["video"]).resolve()
    pose = get_pose(video, seg["start_s"], seg["end_s"], cache_dir, transcode)
    fps = float(pose["fps"])
    info = SCORERS.get(seg["action"])
    opts = {k: seg[k] for k in ("guard", "side") if seg.get(k)}
    # luật đá: chấm trên cả clip của 1 rep (dạng dữ liệu public của tác giả) và chọn đúng chân đá
    opts.update(kick_window=seg.get("kick_window", "none"), fix_leg=True)
    reps_out = []
    T = pose["kpts"].shape[0]
    side_fn = None
    win_s = getattr(info.fn, "window_s", None) if info else None
    if win_s:  # động tác giữ yên (tấn): chia thành các cửa sổ liên tiếp
        n = max(1, int(round(win_s * fps)))
        reps = [(s, min(T - 1, s + n - 1), s) for s in range(0, T, n) if min(T - 1, s + n - 1) - s + 1 >= n // 2]
    else:
        cfg = rep_config(seg["action"], info.fn)
        side_fn = cfg.pop("side_fn", None)
        reps = split_reps(pose, seg["action"], opts=opts, **cfg)
    for i, (s, e, p) in enumerate(reps, 1):
        try:
            ropts = dict(opts, side=side_fn(pose, p)) if side_fn and not seg.get("side") else opts
            r = score_action(seg["action"], _slice(pose, s, e), **ropts)
        except Exception as ex:  # 1 rep lỗi không làm dừng cả đoạn
            from action_scorers import ScoreResult
            r = ScoreResult(action=seg["action"], valid=False, reason=f"Lỗi: {ex}")
        reps_out.append({"rep": i, "t_start": seg["start_s"] + s / fps, "t_end": seg["start_s"] + e / fps,
                         "t_peak": seg["start_s"] + (s + r.phases["t_peak"]) / fps
                         if r.valid and r.phases.get("t_peak") is not None else None,
                         "valid": r.valid, "total": r.total, "acc": r.acc, "expr": r.expr,
                         "items": {it["code"]: it.get("got") for it in r.acc_items + r.expr_items},
                         "maxi": {it["code"]: it["maxi"] for it in r.acc_items + r.expr_items},
                         "labels": {it["code"]: it["label"] for it in r.acc_items + r.expr_items},
                         "reasons": {it["code"]: it.get("reason", "") for it in r.acc_items + r.expr_items},
                         "side": r.side_text, "warnings": r.warnings if r.valid else [r.reason],
                         "raw": _numeric(r.raw)})
    return {"seg": {k: v for k, v in seg.items()}, "fps": fps, "frames": int(pose["kpts"].shape[0]),
            "n_tracks": int(pose.get("n_tracks", 0)), "reps": reps_out}


def summarize(results: List[Dict]) -> List[Dict]:
    """Gộp theo bài (video + label): trung vị điểm các rep hợp lệ."""
    by: Dict[Tuple[str, str, str], List[Dict]] = {}
    for r in results:
        by.setdefault((r["seg"]["video"], r["seg"]["action"], r["seg"]["label"]), []).append(r)
    out = []
    for (video, action, label), rs in by.items():
        reps = [x for r in rs for x in r["reps"]]
        ok = [x for x in reps if x["valid"]]
        row = {"video": Path(video).name, "action": action, "label": label, "segments": len(rs), "reps": len(reps),
               "valid": len(ok)}
        if ok:
            tot = np.array([x["total"] for x in ok])
            row.update(total_median=float(np.median(tot)), total_p25=float(np.percentile(tot, 25)),
                       total_p75=float(np.percentile(tot, 75)), total_min=float(tot.min()), total_max=float(tot.max()),
                       acc_median=float(np.median([x["acc"] for x in ok])),
                       expr_median=float(np.median([x["expr"] for x in ok])))
            items = {}
            for x in ok:
                for c, g in x["items"].items():
                    if g is not None:
                        items.setdefault(c, []).append(g)
            row["items"] = {c: (float(np.median(v)), ok[0]["maxi"][c], ok[0]["labels"][c]) for c, v in items.items()}
        out.append(row)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Chấm điểm từng rep trong các đoạn của video dài")
    ap.add_argument("--segments", required=True, help="CSV: video,start,end,action[,label][,guard][,side]")
    ap.add_argument("--root", default=".", help="thư mục gốc của cột video")
    ap.add_argument("--out", default="outputs/segment_scores", help="thư mục kết quả")
    ap.add_argument("--pose_cache", default="", help="thư mục lưu pose từng đoạn (chạy lại không cần MediaPipe)")
    ap.add_argument("--only", default="", help="chỉ chấm các action này (phẩy)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--author_window", action="store_true",
                    help="luật đá: cắt thêm đoạn của tác giả bên trong mỗi rep (như infer_action/predict_video). "
                         "Mặc định chấm cả rep — trên đoạn cắt, mục thu chân A3 gần như luôn = 0")
    ap.add_argument("--transcode", action="store_true",
                    help="cắt từng đoạn sang H.264 bằng ffmpeg trước khi chạy MediaPipe (nên dùng với video AV1)")
    ap.add_argument("--list", action="store_true", help="in danh sách luật chấm rồi thoát")
    args = ap.parse_args(argv)

    load_plugins()
    if args.list:
        for k in sorted(SCORERS):
            print(f"{k:16s} {SCORERS[k].label_vi}")
        return
    segs = load_segments(Path(args.segments))
    if args.author_window:
        for sg in segs:
            sg["kick_window"] = "ankle"
    if args.only:
        keep = {canonical(a) for a in args.only.split(",")}
        segs = [s for s in segs if s["action"] in keep]
    missing = sorted({s["action"] for s in segs if s["action"] not in SCORERS})
    if missing:
        sys.exit(f"Chưa có luật chấm cho: {missing}")
    root = Path(args.root)
    cache = Path(args.pose_cache) if args.pose_cache else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(score_segment, s, root, cache, args.transcode) for s in segs]
        for s, f in zip(segs, futs):
            try:
                r = f.result()
            except Exception as e:
                print(f"[LỖI] {s['video']} {s['start']}-{s['end']} {s['label']}: {e}", flush=True)
                continue
            ok = [x for x in r["reps"] if x["valid"]]
            med = f"{np.median([x['total'] for x in ok]):.2f}" if ok else "—"
            print(f"[{len(results) + 1}/{len(segs)}] {s['label']:<36s} {s['start']}-{s['end']}  "
                  f"rep {len(ok)}/{len(r['reps'])}  trung vị {med}", flush=True)
            results.append(r)

    # per-rep CSV
    codes: List[str] = []
    for r in results:
        for x in r["reps"]:
            for c in x["items"]:
                if c not in codes:
                    codes.append(c)
    with open(out / "reps.csv", "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["video", "label", "action", "rep", "t_start", "t_end", "t_peak", "side", "valid",
                     "total10", "acc4", "expr6", *codes, "warnings_or_reason"])
        for r in results:
            for x in r["reps"]:
                wr.writerow([Path(r["seg"]["video"]).name, r["seg"]["label"], r["seg"]["action"], x["rep"],
                             f"{x['t_start']:.2f}", f"{x['t_end']:.2f}",
                             "" if x["t_peak"] is None else f"{x['t_peak']:.2f}", x["side"], int(x["valid"]),
                             *("" if v is None else f"{v:.2f}" for v in (x["total"], x["acc"], x["expr"])),
                             *("" if x["items"].get(c) is None else f"{x['items'][c]:.2f}" for c in codes),
                             " | ".join(w for w in x["warnings"] if w)])
    summ = summarize(results)
    with open(out / "summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["video", "label", "action", "segments", "reps", "valid", "total_median", "total_p25", "total_p75",
                     "total_min", "total_max", "acc_median", "expr_median"])
        for row in summ:
            wr.writerow([row["video"], row["label"], row["action"], row["segments"], row["reps"], row["valid"],
                         *(f"{row[k]:.2f}" if k in row else "" for k in
                           ("total_median", "total_p25", "total_p75", "total_min", "total_max",
                            "acc_median", "expr_median"))])
    (out / "results.json").write_text(json.dumps({"segments": results, "summary": summ}, ensure_ascii=False,
                                                  indent=1, default=float), encoding="utf-8")
    print(f"\n[SAVED] {out / 'reps.csv'}\n[SAVED] {out / 'summary.csv'}\n[SAVED] {out / 'results.json'}")


if __name__ == "__main__":
    main()
