# scripts/eval_val_confusion_v2.py
# -*- coding: utf-8 -*-
"""eval_val_confusion_v2.py

用途：统一口径评估 val/test 的混淆矩阵，并支持对齐策略消融。

关键点：
1) 兼容多种 ckpt 格式：
   - {'model': state_dict, ...}
   - {'model_state': state_dict, ...}
   - 直接 state_dict
2) 以 ckpt 内的 label_map 为准生成 labels 顺序（labels[id] = class_name），保证打印/混淆矩阵不乱序。
3) align 与 decision：
   - align=uniform, decision=argmax：整段线性插值后单次前向，取 argmax（常规基线）。
   - align=gt, decision=argmax：用真实类别做相位对齐（上界口径，不可用于真实推理）。
   - align=phase, decision=sweep：对每个候选类别分别相位对齐并取该类概率，选最大（真实推理可用）。
   - align=phase, decision=two_pass：先 uniform argmax 得到粗预测，再按该预测相位对齐二次前向。

示例：
  python .\scripts\eval_val_confusion_v2.py --ckpt .\models\checkpoints\kick3_uni.pt --split val --align uniform --decision argmax
  python .\scripts\eval_val_confusion_v2.py --ckpt .\models\checkpoints\kick3_uni.pt --split val --align phase   --decision sweep
  python .\scripts\eval_val_confusion_v2.py --ckpt .\models\checkpoints\kick3_uni.pt --split val --align gt      --decision argmax
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
SEQ_ROOT = BASE_DIR / "data" / "sequences"
CKPT_DIR = BASE_DIR / "models" / "checkpoints"


# ----------------------------
# Constants
# ----------------------------
VTH = 0.2
HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(13)
def _estimate_body_x_axis(kpts: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """
    估计“身体横轴”（左右方向）的单位向量 body_x，用于横踢 pivot：
    优先用左右髋连线 RH-LH 的中位向量；髋不可用则用肩连线 RS-LS。
    返回 shape=(2,) 的单位向量；兜底为 (1,0)。
    """
    # 收集多个帧的“左右连线向量”，用中位数更稳
    vecs = []

    # hips
    m = (vis[:, LH] > VTH) & (vis[:, RH] > VTH)
    if np.any(m):
        v = kpts[m, RH, :2] - kpts[m, LH, :2]
        vecs.append(np.nanmedian(v, axis=0))

    # shoulders fallback
    m = (vis[:, LS] > VTH) & (vis[:, RS] > VTH)
    if (not vecs) and np.any(m):
        v = kpts[m, RS, :2] - kpts[m, LS, :2]
        vecs.append(np.nanmedian(v, axis=0))

    if not vecs:
        return np.array([1.0, 0.0], dtype=np.float32)

    vx = vecs[0].astype(np.float32)
    n = float(np.linalg.norm(vx))
    if not np.isfinite(n) or n < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return vx / n

# ----------------------------
# Temporal resize + phase align
# ----------------------------
def temporal_resize(arr: np.ndarray, L: int) -> np.ndarray:
    T = arr.shape[0]
    if T == L:
        return arr
    if T <= 1:
        return np.repeat(arr, L, axis=0)
    idxs = np.linspace(0, T - 1, num=L)
    i0 = np.floor(idxs).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (idxs - i0)[..., None]
    return arr[i0] * (1 - w) + arr[i1] * w


def _kick_side_indices(kpts: np.ndarray):
    def path_len(p: np.ndarray) -> float:
        if p.shape[0] <= 1:
            return 0.0
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        return float(np.nansum(d))

    return (RA, RK, RH, LA, LK, LH) if path_len(kpts[:, RA]) >= path_len(kpts[:, LA]) else (LA, LK, LH, RA, RK, RH)


def _pivot_index(kpts: np.ndarray, vis: np.ndarray, ktype: str) -> int:
    """
    枢轴帧（pivot）：
    - axe：攻击脚踝最高点（y 最小）
    - front/roundhouse：攻击脚踝与同侧髋关节距离最大（踢出最远）
    """
    T = kpts.shape[0]
    A, K, H, AS, KS, HS = _kick_side_indices(kpts)

    if T <= 1:
        return 0

    if ktype == "axe":
        ys = np.where(vis[:, A] > VTH, kpts[:, A, 1], np.inf)
        return int(np.argmin(ys))

    dist = np.full((T,), -1e9, np.float32)
    mask = (vis[:, A] > VTH) & (vis[:, H] > VTH)
    if np.any(mask):
        dist[mask] = np.linalg.norm(kpts[mask, A] - kpts[mask, H], axis=1)
        return int(np.argmax(dist))
    return 0



def phase_aligned_resize(
    feat: np.ndarray,
    kpts: np.ndarray,
    vis: np.ndarray,
    L: int,
    ktype: str,
    pre: float = 0.45,
    post: float = 0.55,
) -> np.ndarray:
    T = feat.shape[0]
    if T <= 1:
        return np.repeat(feat, L, axis=0)

    p = _pivot_index(kpts, vis, ktype)
    pre_len = int(round(pre * T))
    post_len = int(round(post * T))
    start = max(0, p - pre_len)
    end = min(T - 1, p + post_len)
    if end <= start:
        return temporal_resize(feat, L)

    idxs = np.linspace(start, end, num=L)
    i0 = np.floor(idxs).astype(int)
    i1 = np.minimum(i0 + 1, end)
    w = (idxs - i0)[..., None]
    return feat[i0] * (1 - w) + feat[i1] * w


# ----------------------------
# Features (match inference)
# ----------------------------
def build_frame_features(kpts: np.ndarray, vis: np.ndarray) -> np.ndarray:
    if kpts.ndim != 3:
        raise ValueError(f"kpts 期望 (T,13,D)，实际 {kpts.shape}")
    T, J, D0 = kpts.shape
    D = 2
    pts = kpts[:, :, :D].astype(np.float32)
    v = vis.astype(np.float32) if vis is not None else np.ones((T, J), np.float32)

    pts_m = pts.copy()
    pts_m[v < VTH] = np.nan

    center = np.zeros((T, D), np.float32)
    has_hip = (v[:, LH] > VTH) & (v[:, RH] > VTH)
    if np.any(has_hip):
        center[has_hip] = 0.5 * (pts_m[has_hip, LH] + pts_m[has_hip, RH])

    has_sh = (v[:, LS] > VTH) & (v[:, RS] > VTH)
    m2 = (~has_hip) & has_sh
    if np.any(m2):
        center[m2] = 0.5 * (pts_m[m2, LS] + pts_m[m2, RS])

    m3 = (~has_hip) & (~has_sh)
    if np.any(m3):
        with np.errstate(all="ignore"):
            cm = np.nanmean(pts_m[m3], axis=1)
        cm = np.nan_to_num(cm, nan=0.0, posinf=0.0, neginf=0.0)
        center[m3] = cm

    with np.errstate(all="ignore"):
        sh = np.linalg.norm(pts_m[:, LS, :] - pts_m[:, RS, :], axis=1, keepdims=True)
        hip = np.linalg.norm(pts_m[:, LH, :] - pts_m[:, RH, :], axis=1, keepdims=True)
    sh[~has_sh] = np.nan
    hip[~has_hip] = np.nan
    ones = np.full_like(sh, 1.0, np.float32)
    with np.errstate(all="ignore"):
        scale = np.nanmax(np.stack([sh, hip, ones], axis=0), axis=0)
    scale = np.where(np.isfinite(scale), scale, 1.0).astype(np.float32)

    pts_n = (pts_m - center[:, None, :]) / scale[:, None, :]
    vel = np.zeros_like(pts_n)
    vel[1:] = pts_n[1:] - pts_n[:-1]
    feat = np.concatenate([pts_n, vel], axis=-1).reshape(T, -1)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------
# Model (standard transformer)
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class KickTransformer(nn.Module):
    def __init__(self, in_dim=52, d_model=128, nhead=4, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, 256, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        h = self.proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        h = self.norm(h)
        return self.head(h.mean(dim=1))


# ----------------------------
# CKPT loading (format-robust)
# ----------------------------
def _normalize_label_map(label_map: Dict) -> Dict[str, int]:
    # 支持 {cls: id} 或 {id: cls}
    if not isinstance(label_map, dict):
        return {}
    # 判断 key 类型
    k0 = next(iter(label_map.keys())) if len(label_map) else None
    if k0 is None:
        return {}
    # case A: cls->id
    if isinstance(k0, str):
        out: Dict[str, int] = {}
        for k, v in label_map.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                pass
        return out

    # case B: id->cls
    out: Dict[str, int] = {}
    for k, v in label_map.items():
        try:
            out[str(v)] = int(k)
        except Exception:
            pass
    return out


def labels_from_label_map(label_map: Dict[str, int]) -> List[str]:
    if not label_map:
        return []
    max_id = max(int(i) for i in label_map.values())
    labels = [""] * (max_id + 1)
    for c, i in label_map.items():
        labels[int(i)] = str(c)
    if any(x == "" for x in labels):
        # 兜底：压缩去空
        labels = [x for x in labels if x != ""]
    return labels


def load_ckpt(ckpt_path: Path, device: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], Dict]:
    obj = torch.load(str(ckpt_path), map_location=device)

    state_dict = None
    meta = {}
    label_map = {}
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            state_dict = obj["model"]
            meta = obj
            label_map = _normalize_label_map(obj.get("label_map", {}))
        elif "model_state" in obj and isinstance(obj["model_state"], dict):
            state_dict = obj["model_state"]
            meta = obj
            label_map = _normalize_label_map(obj.get("label_map", {}))
        else:
            # 可能是 optimizer 等杂项包了一层
            # 常见：{'state_dict':..., ...}
            for k in ("state_dict", "net", "weights"):
                if k in obj and isinstance(obj[k], dict):
                    state_dict = obj[k]
                    meta = obj
                    label_map = _normalize_label_map(obj.get("label_map", {}))
                    break
    if state_dict is None:
        if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
            state_dict = obj
        else:
            raise RuntimeError(f"无法解析 ckpt：{ckpt_path}（不包含可用 state_dict）")

    # label_map fallback：若 ckpt 没写，用同目录 json
    if not label_map:
        json_path = ckpt_path.with_suffix("")
        cand = ckpt_path.with_name(ckpt_path.stem + "_label_map.json")
        if cand.exists():
            try:
                label_map = _normalize_label_map(json.loads(cand.read_text(encoding="utf-8")))
            except Exception:
                label_map = {}
        else:
            # fallback 到通用 label
            label_map = {"front": 0, "roundhouse": 1, "axe": 2}

    return state_dict, label_map, meta


# ----------------------------
# Data scan
# ----------------------------
def find_class_from_path(p: Path, classes: List[str]) -> Optional[str]:
    parts = [x.lower() for x in p.parts]
    for c in classes:
        if c.lower() in parts:
            return c
    return None


def scan_npz(split_dir: Path, classes: List[str]) -> List[Path]:
    if not split_dir.exists():
        raise FileNotFoundError(f"找不到 split 目录：{split_dir}")
    out = []
    for p in sorted(split_dir.rglob("*.npz")):
        if find_class_from_path(p, classes) is not None:
            out.append(p)
    return out


# ----------------------------
# Inference helpers
# ----------------------------
@torch.no_grad()
def forward_prob(model: nn.Module, x_feat: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(x_feat).float().unsqueeze(0).to(device)
    logits = model(x)
    return torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()


def predict_one(
    model: nn.Module,
    frames: int,
    classes: List[str],
    label_map: Dict[str, int],
    kpts: np.ndarray,
    vis: np.ndarray,
    align: str,
    decision: str,
    true_cls: Optional[str],
    device: str,
) -> int:
    feat = build_frame_features(kpts, vis)

    if align == "uniform":
        feat_u = temporal_resize(feat, frames)
        prob = forward_prob(model, feat_u, device)
        return int(prob.argmax())

    if align == "gt":
        if true_cls is None:
            raise ValueError("align=gt 需要 true_cls")
        feat_pa = phase_aligned_resize(feat, kpts, vis, frames, true_cls)
        prob = forward_prob(model, feat_pa, device)
        return int(prob.argmax())

    if align == "phase":
        if decision == "sweep":
            best_cls = None
            best_score = -1.0
            # sweep：对每个候选类别对齐后取该类概率
            for c in classes:
                feat_pa = phase_aligned_resize(feat, kpts, vis, frames, c)
                prob = forward_prob(model, feat_pa, device)
                cid = int(label_map[c])
                sc = float(prob[cid])
                if sc > best_score:
                    best_score = sc
                    best_cls = c
            assert best_cls is not None
            return int(label_map[best_cls])

        if decision == "two_pass":
            # 先粗预测，再按粗预测类别相位对齐二次前向
            feat_u = temporal_resize(feat, frames)
            prob0 = forward_prob(model, feat_u, device)
            pred0 = int(prob0.argmax())
            pred0_cls = classes[pred0] if pred0 < len(classes) else classes[0]
            feat_pa = phase_aligned_resize(feat, kpts, vis, frames, pred0_cls)
            prob1 = forward_prob(model, feat_pa, device)
            return int(prob1.argmax())

        raise ValueError("align=phase 仅支持 decision=sweep/two_pass")

    raise ValueError(f"未知 align：{align}")


def confusion_and_metrics(y_true: List[int], y_pred: List[int], labels: List[str]):
    K = len(labels)
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < K and 0 <= p < K:
            cm[t, p] += 1
    acc = float(np.trace(cm) / max(1, cm.sum()))
    metrics = {}
    for i, name in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = float(tp / max(1, tp + fp))
        recall = float(tp / max(1, tp + fn))
        support = int(cm[i, :].sum())
        metrics[name] = dict(precision=precision, recall=recall, support=support)
    return acc, cm, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--align", type=str, default="uniform", choices=["uniform", "phase", "gt"])
    ap.add_argument("--decision", type=str, default="argmax", choices=["argmax", "sweep", "two_pass"])
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    state_dict, label_map, meta = load_ckpt(ckpt_path, device)
    labels = labels_from_label_map(label_map)

    if not labels:
        raise RuntimeError("无法从 label_map 得到 labels")

    # 从 meta 取超参
    frames = int(meta.get("frames", 96))
    in_dim = int(meta.get("in_dim", 52))
    d_model = int(meta.get("d_model", 128))
    nhead = int(meta.get("nhead", 4))
    num_layers = int(meta.get("num_layers", 2))
    dropout = float(meta.get("dropout", 0.2))

    model = KickTransformer(in_dim=in_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, num_classes=len(labels), dropout=dropout).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    split_dir = SEQ_ROOT / args.split
    files = scan_npz(split_dir, labels)
    if len(files) == 0:
        raise RuntimeError(f"未找到 .npz：{split_dir}（且路径中需要包含类别目录名：{labels}）")

    print(f"[INFO] ckpt={ckpt_path.name} device={device}")
    print(f"[INFO] hparams: in_dim={in_dim} d_model={d_model} nhead={nhead} num_layers={num_layers} dropout={dropout} frames={frames}")
    print(f"[INFO] split={args.split} align={args.align} decision={args.decision}")

    if args.align == "uniform" and args.decision != "argmax":
        raise SystemExit("align=uniform 只支持 decision=argmax")
    if args.align in ("gt",) and args.decision != "argmax":
        raise SystemExit("align=gt 只支持 decision=argmax")
    if args.align == "phase" and args.decision == "argmax":
        raise SystemExit("align=phase 不支持 decision=argmax（请用 sweep 或 two_pass）")

    y_true, y_pred = [], []
    skipped_invalid = 0
    for p in files:
        d = np.load(p, allow_pickle=True)
        if "invalid_for_scoring" in d.files:
            inv = int(np.asarray(d["invalid_for_scoring"]).reshape(-1)[0])
            if inv == 1:
                skipped_invalid += 1
                continue
        kpts = d["kpts"].astype(np.float32)
        vis = d["vis"].astype(np.float32) if "vis" in d.files else np.ones(kpts.shape[:2], np.float32)

        cls = find_class_from_path(p, labels)
        if cls is None:
            continue

        tid = int(label_map[cls])
        pid = predict_one(
            model,
            frames=frames,
            classes=labels,
            label_map=label_map,
            kpts=kpts,
            vis=vis,
            align=args.align,
            decision=args.decision,
            true_cls=cls,
            device=device,
        )
        y_true.append(tid)
        y_pred.append(pid)

    acc, cm, metrics = confusion_and_metrics(y_true, y_pred, labels)

    print(f"[INFO] evaluated samples={len(y_true)} acc={acc:.3f} (skipped_invalid={skipped_invalid})")
    print("\nPer-class metrics:")
    for name in labels:
        m = metrics[name]
        print(f"  {name:<10s} precision={m['precision']:.3f}  recall={m['recall']:.3f}  support={m['support']}")

    # predicted distribution
    print("\nPredicted distribution:")
    pred_cnt = {name: 0 for name in labels}
    for p in y_pred:
        if 0 <= p < len(labels):
            pred_cnt[labels[p]] += 1
    for name in labels:
        n = pred_cnt[name]
        r = n / max(1, len(y_pred))
        print(f"  {name:<10s} n={n:4d}  ratio={r:.3f}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    head = "".join([f"{n:>12s}" for n in labels])
    print(" " * 12 + head)
    for i, name in enumerate(labels):
        row = "".join([f"{int(cm[i, j]):12d}" for j in range(len(labels))])
        print(f"{name:<12s}{row}")


if __name__ == "__main__":
    main()
