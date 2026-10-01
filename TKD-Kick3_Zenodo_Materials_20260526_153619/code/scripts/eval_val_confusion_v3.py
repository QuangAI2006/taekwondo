# -*- coding: utf-8 -*-
"""
eval_val_confusion_v3_fixed.py

用途：统一口径评估 val/test 的混淆矩阵，并支持对齐策略消融（uniform / gt / phase）。

v3_fixed 修复点：
- 修复 NameError: CLASSES 未定义：统一使用函数入参 classes。
- 修复 phase_aligned_resize 调用参数缺失：补齐 (kpts, vis)。
- 增强 ckpt 兼容：支持 {'model'}/{ 'model_state' }/直接 state_dict，并兼容 input_proj/pos_enc 命名。

典型用法：
  python .\\scripts\\eval_val_confusion_v3.py --ckpt .\\models\\checkpoints\\kick3_uni.pt --split val --align uniform --decision argmax
  python .\\scripts\\eval_val_confusion_v3.py --ckpt .\\models\\checkpoints\\kick3_uni.pt --split val --align phase   --decision sweep
  python .\\scripts\\eval_val_confusion_v3.py --ckpt .\\models\\checkpoints\\kick3_pa.pt  --split val --align gt      --decision argmax
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

# ----------------------------
# Constants / joints (13 points)
# ----------------------------
VTH = 0.2
HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(13)

# ----------------------------
# Feature building (same idea as train/infer)
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


def build_frame_features(kpts: np.ndarray, vis: np.ndarray, target_dim: int = 52) -> np.ndarray:
    """
    输入 kpts: (T,13,2/3)  vis: (T,13)
    输出 feat: (T, in_dim) 默认 in_dim=52 (=13*2*(pos+vel))
    """
    if kpts.ndim != 3:
        raise ValueError(f"kpts 期望 (T,13,D)，实际 {kpts.shape}")
    T, J, D0 = kpts.shape
    if J != 13:
        raise ValueError(f"本脚本默认 13 点骨架，实际 J={J}")

    # target_dim 仅支持 52(2D) 或 78(3D)
    if int(target_dim) == 52:
        D = min(2, D0)
    elif int(target_dim) == 78:
        D = min(3, D0)
    else:
        # 兼容：传 2/3 当作坐标维
        if int(target_dim) in (2, 3):
            D = min(int(target_dim), D0)
        else:
            raise ValueError(f"target_dim 仅支持 52/78 或 2/3，但拿到 {target_dim}")

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
# Phase-aligned resize (pivot selection)
# ----------------------------
def _kick_side_indices(kpts: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    """
    在 2D 关键点上估计“攻击侧”：看左右踝轨迹长度，较大者视为踢腿侧。
    返回：(A, K, H, AS, KS, HS) = (踢腿踝,踢腿膝,踢腿髋,支撑踝,支撑膝,支撑髋)
    """
    def path_len(p: np.ndarray) -> float:
        if p.shape[0] <= 1:
            return 0.0
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        return float(np.nansum(d))

    la = path_len(kpts[:, LA])
    ra = path_len(kpts[:, RA])
    if ra >= la:
        return (RA, RK, RH, LA, LK, LH)
    return (LA, LK, LH, RA, RK, RH)


def _pivot_index(kpts: np.ndarray, vis: np.ndarray, ktype: str) -> int:
    A, K, H, AS, KS, HS = _kick_side_indices(kpts)
    T = kpts.shape[0]

    if ktype == "axe":
        ys = np.where(vis[:, A] > VTH, kpts[:, A, 1], np.inf)
        p = int(np.argmin(ys))  # 踝最高（y 最小）
        return max(0, min(T - 1, p))

    # 其它：踝-髋最远
    dist = np.full((T,), -1e9, np.float32)
    mask = (vis[:, A] > VTH) & (vis[:, H] > VTH)
    if np.any(mask):
        dist[mask] = np.linalg.norm(kpts[mask, A] - kpts[mask, H], axis=1)
    p = int(np.argmax(dist))
    return max(0, min(T - 1, p))


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
# Model (Transformer encoder)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class KickTransformer(nn.Module):
    """
    与 train_kick3_transformer_ablation.py 的 KickTransformer 一致（proj/pos/encoder/norm/head）。
    num_classes 可为 3 或 4。
    """
    def __init__(self, in_dim=52, d_model=128, nhead=4, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, 256, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        h = self.norm(h)
        return self.head(h.mean(dim=1))

# ----------------------------
# Checkpoint loading
# ----------------------------
def _extract_hparams(ckpt: dict) -> Dict[str, float]:
    hp = {}
    if "hparams" in ckpt and isinstance(ckpt["hparams"], dict):
        hp.update(ckpt["hparams"])
    # 兼容：直接顶层写入
    for k in ("frames", "in_dim", "d_model", "nhead", "num_layers", "dropout"):
        if k in ckpt:
            hp[k] = ckpt[k]
    return hp


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    兼容旧 ckpt 命名：
      - input_proj.* -> proj.*
      - pos_enc.pe   -> pos.pe
    """
    keys = list(state_dict.keys())
    need_input_proj = any(k.startswith("input_proj.") for k in keys)
    need_pos_enc = any(k.startswith("pos_enc.") for k in keys)

    if not (need_input_proj or need_pos_enc):
        return state_dict

    new_sd: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("input_proj."):
            nk = "proj." + nk[len("input_proj.") :]
        if nk.startswith("pos_enc."):
            nk = "pos." + nk[len("pos_enc.") :]
        new_sd[nk] = v
    return new_sd


def load_ckpt(ckpt_path: Path, device: str):
    ckpt = torch.load(str(ckpt_path), map_location=device)

    state_dict_from = "raw"
    if isinstance(ckpt, dict) and ("model" in ckpt or "model_state" in ckpt):
        if "model" in ckpt:
            state_dict = ckpt["model"]
            state_dict_from = "model"
        else:
            state_dict = ckpt["model_state"]
            state_dict_from = "model_state"
        hp = _extract_hparams(ckpt)
        label_map = ckpt.get("label_map", None)
    else:
        state_dict = ckpt
        hp = {}
        label_map = None

    if not isinstance(state_dict, dict):
        raise ValueError("ckpt 中未找到可用的 state_dict")

    state_dict = _normalize_state_dict_keys(state_dict)

    if label_map is None:
        # 尝试从同目录的 label_map.json 读取
        cand = ckpt_path.parent / "kick3_label_map.json"
        if cand.exists():
            label_map = json.loads(cand.read_text(encoding="utf-8"))
        else:
            raise ValueError("ckpt 未包含 label_map，且未找到同目录 kick3_label_map.json")

    if not isinstance(label_map, dict):
        raise ValueError("label_map 格式不正确，期望 dict{name->id}")

    # 生成 labels（按 id 排序）
    max_id = max(int(i) for i in label_map.values())
    labels = [None] * (max_id + 1)
    for name, idx in label_map.items():
        labels[int(idx)] = str(name)
    if any(x is None for x in labels):
        # 非连续 id：退化为按 id 排序重建（会导致打印 label 与训练不一致的风险）
        items = sorted(((str(n), int(i)) for n, i in label_map.items()), key=lambda x: x[1])
        labels = [n for n, _ in items]

    # hparams default
    frames = int(hp.get("frames", 96))
    in_dim = int(hp.get("in_dim", 52))
    d_model = int(hp.get("d_model", 128))
    nhead = int(hp.get("nhead", 4))
    num_layers = int(hp.get("num_layers", 2))
    dropout = float(hp.get("dropout", 0.2))

    model = KickTransformer(
        in_dim=in_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=len(labels),
        dropout=dropout,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"[INFO] ckpt={ckpt_path.name} device={device} state_dict_from={state_dict_from}")
    print(f"[INFO] hparams: in_dim={in_dim} d_model={d_model} nhead={nhead} num_layers={num_layers} dropout={dropout} frames={frames}")
    return model, labels, label_map, frames, in_dim

# ----------------------------
# Label parsing from path
# ----------------------------
def find_class_from_path(p: Path, classes: List[str]) -> Optional[str]:
    parts = [x.lower() for x in p.parts]
    for c in classes:
        if c is None:
            continue
        if str(c).lower() in parts:
            return str(c)
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
    topk: int = 2,
    gap_tau: float = 0.12,
) -> int:
    feat = build_frame_features(kpts, vis)

    # uniform baseline
    if align == "uniform":
        feat_u = temporal_resize(feat, frames)
        prob = forward_prob(model, feat_u, device)
        return int(prob.argmax())

    # gt upper bound
    if align == "gt":
        if true_cls is None:
            raise ValueError("align=gt 需要 true_cls")
        feat_pa = phase_aligned_resize(feat, kpts, vis, frames, true_cls)
        prob = forward_prob(model, feat_pa, device)
        return int(prob.argmax())

    # phase inference
    if align == "phase":
        cand = [c for c in classes if c is not None and c in label_map]

        if decision == "sweep":
            best_id, best_score = 0, -1.0
            for cls in cand:
                feat_pa = phase_aligned_resize(feat, kpts, vis, frames, cls)
                prob = forward_prob(model, feat_pa, device)
                cid = int(label_map[cls])
                score = float(prob[cid])
                if score > best_score:
                    best_score = score
                    best_id = cid
            return int(best_id)

        if decision == "two_pass":
            feat_u = temporal_resize(feat, frames)
            prob0 = forward_prob(model, feat_u, device)
            pred0 = int(prob0.argmax())
            pred0_cls = classes[pred0] if 0 <= pred0 < len(classes) else None
            if pred0_cls is None:
                return int(pred0)
            feat_pa = phase_aligned_resize(feat, kpts, vis, frames, pred0_cls)
            prob1 = forward_prob(model, feat_pa, device)
            return int(prob1.argmax())

        if decision in ("two_pass_topk", "two_pass_gap"):
            feat_u = temporal_resize(feat, frames)
            prob0 = forward_prob(model, feat_u, device)
            order = np.argsort(prob0)[::-1]
            if order.size == 0:
                return 0

            # gap gating：gap 足够大时直接 two_pass（更快）
            if decision == "two_pass_gap" and order.size >= 2:
                gap = float(prob0[int(order[0])] - prob0[int(order[1])])
                if gap >= float(gap_tau):
                    pred0 = int(order[0])
                    pred0_cls = classes[pred0] if 0 <= pred0 < len(classes) else None
                    if pred0_cls is None:
                        return int(pred0)
                    feat_pa = phase_aligned_resize(feat, kpts, vis, frames, pred0_cls)
                    prob1 = forward_prob(model, feat_pa, device)
                    return int(prob1.argmax())

            k = int(topk) if topk is not None else 2
            k = max(1, min(k, int(order.size)))

            best_id, best_score = int(order[0]), -1.0
            for i in range(k):
                cid = int(order[i])
                cls_name = classes[cid] if 0 <= cid < len(classes) else None
                if cls_name is None or cls_name not in label_map:
                    continue
                feat_pa = phase_aligned_resize(feat, kpts, vis, frames, cls_name)
                prob1 = forward_prob(model, feat_pa, device)
                score = float(prob1[int(label_map[cls_name])])
                if score > best_score:
                    best_score = score
                    best_id = int(label_map[cls_name])
            return int(best_id)

        raise ValueError(f"align=phase 不支持 decision={decision}")

    raise ValueError(f"未知 align={align}")

# ----------------------------
# Metrics
# ----------------------------
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
        metrics[name] = (precision, recall, int(cm[i, :].sum()))
    return cm, acc, metrics

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="模型权重路径（.pt）")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--align", type=str, default="uniform", choices=["uniform", "gt", "phase"])
    ap.add_argument("--decision", type=str, default="argmax",
                    choices=["argmax", "sweep", "two_pass", "two_pass_topk", "two_pass_gap"],
                    help="uniform/gt 只支持 argmax；phase 支持 sweep/two_pass/two_pass_topk/two_pass_gap")
    ap.add_argument("--topk", type=int, default=2, help="two_pass_topk 的 k 值（默认2）")
    ap.add_argument("--gap_tau", type=float, default=0.12, help="two_pass_gap 的 gap 阈值（默认0.12）")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--include_invalid", action="store_true", help="是否包含 invalid_for_scoring=1 的样本（默认跳过）")
    args = ap.parse_args()

    if args.align in ("uniform", "gt") and args.decision != "argmax":
        raise ValueError("align=uniform/gt 只支持 decision=argmax")
    if args.align == "phase" and args.decision == "argmax":
        # 默认给一个更合理的 phase 决策
        args.decision = "sweep"

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda 不可用，自动切回 cpu")
        device = "cpu"

    ckpt_path = Path(args.ckpt)
    model, labels, label_map, frames, in_dim = load_ckpt(ckpt_path, device=device)

    split_dir = SEQ_ROOT / args.split
    npz_files = scan_npz(split_dir, labels)

    y_true: List[int] = []
    y_pred: List[int] = []
    skipped_invalid = 0

    print(f"[INFO] split={args.split} align={args.align} decision={args.decision}")
    for p in npz_files:
        data = np.load(p, allow_pickle=True)
        kpts = data["kpts"].astype(np.float32)
        vis = data["vis"].astype(np.float32)

        invalid_for_scoring = 0
        if "invalid_for_scoring" in data.files:
            try:
                invalid_for_scoring = int(np.asarray(data["invalid_for_scoring"]).reshape(-1)[0])
            except Exception:
                invalid_for_scoring = 0

        if (not args.include_invalid) and invalid_for_scoring == 1:
            skipped_invalid += 1
            continue

        true_cls = find_class_from_path(p, labels)
        if true_cls is None or true_cls not in label_map:
            continue
        tid = int(label_map[true_cls])

        pid = predict_one(
            model=model,
            frames=frames,
            classes=labels,
            label_map=label_map,
            kpts=kpts,
            vis=vis,
            align=args.align,
            decision=args.decision,
            true_cls=true_cls if args.align == "gt" else None,
            device=device,
            topk=args.topk,
            gap_tau=args.gap_tau,
        )

        y_true.append(tid)
        y_pred.append(int(pid))

    cm, acc, metrics = confusion_and_metrics(y_true, y_pred, labels)

    print(f"[INFO] evaluated samples={len(y_true)} acc={acc:.3f} (skipped_invalid={skipped_invalid})\n")
    print("Per-class metrics:")
    for name in labels:
        p, r, s = metrics[name]
        print(f"  {name:<10s} precision={p:.3f}  recall={r:.3f}  support={s}")

    print("\nPredicted distribution:")
    pred_arr = np.asarray(y_pred, dtype=np.int64) if y_pred else np.zeros((0,), dtype=np.int64)
    for i, name in enumerate(labels):
        n = int((pred_arr == i).sum())
        ratio = (n / max(1, len(pred_arr)))
        print(f"  {name:<10s} n={n:4d}  ratio={ratio:.3f}")

    # confusion matrix print
    col_w = max(10, max(len(x) for x in labels) + 2)
    header = " " * (col_w + 2) + "".join([f"{x:>{col_w}s}" for x in labels])
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(header)
    for i, name in enumerate(labels):
        row = "".join([f"{cm[i, j]:{col_w}d}" for j in range(len(labels))])
        print(f"{name:<{col_w}s}  {row}")

if __name__ == "__main__":
    main()
