# -*- coding: utf-8 -*-
"""
eval_val_confusion.py

在 data/sequences/{split}/... 下对 *.npz 做分类评估，输出：
- acc
- per-class precision/recall/support
- predicted distribution
- confusion matrix (rows=true, cols=pred)

并支持消融验证“相位对齐策略”：
--align:
  - uniform : 线性重采样到固定帧数
  - gt      : 用真值类别做相位对齐（上界，用于分析）
  - phase   : 相位对齐（部署可用）

--decision:
  - argmax  :
      * align=uniform : 直接 argmax
      * align=gt      : 相位对齐后 argmax
      * align=phase   : 两阶段：uniform argmax 得到粗预测 -> 用粗预测类别做相位对齐 -> 再 argmax
  - sweep   :（推荐用于验证 phase）
      * align=phase 时：对每个候选类别都做一次相位对齐，取该类别概率作为 score，选最高者

ckpt 兼容：
- 支持 ckpt dict 中存 state_dict 的 key: model / model_state / state_dict / net / network
- 支持直接就是 state_dict
- 支持参数命名差异：input_proj -> proj；classifier -> head；丢弃 pos_enc.* buffer

用法示例（PowerShell）：
  # baseline
  python .\scripts\eval_val_confusion.py --ckpt .\models\checkpoints\kick3_uni.pt --split val --align uniform --decision argmax --device cpu

  # phase（推理端相位对齐收益）
  python .\scripts\eval_val_confusion.py --ckpt .\models\checkpoints\kick3_uni.pt --split val --align phase --decision sweep --device cpu

  # 上界参考
  python .\scripts\eval_val_confusion.py --ckpt .\models\checkpoints\kick3_uni.pt --split val --align gt --decision argmax --device cpu
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


# --------------------
# Paths / Defaults
# --------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SEQ_ROOT = BASE_DIR / "data" / "sequences"
DEFAULT_LABEL_JSON = BASE_DIR / "models" / "checkpoints" / "kick3_label_map.json"

# 13 joints indexing（与你工程一致）
VTH = 0.2
HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(13)


# --------------------
# Utils: temporal resize
# --------------------
def temporal_resize(arr: np.ndarray, L: int) -> np.ndarray:
    """Linear interpolation along time axis to length L. arr: (T, D)"""
    T = int(arr.shape[0])
    if T == L:
        return arr
    if T <= 1:
        return np.repeat(arr, L, axis=0)
    idxs = np.linspace(0, T - 1, num=L)
    i0 = np.floor(idxs).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (idxs - i0)[..., None]
    return arr[i0] * (1 - w) + arr[i1] * w


# --------------------
# Feature engineering（与训练/推理一致的 52 维：pos(26)+vel(26)）
# --------------------
def build_frame_features(kpts: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """
    kpts: (T,13,2 or 3)
    vis:  (T,13)
    return: (T, 52) for 2D
    """
    if kpts.ndim != 3:
        raise ValueError(f"kpts expected (T,13,D), got {kpts.shape}")
    T, J, D0 = kpts.shape
    if J != 13:
        raise ValueError(f"Expected 13 joints, got {J}")

    D = min(2, D0)  # 默认用 xy
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


# --------------------
# Phase-aligned resize
# --------------------
def _kick_side_indices(kpts: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    """依据左右踝轨迹长度，判断攻击侧"""
    def path_len(p: np.ndarray) -> float:
        if p.shape[0] <= 1:
            return 0.0
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        return float(np.nansum(d))

    if path_len(kpts[:, RA]) >= path_len(kpts[:, LA]):
        return (RA, RK, RH, LA, LK, LH)
    return (LA, LK, LH, RA, RK, RH)


def _pivot_index(kpts: np.ndarray, vis: np.ndarray, ktype: str) -> int:
    """
    pivot frame：
    - axe：踝最高（y最小）
    - 其他：||ankle-hip|| 最大
    """
    A, _, H, *_ = _kick_side_indices(kpts)

    if ktype == "axe":
        ys = np.where(vis[:, A] > VTH, kpts[:, A, 1], np.inf)
        return int(np.argmin(ys))

    dist = np.full((kpts.shape[0],), -1e9, np.float32)
    mask = (vis[:, A] > VTH) & (vis[:, H] > VTH)
    dist[mask] = np.linalg.norm(kpts[mask, A] - kpts[mask, H], axis=1)
    return int(np.argmax(dist))


def phase_aligned_resize(
    feat: np.ndarray,
    kpts: np.ndarray,
    vis: np.ndarray,
    L: int,
    ktype: str,
    pre: float = 0.45,
    post: float = 0.55
) -> np.ndarray:
    T = int(feat.shape[0])
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


# --------------------
# Model（用于 eval 的参考结构）
# 注意：训练脚本若命名不同，我们会在 state_dict remap 里兼容
# --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # 位置编码是 buffer，不强依赖从 ckpt 恢复（丢弃也不影响推理一致性）
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class Kick3Transformer(nn.Module):
    def __init__(
        self,
        in_dim: int = 52,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
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


# --------------------
# CKPT helpers
# --------------------
def _looks_like_state_dict(d: dict) -> bool:
    if not isinstance(d, dict) or not d:
        return False
    has_tensor = any(isinstance(v, torch.Tensor) for v in d.values())
    has_dotkey = any(isinstance(k, str) and ("." in k) for k in d.keys())
    return bool(has_tensor and has_dotkey)


def extract_state_dict(ckpt_obj) -> Tuple[Dict, str]:
    """
    返回 (state_dict, from_key)
    兼容：
      - ckpt['model'] / ckpt['model_state'] / ckpt['state_dict'] / ckpt['net'] / ckpt['network']
      - 直接就是 state_dict
    """
    if isinstance(ckpt_obj, dict):
        for k in ("model", "model_state", "state_dict", "net", "network"):
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict) and _looks_like_state_dict(ckpt_obj[k]):
                return ckpt_obj[k], k
        if _looks_like_state_dict(ckpt_obj):
            return ckpt_obj, "<root>"

        raise ValueError(
            "无法从 ckpt 中提取 state_dict。顶层 keys: "
            + ", ".join(list(ckpt_obj.keys())[:60])
        )
    raise ValueError("ckpt 不是 dict，无法解析")


def remap_state_dict_for_compat(sd: Dict) -> Dict:
    """
    兼容不同命名习惯：
      - input_proj.* -> proj.*
      - classifier.* -> head.*
      - pos_enc.*    -> 丢弃（位置编码 buffer，不影响推理）
    """
    out = {}
    for k, v in sd.items():
        if not isinstance(k, str):
            continue

        # 位置编码 buffer：不同实现可能保存/不保存，直接丢弃可避免 unexpected
        if k.startswith("pos_enc.") or k.startswith("positional_encoding.") or k.startswith("pe.") or k.endswith(".pe"):
            # 更保守：只丢弃明显是 PE 的 key
            if "pos" in k and "pe" in k:
                continue
            if k.startswith("pos_enc."):
                continue

        if k.startswith("input_proj."):
            k = k.replace("input_proj.", "proj.", 1)

        if k.startswith("classifier."):
            k = k.replace("classifier.", "head.", 1)

        out[k] = v
    return out


def load_label_map(ckpt_obj: dict, label_json: Path) -> Tuple[List[str], Dict[str, int]]:
    """
    label_map: {class_name: id}
    优先 ckpt 内 label_map，其次 label_json，否则 fallback
    """
    label_map = None
    if isinstance(ckpt_obj, dict) and isinstance(ckpt_obj.get("label_map", None), dict):
        label_map = ckpt_obj["label_map"]

    if label_map is None and label_json.exists():
        try:
            label_map = json.loads(label_json.read_text(encoding="utf-8"))
        except Exception:
            label_map = None

    if label_map is None:
        # 你当前去掉 side，默认 3 类；若你 ckpt 是 4 类也不影响（会被 ckpt/label_json 覆盖）
        label_map = {"front": 0, "roundhouse": 1, "axe": 2}

    label_map = {str(k): int(v) for k, v in label_map.items()}
    labels = sorted(label_map.keys(), key=lambda x: label_map[x])
    return labels, label_map


def infer_hparams_from_ckpt_and_sd(ckpt_obj: dict, sd: Dict) -> Dict[str, float]:
    """
    尽量从 ckpt/hparams 推断：frames/in_dim/d_model/nhead/num_layers/dropout
    推断优先级：
      1) ckpt['hparams'] / ckpt 顶层字段
      2) 从权重形状推断 in_dim/d_model/num_layers
      3) fallback 默认值
    """
    hp = {}
    if isinstance(ckpt_obj, dict):
        if isinstance(ckpt_obj.get("hparams", None), dict):
            for k, v in ckpt_obj["hparams"].items():
                hp[str(k)] = v
        for k in ("frames", "in_dim", "d_model", "nhead", "num_layers", "dropout"):
            if k in ckpt_obj:
                hp[k] = ckpt_obj[k]

    # in_dim / d_model：从 proj/input_proj 推断
    in_dim = hp.get("in_dim", None)
    d_model = hp.get("d_model", None)

    w = None
    if "proj.weight" in sd and isinstance(sd["proj.weight"], torch.Tensor):
        w = sd["proj.weight"]
    elif "input_proj.weight" in sd and isinstance(sd["input_proj.weight"], torch.Tensor):
        w = sd["input_proj.weight"]

    if w is not None and w.ndim == 2:
        if d_model is None:
            d_model = int(w.shape[0])
        if in_dim is None:
            in_dim = int(w.shape[1])

    # num_layers：从 encoder.layers.* 统计
    num_layers = hp.get("num_layers", None)
    if num_layers is None:
        layer_ids = set()
        for k in sd.keys():
            if isinstance(k, str) and k.startswith("encoder.layers."):
                parts = k.split(".")
                if len(parts) >= 3 and parts[2].isdigit():
                    layer_ids.add(int(parts[2]))
        num_layers = (max(layer_ids) + 1) if layer_ids else 2

    # nhead：无法可靠从权重推断，优先 hparams，否则默认 4
    nhead = int(hp.get("nhead", 4))
    dropout = float(hp.get("dropout", 0.2))
    frames = int(hp.get("frames", 96))

    if d_model is None:
        d_model = 128
    if in_dim is None:
        in_dim = 52

    return {
        "frames": int(frames),
        "in_dim": int(in_dim),
        "d_model": int(d_model),
        "nhead": int(nhead),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
    }


@torch.no_grad()
def run_model(model: nn.Module, x_np: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(x_np).unsqueeze(0).to(device).float()  # (1,L,D)
    logits = model(x)
    prob = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
    return prob


def get_true_label(npz_path: Path, split_root: Path, labels: List[str]) -> Optional[str]:
    """
    从目录结构推断真值类别：split_root/{class}/xxx.npz
    若中间还有子目录，也会尝试在路径 parts 里找 labels 中的词
    """
    try:
        rel = npz_path.relative_to(split_root)
    except Exception:
        return None

    # 常见结构：{class}/xxx.npz
    if len(rel.parts) >= 2:
        cand = rel.parts[0]
        if cand in labels:
            return cand

    # fallback：在路径所有 parts 里找
    for p in rel.parts:
        if p in labels:
            return p
    return None


def pretty_confusion(cm: np.ndarray, labels: List[str]) -> str:
    col_w = max(10, max(len(x) for x in labels) + 2)
    head = " " * col_w + "".join(f"{lb:>{col_w}}" for lb in labels)
    rows = [head]
    for i, lb in enumerate(labels):
        row = f"{lb:<{col_w}}" + "".join(f"{int(cm[i, j]):>{col_w}}" for j in range(len(labels)))
        rows.append(row)
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str, help="模型权重路径")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--seq_root", type=str, default=str(DEFAULT_SEQ_ROOT), help="sequences 根目录")
    ap.add_argument("--label_json", type=str, default=str(DEFAULT_LABEL_JSON), help="label_map json（ckpt 内无 label_map 时使用）")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--include_invalid", action="store_true", help="包含 invalid_for_scoring=1（默认跳过）")

    ap.add_argument("--align", type=str, default="uniform", choices=["gt", "uniform", "phase"],
                    help="对齐策略：gt(上界) / uniform(线性) / phase(相位对齐)")
    ap.add_argument("--decision", type=str, default="argmax", choices=["argmax", "sweep"],
                    help="决策策略：argmax / sweep（align=phase时推荐 sweep）")

    ap.add_argument("--pre", type=float, default=0.45, help="phase 对齐窗口 pivot 前比例")
    ap.add_argument("--post", type=float, default=0.55, help="phase 对齐窗口 pivot 后比例")

    ap.add_argument("--max_samples", type=int, default=-1, help="仅评估前 N 个样本（调试用），默认-1表示全部")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 ckpt：{ckpt_path}")

    ckpt_obj = torch.load(str(ckpt_path), map_location=device)
    if not isinstance(ckpt_obj, dict):
        raise ValueError("ckpt 格式不支持（不是 dict）")

    state_dict_raw, sd_from = extract_state_dict(ckpt_obj)
    state_dict = remap_state_dict_for_compat(state_dict_raw)

    labels, label_map = load_label_map(ckpt_obj, Path(args.label_json))
    id2label = {int(v): k for k, v in label_map.items()}

    hp = infer_hparams_from_ckpt_and_sd(ckpt_obj, state_dict_raw)
    frames = int(hp["frames"])
    in_dim = int(hp["in_dim"])
    d_model = int(hp["d_model"])
    nhead = int(hp["nhead"])
    num_layers = int(hp["num_layers"])
    dropout = float(hp["dropout"])

    model = Kick3Transformer(
        in_dim=in_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=len(labels),
        dropout=dropout,
    ).to(device)

    incompat = model.load_state_dict(state_dict, strict=False)
    if len(incompat.missing_keys) > 0 or len(incompat.unexpected_keys) > 0:
        print("[WARN] load_state_dict incompat（通常仅是位置编码buffer或命名差异，不影响推理）:")
        if incompat.missing_keys:
            print("  missing_keys:", incompat.missing_keys[:30], "..." if len(incompat.missing_keys) > 30 else "")
        if incompat.unexpected_keys:
            print("  unexpected_keys:", incompat.unexpected_keys[:30], "..." if len(incompat.unexpected_keys) > 30 else "")

    model.eval()

    print(f"[INFO] ckpt={ckpt_path.name} device={device} state_dict_from={sd_from}")
    print(f"[INFO] hparams: in_dim={in_dim} d_model={d_model} nhead={nhead} num_layers={num_layers} dropout={dropout} frames={frames}")
    print(f"[INFO] split={args.split} align={args.align} decision={args.decision}")

    split_root = Path(args.seq_root) / args.split
    if not split_root.exists():
        raise FileNotFoundError(f"找不到序列目录：{split_root}")

    npz_files = sorted(split_root.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"在 {split_root} 下没有找到 npz")

    if args.max_samples and args.max_samples > 0:
        npz_files = npz_files[: args.max_samples]

    K = len(labels)
    cm = np.zeros((K, K), dtype=np.int64)  # rows=true cols=pred
    support = np.zeros((K,), dtype=np.int64)
    pred_count = np.zeros((K,), dtype=np.int64)

    total = 0
    correct = 0
    skipped_no_label = 0
    skipped_invalid = 0

    for p in npz_files:
        data = np.load(p, allow_pickle=True)

        if (not args.include_invalid) and ("invalid_for_scoring" in data.files):
            try:
                inv = int(np.asarray(data["invalid_for_scoring"]).reshape(-1)[0])
                if inv == 1:
                    skipped_invalid += 1
                    continue
            except Exception:
                pass

        true_cls = get_true_label(p, split_root, labels)
        if true_cls is None:
            skipped_no_label += 1
            continue

        kpts = data["kpts"].astype(np.float32)
        vis = data["vis"].astype(np.float32) if "vis" in data.files else np.ones(kpts.shape[:2], np.float32)

        feat = build_frame_features(kpts, vis)  # (T, 52)
        if feat.shape[1] != in_dim:
            raise ValueError(f"特征维度不匹配：feat={feat.shape[1]} ckpt_in_dim={in_dim} file={p}")

        # -------- inference --------
        if args.align == "uniform":
            feat_in = temporal_resize(feat, frames)
            prob = run_model(model, feat_in, device)
            pred_id = int(np.argmax(prob))

        elif args.align == "gt":
            # 用真值类别做 phase 对齐（分析上界）
            feat_in = phase_aligned_resize(feat, kpts, vis, frames, true_cls, pre=args.pre, post=args.post)
            prob = run_model(model, feat_in, device)
            pred_id = int(np.argmax(prob))

        else:  # phase
            if args.decision == "sweep":
                # 对每个候选类别做相位对齐，然后取“该类别概率”作为 score
                best_id = None
                best_score = -1.0
                for cls_name, cls_id in label_map.items():
                    cls_id = int(cls_id)
                    if cls_id < 0 or cls_id >= K:
                        continue
                    feat_pa = phase_aligned_resize(feat, kpts, vis, frames, cls_name, pre=args.pre, post=args.post)
                    prob = run_model(model, feat_pa, device)
                    score = float(prob[cls_id])
                    if score > best_score:
                        best_score = score
                        best_id = cls_id
                pred_id = int(best_id if best_id is not None else 0)
            else:
                # 两阶段：先 uniform 粗分类，再按粗分类类别做相位对齐后二次分类
                feat_u = temporal_resize(feat, frames)
                prob0 = run_model(model, feat_u, device)
                pred0 = int(np.argmax(prob0))
                pred0_name = id2label.get(pred0, labels[pred0] if 0 <= pred0 < K else labels[0])
                feat_pa = phase_aligned_resize(feat, kpts, vis, frames, pred0_name, pre=args.pre, post=args.post)
                prob = run_model(model, feat_pa, device)
                pred_id = int(np.argmax(prob))

        true_id = int(label_map[true_cls])

        cm[true_id, pred_id] += 1
        support[true_id] += 1
        pred_count[pred_id] += 1

        total += 1
        if pred_id == true_id:
            correct += 1

    acc = 0.0 if total == 0 else correct / total
    print(f"[INFO] evaluated samples={total} acc={acc:.3f} (skipped_invalid={skipped_invalid}, skipped_no_label={skipped_no_label})")

    print("\nPer-class metrics:")
    for i, lb in enumerate(labels):
        tp = int(cm[i, i])
        sup = int(support[i])
        pc = int(pred_count[i])
        prec = 0.0 if pc == 0 else tp / pc
        rec = 0.0 if sup == 0 else tp / sup
        print(f"  {lb:<10s} precision={prec:.3f}  recall={rec:.3f}  support={sup}")

    print("\nPredicted distribution:")
    for i, lb in enumerate(labels):
        n = int(pred_count[i])
        ratio = 0.0 if total == 0 else n / total
        print(f"  {lb:<10s} n={n:4d}  ratio={ratio:.3f}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(pretty_confusion(cm, labels))


if __name__ == "__main__":
    main()
