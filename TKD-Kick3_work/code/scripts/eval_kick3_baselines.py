# scripts/eval_kick3_baselines.py
# Evaluate LSTM / GCN baselines and optionally export a red confusion-matrix heatmap (no title).

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

# ---- same joint order as training ----
HEAD = 0
LS, RS = 1, 2
LE, RE = 3, 4
LW, RW = 5, 6
LH, RH = 7, 8
LK, RK = 9, 10
LA, RA = 11, 12

DEFAULT_CLASSES = ["front", "roundhouse", "axe"]

# ----------------------------
# Loading + preprocessing (same as training)
# ----------------------------
def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(str(path), allow_pickle=True)
    keys = set(z.files)
    k_kpts = "kpts" if "kpts" in keys else ("keypoints" if "keypoints" in keys else None)
    k_vis = "vis" if "vis" in keys else ("visibility" if "visibility" in keys else None)
    if k_kpts is None or k_vis is None:
        raise KeyError(f"npz keys not found: got {keys}, need (kpts/keypoints) and (vis/visibility)")
    kpts = z[k_kpts].astype(np.float32)
    vis = z[k_vis].astype(np.float32)
    return kpts, vis

def scan_npz(split_dir: Path) -> List[Path]:
    return sorted([p for p in split_dir.rglob("*.npz") if p.is_file()])

def find_class_from_path(p: Path, classes: List[str]) -> Optional[str]:
    low_parts = [x.lower() for x in p.parts]
    for c in classes:
        if c in low_parts:
            return c
    stem = p.stem.lower()
    for c in classes:
        if c in stem:
            return c
    return None

def forward_fill_missing(kpts: np.ndarray, vis: np.ndarray, tau: float = 0.5) -> np.ndarray:
    T, J, _ = kpts.shape
    out = kpts.copy()
    for j in range(J):
        last = out[0, j].copy()
        for t in range(T):
            if vis[t, j] >= tau and np.isfinite(out[t, j]).all():
                last = out[t, j].copy()
            else:
                out[t, j] = last
    return out

def normalize_kpts(kpts: np.ndarray, vis: np.ndarray, tau: float = 0.5) -> np.ndarray:
    kpts = forward_fill_missing(kpts, vis, tau=tau)
    T, J, _ = kpts.shape
    out = np.zeros_like(kpts, dtype=np.float32)
    for t in range(T):
        v = vis[t]
        p = kpts[t]
        hip_ok = (v[LH] >= tau) and (v[RH] >= tau)
        sh_ok = (v[LS] >= tau) and (v[RS] >= tau)
        if hip_ok:
            center = 0.5 * (p[LH] + p[RH])
        elif sh_ok:
            center = 0.5 * (p[LS] + p[RS])
        else:
            center = p.mean(axis=0)
        sh_w = float(np.linalg.norm(p[LS] - p[RS])) if sh_ok else 0.0
        hip_w = float(np.linalg.norm(p[LH] - p[RH])) if hip_ok else 0.0
        scale = max(sh_w, hip_w, 1.0)
        out[t] = (p - center[None, :]) / scale
    return out

def resample_1d(x: np.ndarray, T_out: int) -> np.ndarray:
    T_in = x.shape[0]
    if T_in == T_out:
        return x.astype(np.float32)
    t_in = np.linspace(0.0, 1.0, num=T_in, dtype=np.float32)
    t_out = np.linspace(0.0, 1.0, num=T_out, dtype=np.float32)
    if x.ndim == 1:
        return np.interp(t_out, t_in, x).astype(np.float32)
    C = x.shape[1]
    y = np.zeros((T_out, C), dtype=np.float32)
    for c in range(C):
        y[:, c] = np.interp(t_out, t_in, x[:, c]).astype(np.float32)
    return y

def resample_kpts(kpts: np.ndarray, vis: np.ndarray, T_out: int) -> Tuple[np.ndarray, np.ndarray]:
    T_in, J, _ = kpts.shape
    if T_in == T_out:
        return kpts.astype(np.float32), vis.astype(np.float32)
    k2 = kpts.reshape(T_in, J * 2)
    k2r = resample_1d(k2, T_out).reshape(T_out, J, 2)
    vr = resample_1d(vis, T_out)
    vr = np.clip(vr, 0.0, 1.0)
    return k2r.astype(np.float32), vr.astype(np.float32)

def infer_attack_side(kpts_n: np.ndarray) -> str:
    dl = np.linalg.norm(kpts_n[:, LA] - kpts_n[:, LH], axis=1).max()
    dr = np.linalg.norm(kpts_n[:, RA] - kpts_n[:, RH], axis=1).max()
    return "L" if dl >= dr else "R"

def find_pivot(kpts_n: np.ndarray, cls: str) -> int:
    side = infer_attack_side(kpts_n)
    if side == "L":
        ankle, hip = LA, LH
    else:
        ankle, hip = RA, RH
    if cls == "axe":
        return int(np.argmin(kpts_n[:, ankle, 1]))
    dist = np.linalg.norm(kpts_n[:, ankle] - kpts_n[:, hip], axis=1)
    return int(np.argmax(dist))

def phase_aligned_resize(kpts_n: np.ndarray, vis: np.ndarray, cls: str, T_out: int,
                        pre_ratio: float = 0.45, post_ratio: float = 0.55) -> Tuple[np.ndarray, np.ndarray]:
    T_in = kpts_n.shape[0]
    if T_in < 2:
        return resample_kpts(kpts_n, vis, T_out)
    pivot = find_pivot(kpts_n, cls)
    w_pre = int(np.floor(pre_ratio * T_in))
    w_post = int(np.floor(post_ratio * T_in))
    s = max(0, pivot - w_pre)
    e = min(T_in - 1, pivot + w_post)
    if e <= s:
        s = max(0, pivot - 1)
        e = min(T_in - 1, pivot + 1)
    seg_k = kpts_n[s:e+1]
    seg_v = vis[s:e+1]
    return resample_kpts(seg_k, seg_v, T_out)

def build_feat_52(kpts_a: np.ndarray) -> np.ndarray:
    T, J, _ = kpts_a.shape
    pos = kpts_a.reshape(T, J * 2).astype(np.float32)
    vel = np.zeros_like(pos, dtype=np.float32)
    vel[1:] = pos[1:] - pos[:-1]
    return np.concatenate([pos, vel], axis=1).astype(np.float32)

# ----------------------------
# Models (must match ckpt)
# ----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, in_dim: int = 52, hidden: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, num_classes: int = 3, bidir: bool = True) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidir,
        )
        out_dim = hidden * (2 if bidir else 1)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        return self.fc(pooled)

def build_adjacency_13() -> torch.Tensor:
    edges = [
        (HEAD, LS), (HEAD, RS),
        (LS, LE), (LE, LW),
        (RS, RE), (RE, RW),
        (LS, RS),
        (LS, LH), (RS, RH),
        (LH, RH),
        (LH, LK), (LK, LA),
        (RH, RK), (RK, RA),
    ]
    J = 13
    A = torch.zeros(J, J, dtype=torch.float32)
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    A += torch.eye(J, dtype=torch.float32)
    deg = A.sum(dim=1).clamp(min=1.0)
    D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
    return D_inv_sqrt @ A @ D_inv_sqrt

class SpatialGCN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, A_norm: torch.Tensor, dropout: float = 0.2) -> None:
        super().__init__()
        self.register_buffer("A", A_norm)
        self.lin = nn.Linear(in_ch, out_ch, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.res = None
        if in_ch != out_ch:
            self.res = nn.Linear(in_ch, out_ch, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, J, Cin = x.shape
        xt = x.reshape(B * T, J, Cin)
        xt = torch.matmul(self.A, xt)
        xt = self.lin(xt)
        y = xt.reshape(B * T * J, -1)
        y = self.bn(y)
        y = F.relu(y)
        y = self.drop(y)
        y = y.reshape(B, T, J, -1)
        r = self.res(x) if self.res is not None else x
        return y + r

class GCNClassifier(nn.Module):
    def __init__(self, in_dim: int = 52, num_classes: int = 3, d_model: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        assert in_dim == 52
        A = build_adjacency_13()
        self.proj = nn.Linear(4, d_model)
        self.g1 = SpatialGCN(d_model, d_model, A, dropout=dropout)
        self.g2 = SpatialGCN(d_model, d_model, A, dropout=dropout)
        self.tconv = nn.Conv1d(d_model * 13, d_model * 13, kernel_size=3, padding=1, groups=13)
        self.bn = nn.BatchNorm1d(d_model * 13)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        xj = x.view(B, T, 13, 4)
        h = self.proj(xj)
        h = self.g1(h)
        h = self.g2(h)
        h = h.permute(0, 2, 3, 1).contiguous()  # (B,J,C,T)
        h = h.view(B, 13 * h.shape[2], T)
        h = self.tconv(h)
        h = self.bn(h)
        h = F.relu(h)
        h = self.drop(h)
        h = h.view(B, 13, -1, T).mean(dim=1).mean(dim=-1)
        return self.fc(h)

def build_model(model_type: str, hparams: Dict, num_classes: int) -> nn.Module:
    if model_type == "lstm":
        return LSTMClassifier(
            in_dim=int(hparams.get("in_dim", 52)),
            hidden=int(hparams.get("lstm_hidden", 128)),
            num_layers=int(hparams.get("lstm_layers", 2)),
            dropout=float(hparams.get("dropout", 0.2)),
            num_classes=num_classes,
            bidir=bool(hparams.get("lstm_bidir", True)),
        )
    if model_type == "gcn":
        return GCNClassifier(
            in_dim=int(hparams.get("in_dim", 52)),
            num_classes=num_classes,
            d_model=int(hparams.get("gcn_d_model", 64)),
            dropout=float(hparams.get("dropout", 0.2)),
        )
    raise ValueError(f"Unknown model_type={model_type}")

# ----------------------------
# Metrics
# ----------------------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> np.ndarray:
    cm = np.zeros((C, C), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def prf_from_cm(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    C = cm.shape[0]
    tp = np.diag(cm).astype(np.float32)
    fp = cm.sum(axis=0).astype(np.float32) - tp
    fn = cm.sum(axis=1).astype(np.float32) - tp
    sup = cm.sum(axis=1).astype(np.float32)

    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1, sup

def macro_f1(f1: np.ndarray) -> float:
    return float(np.mean(f1))

def weighted_f1(f1: np.ndarray, sup: np.ndarray) -> float:
    w = sup / max(1.0, float(sup.sum()))
    return float(np.sum(w * f1))

def balanced_acc(rec: np.ndarray) -> float:
    return float(np.mean(rec))

def plot_cm_red(cm: np.ndarray, classes: List[str], save_path: Path, normalize: bool = True) -> None:
    cm_plot = cm.astype(np.float32)
    if normalize:
        row_sum = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm_plot, row_sum, out=np.zeros_like(cm_plot), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
    im = ax.imshow(cm_plot, cmap="Reds", vmin=0.0, vmax=1.0 if normalize else None)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    # no title (as requested)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # annotate
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val = cm_plot[i, j]
            txt = f"{val:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# Inference
# ----------------------------
@torch.no_grad()
def forward_one(model: nn.Module, feat_T52: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(feat_T52).unsqueeze(0).to(device)  # (1,T,52)
    logits = model(x)
    prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return prob

@torch.no_grad()
def predict_one(model: nn.Module, kpts: np.ndarray, vis: np.ndarray,
                classes: List[str], T_out: int, align: str, decision: str,
                device: str) -> int:
    kpts_n = normalize_kpts(kpts, vis, tau=0.5)

    if decision == "argmax":
        if align == "uniform":
            kpts_a, _ = resample_kpts(kpts_n, vis, T_out)
        elif align == "phase":
            # NOTE: argmax+phase is not deployable unless you already know class;
            # we still provide it for analysis (equivalent to a non-sweep phase inference).
            # Use with caution in papers.
            # Here we simply fall back to uniform if requested.
            kpts_a, _ = resample_kpts(kpts_n, vis, T_out)
        else:
            raise ValueError(f"align={align} not supported for argmax")
        feat = build_feat_52(kpts_a)
        prob = forward_one(model, feat, device=device)
        return int(np.argmax(prob))

    if decision == "sweep":
        # deployable: for each candidate class, phase-align by that class rule and pick self-consistent prob(c)
        scores = []
        for ci, cls in enumerate(classes):
            kpts_a, _ = phase_aligned_resize(kpts_n, vis, cls, T_out)
            feat = build_feat_52(kpts_a)
            prob = forward_one(model, feat, device=device)
            scores.append(float(prob[ci]))
        return int(np.argmax(scores))

    raise ValueError(f"Unknown decision={decision}")

# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--align", type=str, default="uniform", choices=["uniform", "phase"])
    ap.add_argument("--decision", type=str, default="argmax", choices=["argmax", "sweep"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seq_root", type=str, default=str(Path("scripts") / "data" / "sequences"))
    ap.add_argument("--save_cm", type=str, default="")
    ap.add_argument("--normalize_cm", action="store_true")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    model_type = ckpt.get("model_type", "")
    classes = ckpt.get("classes", DEFAULT_CLASSES)
    hparams = ckpt.get("hparams", {})
    state = ckpt.get("model", ckpt)  # allow loading pure state_dict too
    T_out = int(hparams.get("T", 96))

    model = build_model(model_type, hparams, num_classes=len(classes))
    model.load_state_dict(state, strict=True)
    model.to(args.device)
    model.eval()

    split_dir = Path(args.seq_root) / args.split
    paths = scan_npz(split_dir)

    y_true = []
    y_pred = []
    skipped_invalid = 0
    skipped_no_label = 0

    for p in paths:
        cls = find_class_from_path(p, classes)
        if cls is None:
            skipped_no_label += 1
            continue
        try:
            kpts, vis = load_npz(p)
            pid = predict_one(
                model=model, kpts=kpts, vis=vis,
                classes=classes, T_out=T_out,
                align=args.align, decision=args.decision,
                device=args.device,
            )
            y_true.append(int(classes.index(cls)))
            y_pred.append(int(pid))
        except Exception:
            skipped_invalid += 1
            continue

    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    cm = confusion_matrix(y_true, y_pred, C=len(classes))
    prec, rec, f1, sup = prf_from_cm(cm)

    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    mf1 = macro_f1(f1)
    wf1 = weighted_f1(f1, sup)
    bacc = balanced_acc(rec)

    print(f"[INFO] ckpt={ckpt_path.name} device={args.device} model_type={model_type}")
    print(f"[INFO] split={args.split} align={args.align} decision={args.decision}")
    print(f"[INFO] evaluated samples={len(y_true)} acc={acc:.3f} (skipped_invalid={skipped_invalid}, skipped_no_label={skipped_no_label})\n")

    print("Per-class metrics:")
    for i, c in enumerate(classes):
        print(f"  {c:<10s} precision={prec[i]:.3f}  recall={rec[i]:.3f}  f1={f1[i]:.3f}  support={int(sup[i])}")
    print()
    print(f"Macro-F1={mf1:.3f}  Weighted-F1={wf1:.3f}  BalancedAcc={bacc:.3f}\n")

    print("Confusion Matrix (rows=true, cols=pred):")
    header = " " * 12 + "".join([f"{c:>12s}" for c in classes])
    print(header)
    for i, c in enumerate(classes):
        row = "".join([f"{cm[i, j]:>12d}" for j in range(len(classes))])
        print(f"{c:<12s}{row}")

    if args.save_cm.strip():
        save_path = Path(args.save_cm)
        plot_cm_red(cm, classes, save_path, normalize=bool(args.normalize_cm))
        print(f"\n[SAVED] confusion matrix -> {save_path}")

if __name__ == "__main__":
    main()
