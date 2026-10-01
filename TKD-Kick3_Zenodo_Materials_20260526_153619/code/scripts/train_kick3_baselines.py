# scripts/train_kick3_baselines.py
# Train LSTM / GCN baselines under the same preprocessing + alignment protocol.

from __future__ import annotations
import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ----------------------------
# Constants (13 joints order)
# ----------------------------
# Must be consistent with your project (HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA)
HEAD = 0
LS, RS = 1, 2
LE, RE = 3, 4
LW, RW = 5, 6
LH, RH = 7, 8
LK, RK = 9, 10
LA, RA = 11, 12

CLASSES = ["front", "roundhouse", "axe"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: deterministic can reduce speed; enable only if needed
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ----------------------------
# Utilities: scan npz and infer label
# ----------------------------
def scan_npz(split_dir: Path) -> List[Path]:
    return sorted([p for p in split_dir.rglob("*.npz") if p.is_file()])


def find_class_from_path(p: Path) -> Optional[str]:
    # Try parent folder first: .../val/front/xxx.npz
    low_parts = [x.lower() for x in p.parts]
    for c in CLASSES:
        if c in low_parts:
            return c

    # Fallback: filename contains class token
    stem = p.stem.lower()
    for c in CLASSES:
        if c in stem:
            return c
    return None


# ----------------------------
# Load npz (robust)
# ----------------------------
def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(str(path), allow_pickle=True)
    keys = set(z.files)

    # common variants
    k_kpts = "kpts" if "kpts" in keys else ("keypoints" if "keypoints" in keys else None)
    k_vis = "vis" if "vis" in keys else ("visibility" if "visibility" in keys else None)

    if k_kpts is None or k_vis is None:
        raise KeyError(f"npz keys not found: got {keys}, need (kpts/keypoints) and (vis/visibility)")

    kpts = z[k_kpts].astype(np.float32)  # (T, 13, 2)
    vis = z[k_vis].astype(np.float32)    # (T, 13)
    return kpts, vis


# ----------------------------
# Preprocess: fill missing -> normalize -> align -> build features(52)
# ----------------------------
def forward_fill_missing(kpts: np.ndarray, vis: np.ndarray, tau: float = 0.5) -> np.ndarray:
    """If a joint is invisible, forward-fill its (x,y) using last visible value."""
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
    """
    Per-frame center: mid-hip if both hips visible; else mid-shoulder; else mean of joints.
    Per-frame scale: max(shoulder_width, hip_width, 1.0)
    """
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
    """Linear resample along time for 1D or 2D arrays: (T,) or (T,C)."""
    T_in = x.shape[0]
    if T_in == T_out:
        return x.astype(np.float32)

    t_in = np.linspace(0.0, 1.0, num=T_in, dtype=np.float32)
    t_out = np.linspace(0.0, 1.0, num=T_out, dtype=np.float32)

    if x.ndim == 1:
        y = np.interp(t_out, t_in, x).astype(np.float32)
        return y

    # (T,C)
    C = x.shape[1]
    y = np.zeros((T_out, C), dtype=np.float32)
    for c in range(C):
        y[:, c] = np.interp(t_out, t_in, x[:, c]).astype(np.float32)
    return y


def resample_kpts(kpts: np.ndarray, vis: np.ndarray, T_out: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resample (T,J,2) + (T,J) to fixed length."""
    T_in, J, _ = kpts.shape
    if T_in == T_out:
        return kpts.astype(np.float32), vis.astype(np.float32)

    k2 = kpts.reshape(T_in, J * 2)
    k2r = resample_1d(k2, T_out).reshape(T_out, J, 2)
    vr = resample_1d(vis, T_out)
    vr = np.clip(vr, 0.0, 1.0)
    return k2r.astype(np.float32), vr.astype(np.float32)


def infer_attack_side(kpts_n: np.ndarray) -> str:
    """
    Heuristic: choose the side (L/R) with larger max ankle-hip distance across time.
    """
    # left
    dl = np.linalg.norm(kpts_n[:, LA] - kpts_n[:, LH], axis=1).max()
    dr = np.linalg.norm(kpts_n[:, RA] - kpts_n[:, RH], axis=1).max()
    return "L" if dl >= dr else "R"


def find_pivot(kpts_n: np.ndarray, cls: str) -> int:
    """
    front/roundhouse: pivot at max ankle-hip distance (attack side)
    axe: pivot at highest ankle (min y) (attack side)
    """
    side = infer_attack_side(kpts_n)
    if side == "L":
        ankle, hip = LA, LH
    else:
        ankle, hip = RA, RH

    if cls == "axe":
        # y smaller => higher (image coordinates)
        return int(np.argmin(kpts_n[:, ankle, 1]))
    else:
        dist = np.linalg.norm(kpts_n[:, ankle] - kpts_n[:, hip], axis=1)
        return int(np.argmax(dist))


def phase_aligned_resize(kpts_n: np.ndarray, vis: np.ndarray, cls: str, T_out: int,
                        pre_ratio: float = 0.45, post_ratio: float = 0.55) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop an asymmetric window around pivot, then resample to T_out.
    """
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
    """
    Build (T,52) = [pos(13*2), vel(13*2)] where vel[0]=0.
    """
    T, J, _ = kpts_a.shape
    pos = kpts_a.reshape(T, J * 2).astype(np.float32)
    vel = np.zeros_like(pos, dtype=np.float32)
    vel[1:] = pos[1:] - pos[:-1]
    feat = np.concatenate([pos, vel], axis=1)  # (T, 52)
    return feat.astype(np.float32)


# ----------------------------
# Dataset
# ----------------------------
class KickNPZDataset(Dataset):
    def __init__(self, seq_root: Path, split: str, align: str, T_out: int = 96) -> None:
        self.seq_dir = seq_root / split
        self.paths = scan_npz(self.seq_dir)
        self.align = align
        self.T_out = T_out

        self.items: List[Tuple[Path, int]] = []
        for p in self.paths:
            cls = find_class_from_path(p)
            if cls is None:
                continue
            self.items.append((p, CLASS_TO_ID[cls]))

        if len(self.items) == 0:
            raise RuntimeError(f"No valid npz found under {self.seq_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p, y = self.items[idx]
        kpts, vis = load_npz(p)

        kpts_n = normalize_kpts(kpts, vis, tau=0.5)

        if self.align == "uniform":
            kpts_a, _ = resample_kpts(kpts_n, vis, self.T_out)
        elif self.align == "phase":
            cls = CLASSES[y]
            kpts_a, _ = phase_aligned_resize(kpts_n, vis, cls, self.T_out)
        else:
            raise ValueError(f"Unknown align={self.align}")

        feat = build_feat_52(kpts_a)  # (T,52)
        x = torch.from_numpy(feat)          # float32
        yt = torch.tensor(y, dtype=torch.long)
        return x, yt


def collate_batch(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)  # (B,T,52)
    y = torch.stack(ys, dim=0)  # (B,)
    return x, y


# ----------------------------
# Models
# ----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, in_dim: int = 52, hidden: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, num_classes: int = 3, bidir: bool = True) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.bidir = bidir

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
        # x: (B,T,52)
        out, _ = self.lstm(x)  # (B,T,H*)
        pooled = out.mean(dim=1)  # GAP over time
        logits = self.fc(pooled)
        return logits


def build_adjacency_13() -> torch.Tensor:
    """
    Simple skeleton graph (13 joints):
    head-shoulders, shoulders-elbows-wrists, shoulders-hips, hips-knees-ankles,
    plus LS-RS and LH-RH connections.
    """
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
    # self loops
    A += torch.eye(J, dtype=torch.float32)
    # normalize D^{-1/2} A D^{-1/2}
    deg = A.sum(dim=1).clamp(min=1.0)
    D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm


class SpatialGCN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, A_norm: torch.Tensor, dropout: float = 0.2) -> None:
        super().__init__()
        self.register_buffer("A", A_norm)  # (J,J)
        self.lin = nn.Linear(in_ch, out_ch, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)

        self.res = None
        if in_ch != out_ch:
            self.res = nn.Linear(in_ch, out_ch, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,J,Cin)
        """
        B, T, J, Cin = x.shape
        # graph propagation: A @ X (over J)
        # (B,T,J,C) -> (B*T,J,C)
        xt = x.reshape(B * T, J, Cin)
        xt = torch.matmul(self.A, xt)  # (B*T,J,Cin)
        xt = self.lin(xt)              # (B*T,J,Cout)

        # BN over channel: reshape to (B*T*J,C)
        y = xt.reshape(B * T * J, -1)
        y = self.bn(y)
        y = F.relu(y)
        y = self.drop(y)
        y = y.reshape(B, T, J, -1)

        if self.res is not None:
            r = self.res(x)
        else:
            r = x
        return y + r


class GCNClassifier(nn.Module):
    """
    Lightweight ST-GCN-like baseline:
    input (B,T,52) -> reshape (B,T,13,4) -> spatial GCN blocks -> temporal conv -> pooling -> cls
    """
    def __init__(self, in_dim: int = 52, num_classes: int = 3, d_model: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        assert in_dim == 52, "This implementation assumes (13 joints) * (x,y,vx,vy)=52."
        A = build_adjacency_13()

        self.proj = nn.Linear(4, d_model)
        self.g1 = SpatialGCN(d_model, d_model, A, dropout=dropout)
        self.g2 = SpatialGCN(d_model, d_model, A, dropout=dropout)

        # temporal conv on flattened joints
        self.tconv = nn.Conv1d(d_model * 13, d_model * 13, kernel_size=3, padding=1, groups=13)
        self.bn = nn.BatchNorm1d(d_model * 13)
        self.drop = nn.Dropout(dropout)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,52)
        B, T, D = x.shape
        xj = x.view(B, T, 13, 4)        # (B,T,J,4)
        h = self.proj(xj)               # (B,T,J,C)
        h = self.g1(h)
        h = self.g2(h)

        # temporal conv expects (B, C*J, T)
        h = h.permute(0, 2, 3, 1).contiguous()  # (B,J,C,T)
        h = h.view(B, 13 * h.shape[2], T)       # (B,J*C,T)
        h = self.tconv(h)
        h = self.bn(h)
        h = F.relu(h)
        h = self.drop(h)

        # pool: (B,J*C,T)->(B,J,C,T)->avg over J and T
        h = h.view(B, 13, -1, T).mean(dim=1).mean(dim=-1)  # (B,C)
        logits = self.fc(h)
        return logits


# ----------------------------
# Metrics (for validation during training)
# ----------------------------
@torch.no_grad()
def eval_argmax(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return float(correct / max(1, total))


# ----------------------------
# Training
# ----------------------------
@dataclass
class HParams:
    model_type: str
    in_dim: int = 52
    T: int = 96
    num_classes: int = 3
    # lstm
    lstm_hidden: int = 128
    lstm_layers: int = 2
    lstm_bidir: bool = True
    # gcn
    gcn_d_model: int = 64
    dropout: float = 0.2


def build_model(model_type: str, hp: HParams) -> nn.Module:
    if model_type == "lstm":
        return LSTMClassifier(
            in_dim=hp.in_dim,
            hidden=hp.lstm_hidden,
            num_layers=hp.lstm_layers,
            dropout=hp.dropout,
            num_classes=hp.num_classes,
            bidir=hp.lstm_bidir,
        )
    if model_type == "gcn":
        return GCNClassifier(
            in_dim=hp.in_dim,
            num_classes=hp.num_classes,
            d_model=hp.gcn_d_model,
            dropout=hp.dropout,
        )
    raise ValueError(f"Unknown model_type={model_type}")


def compute_class_weights(dataset: KickNPZDataset) -> torch.Tensor:
    counts = np.zeros(len(CLASSES), dtype=np.int64)
    for _, y in dataset.items:
        counts[y] += 1
    weights = 1.0 / np.maximum(counts, 1)
    weights = weights / weights.sum() * len(CLASSES)
    return torch.tensor(weights, dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, choices=["lstm", "gcn"], required=True)
    ap.add_argument("--train_align", type=str, choices=["uniform", "phase"], required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--save_dir", type=str, default=str(Path("models") / "checkpoints"))
    ap.add_argument("--save_name", type=str, default="")
    ap.add_argument("--seq_root", type=str, default=str(Path("scripts") / "data" / "sequences"))
    args = ap.parse_args()

    set_seed(args.seed)

    device = args.device
    seq_root = Path(args.seq_root)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    hp = HParams(model_type=args.model, dropout=float(args.dropout))

    train_ds = KickNPZDataset(seq_root=seq_root, split="train", align=args.train_align, T_out=hp.T)
    val_ds = KickNPZDataset(seq_root=seq_root, split="val", align=args.train_align, T_out=hp.T)

    # sampler for class imbalance (optional but recommended)
    class_w = compute_class_weights(train_ds)  # (3,)
    sample_w = []
    for _, y in train_ds.items:
        sample_w.append(float(class_w[y].item()))
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_batch)

    model = build_model(args.model, hp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_path = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_sum = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_sum += float(loss.item()) * y.numel()
            n_sum += y.numel()

        val_acc = eval_argmax(model, val_loader, device=device)
        train_loss = loss_sum / max(1, n_sum)

        print(f"[EPOCH {epoch:03d}] loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_name = args.save_name.strip()
            if not save_name:
                save_name = f"kick3_{args.model}_{args.train_align}_seed{args.seed}.pt"
            best_path = save_dir / save_name

            ckpt = {
                "model_type": args.model,
                "classes": CLASSES,
                "hparams": hp.__dict__,
                "seed": args.seed,
                "train_align": args.train_align,
                "val_acc": best_acc,
                "model": model.state_dict(),
            }
            torch.save(ckpt, best_path)
            print(f"[SAVE] best_acc={best_acc:.4f} -> {best_path}")

    if best_path is not None:
        print(f"[DONE] best ckpt: {best_path} (val_acc={best_acc:.4f})")
    else:
        print("[DONE] no checkpoint saved")


if __name__ == "__main__":
    main()
