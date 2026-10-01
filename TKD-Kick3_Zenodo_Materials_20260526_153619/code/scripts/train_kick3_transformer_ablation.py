# scripts/train_kick3_transformer_ablation.py
# Train Kick3 (front/roundhouse/axe) with ablation switch: uniform vs phase-aligned resize
# Checkpoints:
#   models/checkpoints/kick3_{align}_seed{seed}.pt
#   models/checkpoints/kick3_{align}_seed{seed}_last.pt

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
SEQ_ROOT = BASE_DIR / "data" / "sequences"
CKPT_DIR = BASE_DIR / "models" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Labels / constants
# ----------------------------
ALL_CLASSES = ["front", "roundhouse", "axe"]
DEFAULT_LABELS = ALL_CLASSES
DEFAULT_LABEL_MAP = {c: i for i, c in enumerate(DEFAULT_LABELS)}

VTH = 0.2
HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(13)


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def find_class_from_path(p: Path) -> str | None:
    parts = [x.lower() for x in p.parts]
    for c in ALL_CLASSES:
        if c in parts:
            return c
    return None


# ----------------------------
# Utils: temporal resize + phase align
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
    T = kpts.shape[0]
    A, K, H, AS, KS, HS = _kick_side_indices(kpts)

    if ktype == "axe":
        ys = np.where(vis[:, A] > VTH, kpts[:, A, 1], np.inf)
        return int(np.argmin(ys))
    else:
        dist = np.full((T,), -1e9, np.float32)
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
# Features (match inference side)
# ----------------------------
def build_frame_features(kpts: np.ndarray, vis: np.ndarray, target_dim: int | None = None) -> np.ndarray:
    """
    Output in_dim = 13 * D * 2. For 2D: D=2 -> 52.
    """
    if kpts.ndim != 3:
        raise ValueError(f"kpts expected (T,13,D), got {kpts.shape}")
    T, J, D0 = kpts.shape

    D = D0
    if target_dim is not None:
        td = int(target_dim)
        if td in (2, 3):
            D = min(td, D0)
        else:
            base = J * 2  # 13*2
            if td % base == 0:
                D_guess = td // base
                if D_guess in (2, 3):
                    D = min(D_guess, D0)
                else:
                    raise ValueError(f"target_dim={td} -> D={D_guess}, only support 2/3")
            else:
                raise ValueError(f"target_dim only support 2/3 or 52/78, got {td}")

    pts = kpts[:, :, :D].astype(np.float32)
    v = vis.astype(np.float32) if vis is not None else np.ones((T, J), np.float32)

    pts_m = pts.copy()
    pts_m[v < VTH] = np.nan

    # center
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

    # scale
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
# Dataset
# ----------------------------
class KickSeqDataset(Dataset):
    def __init__(
        self,
        files: List[Path],
        label_map: Dict[str, int],
        frames: int = 96,
        time_jitter: bool = False,
        align: str = "phase",  # "phase" or "uniform"
        pre: float = 0.45,
        post: float = 0.55,
    ):
        self.files = files
        self.label_map = label_map
        self.frames = frames
        self.time_jitter = time_jitter
        self.align = align
        self.pre = float(pre)
        self.post = float(post)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        data = np.load(p, allow_pickle=True)

        kpts = data["kpts"].astype(np.float32)
        vis = data["vis"].astype(np.float32) if "vis" in data.files else np.ones(kpts.shape[:2], np.float32)

        feat = build_frame_features(kpts, vis)

        # time jitter (train only)
        if self.time_jitter and feat.shape[0] > 10:
            keep = np.random.uniform(0.85, 1.00)
            Lw = max(10, int(feat.shape[0] * keep))
            start = np.random.randint(0, max(1, feat.shape[0] - Lw + 1))
            feat = feat[start : start + Lw]
            kpts = kpts[start : start + Lw]
            vis = vis[start : start + Lw]

        cls = find_class_from_path(p)
        if cls is None:
            raise RuntimeError(f"Cannot parse class from path: {p}")

        if self.align == "phase":
            feat = phase_aligned_resize(feat, kpts, vis, self.frames, cls, pre=self.pre, post=self.post)
        else:
            feat = temporal_resize(feat, self.frames)

        y = self.label_map[cls]
        return torch.from_numpy(feat), torch.tensor(y, dtype=torch.long)


def make_collate():
    def _collate(batch):
        xs = torch.stack([b[0] for b in batch], 0)
        ys = torch.stack([b[1] for b in batch], 0)
        return xs, ys

    return _collate


# ----------------------------
# Model
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
# File scan / invalid filter
# ----------------------------
def _scan_split_npz(split_dir: Path) -> List[Path]:
    return sorted([p for p in split_dir.rglob("*.npz") if find_class_from_path(p) is not None])


def _filter_invalid(files: List[Path], include_invalid: bool) -> Tuple[List[Path], int]:
    if include_invalid:
        return files, 0
    kept: List[Path] = []
    removed = 0
    for p in files:
        try:
            d = np.load(p, allow_pickle=True)
            if "invalid_for_scoring" in d.files:
                inv = int(np.asarray(d["invalid_for_scoring"]).reshape(-1)[0])
                if inv == 1:
                    removed += 1
                    continue
        except Exception:
            pass
        kept.append(p)
    return kept, removed


def load_split_files(seq_root: Path, split: str, include_invalid: bool) -> List[Path]:
    split_dir = seq_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")
    files = _scan_split_npz(split_dir)
    files, removed = _filter_invalid(files, include_invalid=include_invalid)
    if removed > 0:
        print(f"[INFO] split={split} removed invalid_for_scoring=1: {removed}")
    return files


def split_stats(files: List[Path]) -> Dict[str, int]:
    counts = {c: 0 for c in ALL_CLASSES}
    for p in files:
        c = find_class_from_path(p)
        if c is not None:
            counts[c] += 1
    return counts


# ----------------------------
# Train loop
# ----------------------------
def run_epoch(model, loader, crit, opt=None, device="cpu"):
    model.train(opt is not None)
    total, correct, loss_sum = 0, 0, 0.0

    for xs, ys in loader:
        xs = xs.to(device).float()
        ys = ys.to(device)

        logits = model(xs)
        loss = crit(logits, ys)

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        with torch.no_grad():
            pred = logits.argmax(1)
            correct += int((pred == ys).sum().item())
            total += ys.size(0)
            loss_sum += float(loss.item()) * ys.size(0)

    return (correct / max(1, total)), (loss_sum / max(1, total))


def _clone_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        out[k] = v.detach().cpu().clone()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="val")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true", help="Enable deterministic behavior (recommended for multi-seed tables).")

    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--include_invalid", action="store_true")

    # Ablation knobs
    ap.add_argument("--align", type=str, default="phase", choices=["phase", "uniform"])
    ap.add_argument("--pre", type=float, default=0.45)
    ap.add_argument("--post", type=float, default=0.55)

    # Naming
    ap.add_argument("--ckpt_prefix", type=str, default="kick3", help="Prefix for ckpt names, e.g., kick3.")
    ap.add_argument("--ckpt_dir", type=str, default=str(CKPT_DIR))

    args = ap.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")
    print(f"[INFO] SEQ_ROOT={SEQ_ROOT}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = ckpt_dir / f"{args.ckpt_prefix}_{args.align}_seed{args.seed}.pt"
    last_ckpt = ckpt_dir / f"{args.ckpt_prefix}_{args.align}_seed{args.seed}_last.pt"
    label_map_path = ckpt_dir / f"{args.ckpt_prefix}_label_map.json"

    label_map = {c: i for i, c in enumerate(ALL_CLASSES)}

    train_files = load_split_files(SEQ_ROOT, args.train_split, include_invalid=args.include_invalid)
    val_files = load_split_files(SEQ_ROOT, args.val_split, include_invalid=args.include_invalid)

    tr_counts = split_stats(train_files)
    va_counts = split_stats(val_files)

    print(f"[INFO] train={len(train_files)} | " + ", ".join([f"{c}:{tr_counts[c]}" for c in ALL_CLASSES]))
    print(f"[INFO]   val={len(val_files)} | " + ", ".join([f"{c}:{va_counts[c]}" for c in ALL_CLASSES]))

    if len(train_files) == 0 or len(val_files) == 0:
        raise RuntimeError("train/val empty. Please generate .npz under data/sequences/train and data/sequences/val")

    total = sum(tr_counts.values())
    class_weights = torch.tensor(
        [(total / max(1, tr_counts[c]) if tr_counts[c] > 0 else 0.0) for c in ALL_CLASSES],
        dtype=torch.float32,
    )

    train_ds = KickSeqDataset(
        train_files,
        label_map,
        frames=args.frames,
        time_jitter=True,
        align=args.align,
        pre=args.pre,
        post=args.post,
    )
    val_ds = KickSeqDataset(
        val_files,
        label_map,
        frames=args.frames,
        time_jitter=False,
        align=args.align,
        pre=args.pre,
        post=args.post,
    )

    sample_weights = [1.0 / max(1, tr_counts[find_class_from_path(p)]) for p in train_files]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=make_collate())
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=make_collate())

    in_dim = int(train_ds[0][0].shape[-1])
    print(f"[INFO] in_dim={in_dim} align={args.align} pre={args.pre} post={args.post}")

    model = KickTransformer(in_dim=in_dim, dropout=args.dropout, num_classes=len(ALL_CLASSES)).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=args.label_smoothing)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    best_acc = 0.0
    best_epoch = 0

    if args.resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch = int(state.get("epoch", 1))
        best_acc = float(state.get("best_acc", 0.0))
        best_epoch = int(state.get("best_epoch", 0))
        print(f"[RESUME] {last_ckpt} epoch={start_epoch} best_acc={best_acc:.3f} best_epoch={best_epoch}")

    best_state = None

    for epoch in range(start_epoch, args.epochs + 1):
        tr_acc, tr_loss = run_epoch(model, train_ld, crit, opt=opt, device=device)
        va_acc, va_loss = run_epoch(model, val_ld, crit, opt=None, device=device)

        print(f"Epoch {epoch:02d} | train_acc={tr_acc:.3f} loss={tr_loss:.4f} | val_acc={va_acc:.3f} loss={va_loss:.4f}")

        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch + 1,
                "best_acc": max(best_acc, va_acc),
                "best_epoch": best_epoch if best_epoch > 0 else epoch,
                "in_dim": in_dim,
                "align": args.align,
                "seed": args.seed,
            },
            last_ckpt,
        )

        if va_acc >= best_acc:
            best_acc = float(va_acc)
            best_epoch = int(epoch)
            best_state = {
                "model": _clone_state_dict(model.state_dict()),
                "label_map": label_map,
                "frames": int(args.frames),
                "in_dim": int(in_dim),
                "d_model": 128,
                "nhead": 4,
                "num_layers": 2,
                "dropout": float(args.dropout),
                "best_acc": float(best_acc),
                "best_epoch": int(best_epoch),
                "align": args.align,
                "seed": args.seed,
                "pre": float(args.pre),
                "post": float(args.post),
            }
            torch.save(best_state, best_ckpt)

    if best_state is None:
        best_state = {
            "model": _clone_state_dict(model.state_dict()),
            "label_map": label_map,
            "frames": int(args.frames),
            "in_dim": int(in_dim),
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2,
            "dropout": float(args.dropout),
            "best_acc": float(best_acc),
            "best_epoch": int(best_epoch),
            "align": args.align,
            "seed": args.seed,
            "pre": float(args.pre),
            "post": float(args.post),
        }
        torch.save(best_state, best_ckpt)

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] best_ckpt: {best_ckpt}")
    print(f"[SAVE] last_ckpt: {last_ckpt}")
    print(f"[SAVE] label_map: {label_map_path}")
    print(f"[INFO] best_acc={best_state.get('best_acc', best_acc):.3f} best_epoch={best_state.get('best_epoch', best_epoch)}")


if __name__ == "__main__":
    main()
