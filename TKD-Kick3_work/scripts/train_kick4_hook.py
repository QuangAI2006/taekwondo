# scripts/train_kick4_hook.py
# Train a 4-class recognizer: front / roundhouse / axe (existing kick data, read-only)
# + hook_right (data/sequences_hook, built by hook_build_dataset.py).
#
# Reuses the exact model, preprocessing and features of train_kick3_baselines.py
# (imported, not modified). The original 3-class checkpoints in
# models/checkpoints/sample_level are never written; new checkpoints go to
# models/checkpoints/kick4_hook/ and results to outputs/hook/results/.
#
# Usage:
#   python scripts/train_kick4_hook.py                       (LSTM uniform, seeds 0 1 2)
#   python scripts/train_kick4_hook.py --model gcn --seeds 0

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_kick3_baselines import (
    CLASSES as KICK_CLASSES, HParams, build_feat_52, build_model, collate_batch, load_npz,
    normalize_kpts, resample_kpts, scan_npz, set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
CLASSES = KICK_CLASSES + ["hook_right"]  # kick indices unchanged, hook appended
DEFAULT_ROOTS = [ROOT / "data" / "sequences", ROOT / "data" / "sequences_hook"]
CKPT_DIR = ROOT / "models" / "checkpoints" / "kick4_hook"
RESULTS_DIR = ROOT / "outputs" / "hook" / "results"


def class_of(p: Path, seq_root: Path, split: str) -> str | None:
    """Class = first folder under <seq_root>/<split>/ (exact match, no substring guessing)."""
    rel = p.relative_to(seq_root / split)
    return rel.parts[0] if len(rel.parts) >= 2 and rel.parts[0] in CLASSES else None


def collect(roots: List[Path], split: str) -> List[Tuple[Path, int]]:
    items = []
    for r in roots:
        d = r / split
        if not d.exists():
            continue
        for p in scan_npz(d):
            c = class_of(p, r, split)
            if c is not None:
                items.append((p, CLASSES.index(c)))
    return items


class Kick4Dataset(Dataset):
    """Uniform alignment, identical to KickNPZDataset(align='uniform')."""

    def __init__(self, items: List[Tuple[Path, int]], T_out: int = 96) -> None:
        self.items = items
        self.T_out = T_out

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        p, y = self.items[idx]
        kpts, vis = load_npz(p)
        kpts = kpts[..., :2]  # the 13 legacy test files are (T,13,3); models use (x,y) only
        kpts_a, _ = resample_kpts(normalize_kpts(kpts, vis, tau=0.5), vis, self.T_out)
        return torch.from_numpy(build_feat_52(kpts_a)), torch.tensor(y, dtype=torch.long)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    C = len(CLASSES)
    cm = np.zeros((C, C), dtype=int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    per = {}
    recalls = []
    for i, c in enumerate(CLASSES):
        support = int(cm[i].sum())
        tp = int(cm[i, i])
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / support if support else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if support and (prec + rec) else 0.0
        per[c] = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": support}
        if support:
            recalls.append(rec)
    present = [c for c in CLASSES if per[c]["support"]]
    return {
        "n": int(len(y)),
        "accuracy": float((y == p).mean()) if len(y) else float("nan"),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_f1": float(np.mean([per[c]["f1"] for c in present])) if present else float("nan"),
        "per_class": per,
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
    }


def train_one(args, seed: int, roots: List[Path]) -> dict:
    set_seed(seed)
    hp = HParams(model_type=args.model, dropout=float(args.dropout), num_classes=len(CLASSES))

    tr_items, va_items, te_items = (collect(roots, s) for s in ("train", "val", "test"))
    counts = np.bincount([y for _, y in tr_items], minlength=len(CLASSES))
    print(f"[INFO] seed={seed} train counts " + ", ".join(f"{c}:{n}" for c, n in zip(CLASSES, counts))
          + f" | val={len(va_items)} test={len(te_items)}")
    if (counts == 0).any():
        raise RuntimeError(f"empty class in train: {dict(zip(CLASSES, counts.tolist()))}")

    # class-balanced sampler, same scheme as train_kick3_baselines.compute_class_weights
    class_w = 1.0 / np.maximum(counts, 1)
    class_w = class_w / class_w.sum() * len(CLASSES)
    sampler = WeightedRandomSampler([float(class_w[y]) for _, y in tr_items],
                                    num_samples=len(tr_items), replacement=True)

    train_ld = DataLoader(Kick4Dataset(tr_items, hp.T), batch_size=args.batch_size, sampler=sampler,
                          num_workers=0, collate_fn=collate_batch)
    val_ld = DataLoader(Kick4Dataset(va_items, hp.T), batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_batch)
    test_ld = DataLoader(Kick4Dataset(te_items, hp.T), batch_size=args.batch_size, shuffle=False,
                         num_workers=0, collate_fn=collate_batch)

    model = build_model(args.model, hp).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    name = f"kick4_{args.model}_uniform_seed{seed}.pt"
    ckpt_path = CKPT_DIR / name
    best, best_epoch = -1.0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in train_ld:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss.item()) * y.numel()
            n += y.numel()

        m = metrics(*predict(model, val_ld, args.device))
        # select on balanced accuracy: plain accuracy is dominated by the ~100 kick val samples
        sel = m["balanced_accuracy"]
        print(f"[EPOCH {epoch:03d}] loss={loss_sum / max(1, n):.4f} val_acc={m['accuracy']:.4f} "
              f"val_bacc={sel:.4f} hook_recall={m['per_class']['hook_right']['recall']:.2f}")
        if sel > best:
            best, best_epoch = sel, epoch
            torch.save({
                "model_type": args.model, "classes": CLASSES, "hparams": hp.__dict__, "seed": seed,
                "train_align": "uniform", "val_acc": m["accuracy"], "val_balanced_acc": sel,
                "best_epoch": epoch, "data_roots": [str(r) for r in roots], "model": model.state_dict(),
            }, ckpt_path)

    state = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state["model"])
    res = {
        "checkpoint": str(ckpt_path.relative_to(ROOT)), "model": args.model, "align": "uniform", "seed": seed,
        "best_epoch": best_epoch, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "dropout": args.dropout, "selection": "best val balanced accuracy",
        "train_counts": dict(zip(CLASSES, counts.tolist())), "elapsed_sec": round(time.time() - t0, 1),
        "val": metrics(*predict(model, val_ld, args.device)),
        "test": metrics(*predict(model, test_ld, args.device)),
        "protocol_note": "sample-level split; participant identifiers unavailable; "
                         "all hook_right clips come from one source video / one performer",
    }
    out = RESULTS_DIR / f"kick4_{args.model}_uniform_seed{seed}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[SAVE] {ckpt_path}\n[SAVE] {out}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, choices=["lstm", "gcn"], default="lstm")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seq_roots", type=str, nargs="+", default=[str(r) for r in DEFAULT_ROOTS])
    args = ap.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    roots = [Path(r) for r in args.seq_roots]

    rows = []
    for s in args.seeds:
        r = train_one(args, s, roots)
        for split in ("val", "test"):
            m = r[split]
            rows.append({"model": args.model, "seed": s, "split": split, "n": m["n"],
                         "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                         "macro_f1": m["macro_f1"],
                         **{f"recall_{c}": m["per_class"][c]["recall"] for c in CLASSES}})

    out = RESULTS_DIR / f"kick4_{args.model}_uniform_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[SAVE] {out}")


if __name__ == "__main__":
    main()
