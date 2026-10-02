# scripts/train_actions.py
"""
Train model nhận dạng (LSTM / GCN của TKD-Kick3) với SỐ ĐỘNG TÁC BẤT KỲ, trên dữ liệu từ prepare_action_data.py.

- Danh sách động tác lấy từ <data>/meta.json (hoặc tên thư mục <data>/train/*).
- Tiền xử lý dùng chung action_classifier.prepare_input (= eval_kick3_baselines): checkpoint chạy được
  ngay với infer_action.py --ckpt.
- Tăng cường dữ liệu: lật trái-phải (đổi tay/chân trái<->phải) và co giãn thời gian nhẹ.
- Lưu checkpoint tốt nhất theo macro-F1 trên val: <out>/<model>_seed<k>.pt (kèm classes, window_mode).

Ví dụ:
  python code/scripts/train_actions.py --data data/my_sequences --out models/checkpoints/my_actions
  python code/scripts/infer_action.py video.mp4 --ckpt "models/checkpoints/my_actions/*.pt"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from action_classifier import prepare_input  # noqa: E402
from eval_kick3_baselines import build_model  # noqa: E402

# hoán đổi trái <-> phải khi lật ảnh (thứ tự 13 khớp: HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA)
FLIP_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]


def load_split(data: Path, split: str, classes: List[str]) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    out = []
    for ci, c in enumerate(classes):
        for f in sorted((data / split / c).glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            k, v = z["kpts"].astype(np.float32), z["vis"].astype(np.float32)
            if k.shape[0] >= 2:
                out.append((k, v, ci))
    return out


def augment(k: np.ndarray, v: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:  # lật trái-phải: động tác không đổi, chỉ đổi bên
        k = k[:, FLIP_PERM].copy()
        v = v[:, FLIP_PERM].copy()
        k[..., 0] = -k[..., 0]
    T = k.shape[0]
    if T >= 10 and rng.random() < 0.5:  # cắt bớt tối đa 10% mỗi đầu (co giãn thời gian)
        a = rng.randint(0, T // 10)
        b = T - rng.randint(0, T // 10)
        k, v = k[a:b], v[a:b]
    return k, v


def to_tensor(samples, T_out: int, rng: random.Random = None):
    feats, ys = [], []
    for k, v, y in samples:
        if rng is not None:
            k, v = augment(k, v, rng)
        feats.append(prepare_input(k, v, T_out))
        ys.append(y)
    return torch.from_numpy(np.stack(feats)).float(), torch.tensor(ys, dtype=torch.long)


def evaluate(model, X, Y, n_cls):
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(1).numpy()
    y = Y.numpy()
    cm = np.zeros((n_cls, n_cls), np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(n_cls), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(n_cls), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n_cls), where=(prec + rec) > 0)
    present = cm.sum(1) > 0
    return float((pred == y).mean()), float(f1[present].mean()), cm, prec, rec, f1


def main() -> None:
    ap = argparse.ArgumentParser(description="Train model nhận dạng nhiều động tác")
    ap.add_argument("--data", required=True, help="thư mục từ prepare_action_data.py")
    ap.add_argument("--out", required=True, help="thư mục lưu checkpoint")
    ap.add_argument("--model", choices=["lstm", "gcn"], default="lstm")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seeds", default="0", help="vd '0,1,2' để train 3 model (ensemble khi chạy)")
    ap.add_argument("--T", type=int, default=96)
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    meta_p = data / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
    classes = meta.get("classes") or sorted(p.name for p in (data / "train").iterdir() if p.is_dir())
    window_mode = meta.get("window_mode", "motion")
    n_cls = len(classes)

    train = load_split(data, "train", classes)
    val = load_split(data, "val", classes)
    if not val:
        sys.exit("Không có dữ liệu val — chạy lại prepare_action_data.py (mặc định tự chia 20% làm val).")
    print(f"[data] {n_cls} động tác {classes} · train={len(train)} val={len(val)} · cắt đoạn: {window_mode}")
    cnt = np.bincount([y for *_, y in train], minlength=n_cls)
    for c, n in zip(classes, cnt):
        print(f"   {c:<14s} train={n}")
    weights = torch.tensor(cnt.sum() / np.maximum(cnt, 1) / n_cls, dtype=torch.float32)
    Xv, Yv = to_tensor(val, args.T)

    hp = {"model_type": args.model, "in_dim": 52, "T": args.T, "num_classes": n_cls,
          "lstm_hidden": 128, "lstm_layers": 2, "lstm_bidir": True, "gcn_d_model": 64, "dropout": args.dropout}
    out.mkdir(parents=True, exist_ok=True)

    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = random.Random(seed)
        model = build_model(args.model, hp, num_classes=n_cls)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        lossf = nn.CrossEntropyLoss(weight=weights)
        best = (-1.0, None, None)
        t_start = time.perf_counter()
        for ep in range(1, args.epochs + 1):
            X, Y = to_tensor(train, args.T, rng)  # tăng cường mới mỗi epoch
            perm = torch.randperm(len(Y))
            model.train()
            tot = 0.0
            for i in range(0, len(Y), args.batch_size):
                idx = perm[i:i + args.batch_size]
                opt.zero_grad()
                loss = lossf(model(X[idx]), Y[idx])
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            acc, mf1, *_ = evaluate(model, Xv, Yv, n_cls)
            if mf1 > best[0]:
                best = (mf1, acc, {k: v.clone() for k, v in model.state_dict().items()})
            if ep == 1 or ep % 5 == 0 or ep == args.epochs:
                print(f"[seed {seed}] epoch {ep:3d}  loss={tot / len(Y):.4f}  val acc={acc:.3f}  macroF1={mf1:.3f}"
                      f"  ({time.perf_counter() - t_start:.0f}s)")

        model.load_state_dict(best[2])
        acc, mf1, cm, prec, rec, f1 = evaluate(model, Xv, Yv, n_cls)
        path = out / f"{args.model}_seed{seed}.pt"
        torch.save({"model_type": args.model, "classes": classes, "hparams": hp, "seed": seed,
                    "window_mode": window_mode, "val_acc": acc, "val_macro_f1": mf1,
                    "trained_with": "train_actions.py", "model": best[2]}, path)
        print(f"\n[SAVED] {path}  (val acc={acc:.3f}, macro-F1={mf1:.3f})")
        print("Theo từng động tác (val):")
        for i, c in enumerate(classes):
            print(f"   {c:<14s} precision={prec[i]:.2f} recall={rec[i]:.2f} f1={f1[i]:.2f} n={int(cm[i].sum())}")
        print("Ma trận nhầm lẫn (hàng = thật, cột = dự đoán):")
        print(" " * 15 + "".join(f"{c[:9]:>10s}" for c in classes))
        for i, c in enumerate(classes):
            print(f"   {c:<12s}" + "".join(f"{x:>10d}" for x in cm[i]))

    print(f"\nChạy thử: python code/scripts/infer_action.py <video> --ckpt \"{out}/*.pt\"")


if __name__ == "__main__":
    main()
