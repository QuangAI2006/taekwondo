# scripts/action_classifier.py
"""
Model nhận dạng động tác (LSTM/GCN) — nạp checkpoint bất kỳ, số lớp & tên lớp đọc từ checkpoint.

- Checkpoint của tác giả (sample_lstm_uniform_seed*.pt): 3 lớp front/roundhouse/axe, cắt đoạn theo cổ chân.
- Checkpoint tự train bằng train_actions.py: bao nhiêu lớp cũng được; lưu thêm "window_mode" để lúc chạy
  cắt đoạn giống hệt lúc chuẩn bị dữ liệu train.

Tiền xử lý giống hệt eval_kick3_baselines.py: chuẩn hóa theo tâm hông + bề rộng vai/hông, resample đều
về T=96 frame, đặc trưng 52 chiều (26 vị trí + 26 vận tốc).
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

from action_pose import action_window
from eval_kick3_baselines import build_feat_52, build_model, normalize_kpts, resample_kpts

SCRIPTS_DIR = Path(__file__).resolve().parent
PKG_ROOT = SCRIPTS_DIR.parents[1]
CKPT_DIR = PKG_ROOT / "models" / "checkpoints" / "sample_level"


def default_ckpts(model: str = "lstm", align: str = "uniform", seeds: Sequence[int] = (0, 1, 2)) -> List[Path]:
    return [CKPT_DIR / f"sample_{model}_{align}_seed{s}.pt" for s in seeds]


def resolve_ckpts(spec: str) -> List[Path]:
    """spec: 'a.pt,b.pt', glob 'models/my/*.pt', hoặc thư mục (lấy mọi *.pt)."""
    out: List[Path] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        p = Path(part)
        if p.is_dir():
            out += sorted(p.glob("*.pt"))
        elif any(ch in part for ch in "*?["):
            out += [Path(x) for x in sorted(glob.glob(part))]
        else:
            out.append(p)
    if not out:
        raise FileNotFoundError(f"Không tìm thấy checkpoint nào từ: {spec}")
    return out


def prepare_input(kpts: np.ndarray, vis: np.ndarray, T_out: int = 96) -> np.ndarray:
    """Đoạn pose đã cắt -> đặc trưng (T_out, 52). Dùng chung cho train và inference."""
    kpts_n = normalize_kpts(kpts.astype(np.float32), vis.astype(np.float32), tau=0.5)
    kpts_a, _ = resample_kpts(kpts_n, vis, T_out)
    return build_feat_52(kpts_a)


class ActionClassifier:
    def __init__(self, ckpt_paths: Sequence[Path], device: str = "cpu"):
        self.device = device
        self.models, self.paths = [], [Path(p) for p in ckpt_paths]
        self.classes: List[str] = []
        self.window_mode = "ankle"
        self.T_out = 96
        self.model_type = ""
        for p in self.paths:
            if not p.is_file():
                raise FileNotFoundError(f"Không thấy checkpoint: {p}")
            ck = torch.load(str(p), map_location="cpu")
            classes = list(ck["classes"])
            if self.classes and classes != self.classes:
                raise ValueError(f"Các checkpoint trong ensemble có danh sách lớp khác nhau: {p.name}")
            self.classes = classes
            hp = ck["hparams"]
            self.T_out = int(hp.get("T", 96))
            # checkpoint của tác giả không có window_mode -> cắt theo cổ chân như lúc họ train
            self.window_mode = ck.get("window_mode", "ankle")
            self.model_type = ck["model_type"]
            m = build_model(ck["model_type"], hp, num_classes=len(classes))
            m.load_state_dict(ck["model"], strict=True)
            self.models.append(m.to(device).eval())

    @property
    def name(self) -> str:
        n = len(self.models)
        return f"{self.model_type.upper()}" + (f" ×{n}" if n > 1 else "")

    @torch.no_grad()
    def predict(self, kpts: np.ndarray, vis: np.ndarray, fps: float) -> Tuple[np.ndarray, Tuple[int, int, str]]:
        """Trả về (xác suất từng lớp, (s, e, window_mode thực tế))."""
        s, e, mode = action_window(kpts, vis, fps, self.window_mode)
        feat = prepare_input(kpts[s:e + 1], vis[s:e + 1], self.T_out)
        x = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        probs = [torch.softmax(m(x), dim=1).squeeze(0).cpu().numpy() for m in self.models]
        return np.mean(probs, axis=0), (s, e, mode)
