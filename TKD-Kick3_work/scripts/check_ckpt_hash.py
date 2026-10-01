# -*- coding: utf-8 -*-
"""
check_ckpt_hash.py
比较两个 PyTorch checkpoint 的“模型权重”是否一致（用于定位 best 是否被覆盖成 last）。

用法示例：
  python .\scripts\check_ckpt_hash.py
  python .\scripts\check_ckpt_hash.py --ckpt1 .\models\checkpoints\kick3_uni.pt --ckpt2 .\models\checkpoints\kick3_uni_last.pt
  python .\scripts\check_ckpt_hash.py --ckpt1 .\models\checkpoints\kick3_pa.pt  --ckpt2 .\models\checkpoints\kick3_pa_last.pt
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Any

import torch


def _extract_state_dict(ckpt_obj: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    """
    从各种可能的 ckpt 格式中提取 state_dict。
    返回：(state_dict, source_tag)
    """
    if isinstance(ckpt_obj, dict):
        # 你项目里常见格式：{"model": state_dict, ...}
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"], "model"
        # 训练脚本常见格式：{"model_state": state_dict, ...}
        if "model_state" in ckpt_obj and isinstance(ckpt_obj["model_state"], dict):
            return ckpt_obj["model_state"], "model_state"
        # 有些保存会用 state_dict 字段
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"], "state_dict"

        # 兜底：判断是否“本身就是一个 state_dict”
        # 典型特征：key 是字符串，value 是 Tensor
        is_sd = True
        for k, v in ckpt_obj.items():
            if not isinstance(k, str):
                is_sd = False
                break
            if not torch.is_tensor(v):
                is_sd = False
                break
        if is_sd and len(ckpt_obj) > 0:
            return ckpt_obj, "raw_dict_as_state_dict"

    raise ValueError("无法从 ckpt 中解析出 state_dict（不支持的保存格式）。")


def sha_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    计算 state_dict 的 SHA256 哈希（对 key + tensor bytes 做稳定拼接）。
    """
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        if not torch.is_tensor(v):
            continue
        t = v.detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def summarize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    """
    返回：(参数张量数, 总元素数, 总字节数)
    """
    n_tensors = 0
    n_elems = 0
    n_bytes = 0
    for _, v in state_dict.items():
        if not torch.is_tensor(v):
            continue
        n_tensors += 1
        t = v.detach().cpu()
        n_elems += int(t.numel())
        n_bytes += int(t.numel() * t.element_size())
    return n_tensors, n_elems, n_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt1", type=str, default=r".\models\checkpoints\kick3_uni.pt", help="第一个 ckpt 路径")
    ap.add_argument("--ckpt2", type=str, default=r".\models\checkpoints\kick3_uni_last.pt", help="第二个 ckpt 路径")
    ap.add_argument("--device", type=str, default="cpu", help="map_location 设备（默认 cpu）")
    args = ap.parse_args()

    p1 = Path(args.ckpt1)
    p2 = Path(args.ckpt2)

    if not p1.exists():
        raise FileNotFoundError(f"找不到 ckpt1: {p1.resolve()}")
    if not p2.exists():
        raise FileNotFoundError(f"找不到 ckpt2: {p2.resolve()}")

    ckpt1 = torch.load(str(p1), map_location=args.device)
    ckpt2 = torch.load(str(p2), map_location=args.device)

    sd1, tag1 = _extract_state_dict(ckpt1)
    sd2, tag2 = _extract_state_dict(ckpt2)

    h1 = sha_state_dict(sd1)
    h2 = sha_state_dict(sd2)

    n1 = summarize_state_dict(sd1)
    n2 = summarize_state_dict(sd2)

    print(f"[CKPT1] {p1}  state_dict_from={tag1}")
    print(f"        tensors={n1[0]} elems={n1[1]} bytes={n1[2]}")
    print(f"        sha256={h1}")
    print(f"[CKPT2] {p2}  state_dict_from={tag2}")
    print(f"        tensors={n2[0]} elems={n2[1]} bytes={n2[2]}")
    print(f"        sha256={h2}")
    print(f"[SAME ] {h1 == h2}")


if __name__ == "__main__":
    main()
