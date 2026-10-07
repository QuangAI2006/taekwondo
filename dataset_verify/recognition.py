"""Adapter for the project's checkpoint-based action recognizer."""
from __future__ import annotations

import importlib
import sys

import numpy as np


class RecognitionBackend:
    def __init__(self, scripts, checkpoint_spec=None, device='cpu'):
        sys.path.insert(0, str(scripts))
        module = importlib.import_module('action_classifier')
        paths = (module.resolve_ckpts(checkpoint_spec) if checkpoint_spec else
                 [scripts.parent.parent / 'models/checkpoints/sample_level' /
                  f'sample_lstm_uniform_seed{s}.pt' for s in (0, 1, 2)])
        self.classifier = module.ActionClassifier(paths, device=device)
        self.paths = paths
        self.classes = list(self.classifier.classes)
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError('Checkpoint cần danh sách lớp duy nhất, không rỗng.')
        # The upstream ensemble uses one preprocessing setup for every member.
        import torch
        settings = []
        for path in paths:
            ck = torch.load(str(path), map_location='cpu', weights_only=True)
            settings.append((int(ck['hparams'].get('T', 96)), ck.get('window_mode', 'ankle')))
        if len(set(settings)) != 1:
            raise ValueError('Checkpoint ensemble phải có cùng T và window_mode.')

    def predict(self, pose):
        k, v = np.asarray(pose['kpts']), np.asarray(pose['vis'])
        if k.ndim != 3 or k.shape[1:] != (13, 2) or v.shape != k.shape[:2] or len(k) < 2:
            raise ValueError('Không đủ pose gốc hợp lệ để nhận diện động tác.')
        present = np.isfinite(k).all(axis=2) & np.isfinite(v) & (v >= .5)
        if np.count_nonzero(present[:, [1, 2, 7, 8]].all(axis=1)) < 2:
            raise ValueError('Không thấy rõ vai/hông ở đủ frame để nhận diện.')
        fps = float(pose['fps'])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('FPS pose không hợp lệ để nhận diện.')
        probs, window = self.classifier.predict(k, v, fps)
        probs = np.asarray(probs, dtype=float)
        if probs.shape != (len(self.classes),) or not np.isfinite(probs).all() or np.any(probs < 0) or not np.isclose(probs.sum(), 1, atol=1e-4):
            raise ValueError('Model trả xác suất không hợp lệ.')
        order = np.argsort(probs)[::-1]
        winner = int(order[0])
        margin = float(probs[winner] - probs[order[1]]) if len(order) > 1 else float(probs[winner])
        return dict(used=True, model=self.classifier.name, classes=self.classes,
                    probabilities={c: float(p) for c, p in zip(self.classes, probs)},
                    predicted_action=self.classes[winner], confidence=float(probs[winner]),
                    margin=margin, window=list(window))
