# scripts/pipeline_viewer.py
"""
Cửa sổ trực quan hóa pipeline (matplotlib, cập nhật trực tiếp khi code chạy) — dùng cho mọi động tác.

Bố cục:
  - Hàng trên : sơ đồ các bước; bước đang chạy tô cam, xong tô xanh, lỗi tô đỏ, kèm kết quả từng bước.
  - Trái      : video. Lúc chạy MediaPipe: khung xương vẽ trực tiếp từng frame. Sau khi chấm: phát lại
                chậm 2×, tay/chân ra đòn tô đỏ, ghi tên pha.
  - Trái dưới : xác suất từng động tác của model nhận dạng (hoặc lý do bỏ qua bước nhận dạng).
  - Phải trên : tín hiệu dùng để chia pha + các pha + đoạn đưa vào model; con trỏ chạy theo video.
  - Phải dưới : điểm từng hạng mục (Chính xác 4 + Biểu hiện 6).

Dùng trong infer_action.py (và các lớp bọc infer_video.py / infer_punch_video.py).
"""

from __future__ import annotations

import textwrap
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from viewer_text_en import en, item_en

EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8),
         (7, 9), (9, 11), (8, 10), (10, 12)]
# giao diện cửa sổ bằng tiếng Anh; chuỗi tiếng Việt từ luật chấm được dịch qua viewer_text_en.py
STEPS_ACTION = ["Read video", "MediaPipe Pose\n(2D + 3D)", "Select person\n& crop", "Recognition\n(kicks only)",
                "Phases\n& metrics", "Scoring\n4 + 6"]
FACE = {"todo": "#eeeeee", "active": "#ffcc80", "done": "#c8e6c9", "fail": "#ffcdd2", "skip": "#e0e0e0"}
EDGE = {"todo": "#bdbdbd", "active": "#ef6c00", "done": "#2e7d32", "fail": "#c62828", "skip": "#9e9e9e"}


def _draw_skeleton(img: np.ndarray, k: np.ndarray, scale: float, highlight: Sequence[int] = (),
                   thick: int = 2) -> None:
    hl = set(highlight)
    for a, b in EDGES:
        if not np.all(np.isfinite(k[[a, b]])):
            continue
        pa = tuple(int(v * scale) for v in k[a])
        pb = tuple(int(v * scale) for v in k[b])
        on = a in hl and b in hl
        cv2.line(img, pa, pb, (255, 40, 40) if on else (40, 220, 40), thick + (1 if on else 0), cv2.LINE_AA)
    for p in k:
        if np.all(np.isfinite(p)):
            cv2.circle(img, (int(p[0] * scale), int(p[1] * scale)), thick + 1, (255, 255, 255), -1, cv2.LINE_AA)


class PipelineViewer:
    def __init__(self, title: str, steps: Sequence[str] = STEPS_ACTION, show_probs: bool = True,
                 interactive: bool = True, disp_h: int = 360, max_store: int = 900):
        self.interactive = interactive and matplotlib.get_backend().lower() != "agg"
        self.disp_h, self.max_store = disp_h, max_store
        self.frames: List[np.ndarray] = []
        self.scale = 1.0
        self._last_draw = 0.0
        self.closed = False
        self.anim = None
        self.res = None
        self.cursor = None

        if self.interactive:
            plt.ion()
        self.fig = plt.figure(figsize=(15.5, 8.8))
        try:
            self.fig.canvas.manager.set_window_title(f"Scoring pipeline — {title}")
        except Exception:
            pass
        self.fig.suptitle(title, fontsize=13, fontweight="bold", y=0.985)
        gs = GridSpec(3, 2, figure=self.fig, height_ratios=[0.7, 2.3, 1.9], width_ratios=[1.0, 0.85],
                      left=0.02, right=0.95, top=0.95, bottom=0.07, hspace=0.45, wspace=0.42)
        self.ax_flow = self.fig.add_subplot(gs[0, :])
        if show_probs:
            self.ax_vid = self.fig.add_subplot(gs[1, 0])
            self.ax_prob = self.fig.add_subplot(gs[2, 0])
        else:
            self.ax_vid = self.fig.add_subplot(gs[1:, 0])
            self.ax_prob = None
        self.ax_sig = self.fig.add_subplot(gs[1, 1])
        self.ax_bar = self.fig.add_subplot(gs[2, 1])

        self._init_flow(steps)
        self.ax_vid.axis("off")
        self.ax_vid.set_title("Video", fontsize=11)
        self.vid_text = self.ax_vid.text(0.5, 0.5, "Opening video...", ha="center", va="center",
                                         transform=self.ax_vid.transAxes, fontsize=12, color="#616161")
        self.im = None
        waits = [(self.ax_sig, "Waiting for phase detection"), (self.ax_bar, "Waiting for scoring")]
        if self.ax_prob is not None:
            waits.append((self.ax_prob, "Waiting for recognition"))
        for ax, msg in waits:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#9e9e9e")
        self.note = self.fig.text(0.02, 0.012, "", fontsize=9, color="#b71c1c", va="bottom")
        self._refresh()
        if self.interactive:
            plt.show(block=False)

    # ---------- sơ đồ pipeline ----------
    def _init_flow(self, steps: Sequence[str]) -> None:
        ax = self.ax_flow
        n = len(steps)
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.axis("off")
        self.boxes, self.status, self.states = [], [], ["todo"] * n
        for i, name in enumerate(steps):
            box = FancyBboxPatch((i + 0.08, 0.38), 0.84, 0.58, boxstyle="round,pad=0.01,rounding_size=0.06",
                                 fc=FACE["todo"], ec=EDGE["todo"], lw=2)
            ax.add_patch(box)
            ax.text(i + 0.5, 0.67, f"{i + 1}. {name}", ha="center", va="center", fontsize=10, fontweight="bold")
            self.boxes.append(box)
            self.status.append(ax.text(i + 0.5, 0.17, "", ha="center", va="center", fontsize=8.5, color="#424242"))
            if i < n - 1:
                ax.add_patch(FancyArrowPatch((i + 0.93, 0.67), (i + 1.07, 0.67), arrowstyle="-|>",
                                             mutation_scale=14, color="#757575"))

    def step(self, i: int, state: str = "active", status: Optional[str] = None) -> None:
        """Đặt trạng thái bước i (0-based): active / done / fail / skip. Các bước trước còn dở -> done."""
        for j in range(i):
            if self.states[j] in ("todo", "active"):
                self.states[j] = "done"
        self.states[i] = state
        for box, st in zip(self.boxes, self.states):
            box.set_facecolor(FACE[st])
            box.set_edgecolor(EDGE[st])
        if status is not None:
            self.status[i].set_text(status)
        self._refresh()

    # ---------- MediaPipe chạy từng frame ----------
    def on_frame(self, t: int, frame_bgr: np.ndarray, people: Sequence[np.ndarray], n_frames: int) -> None:
        h, w = frame_bgr.shape[:2]
        self.scale = self.disp_h / h
        small = cv2.cvtColor(cv2.resize(frame_bgr, (int(w * self.scale), self.disp_h)), cv2.COLOR_BGR2RGB)
        if len(self.frames) < self.max_store:
            self.frames.append(small)
        if self.closed:
            return
        now = time.perf_counter()
        if now - self._last_draw < 0.06 and t + 1 < n_frames:  # tối đa ~16 lần vẽ/giây
            return
        self._last_draw = now
        img = small.copy()
        for k in people:
            _draw_skeleton(img, k, self.scale)
        self._show_image(img, f"MediaPipe processing: frame {t + 1}/{n_frames} — {len(people)} person(s)")
        self.status[1].set_text(f"frame {t + 1}/{n_frames}")
        self._refresh()

    def _show_image(self, img: np.ndarray, title: str) -> None:
        if self.im is None:
            self.vid_text.set_visible(False)
            self.im = self.ax_vid.imshow(img)
        else:
            self.im.set_data(img)
        self.ax_vid.set_title(title, fontsize=11)

    # ---------- nhận dạng ----------
    def show_probs(self, labels: Sequence[str], probs: Sequence[float], model_name: str,
                   scored_as: Optional[str] = None, note: str = "") -> None:
        ax = self.ax_prob
        if ax is None:
            return
        ax.clear()
        order = np.argsort(probs)
        ys = np.arange(len(order))
        best = int(order[-1])
        # tên động tác ghi phía trên mỗi thanh (không dùng nhãn trục để khỏi bị cắt khi tên dài)
        ax.barh(ys, [probs[i] * 100 for i in order], height=0.42,
                color=["#fb8c00" if i == best else "#90a4ae" for i in order])
        for y, i in zip(ys, order):
            ax.text(0, y + 0.27, labels[i], va="bottom", fontsize=9,
                    fontweight="bold" if i == best else "normal")
            ax.text(probs[i] * 100 + 1, y, f"{probs[i] * 100:.1f}%", va="center", fontsize=9)
        ax.set_xlim(0, 115)
        ax.set_ylim(-0.5, len(order) - 0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"Recognition ({model_name})" + (f"  →  scored as: {scored_as}" if scored_as else ""),
                     fontsize=10)
        if note:
            ax.text(0.5, -0.06, note, transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
                    color="#b71c1c")
        self._refresh()

    def skip_probs(self, msg: str) -> None:
        if self.ax_prob is None:
            return
        self.ax_prob.clear()
        self.ax_prob.axis("off")
        self.ax_prob.text(0.5, 0.5, msg, ha="center", va="center", transform=self.ax_prob.transAxes,
                          fontsize=10, color="#616161")
        self._refresh()

    # ---------- kết quả ----------
    def show_result(self, kpts: np.ndarray, fps: float, res, model_window: Optional[Tuple[int, int]] = None) -> None:
        """res: action_scorers.ScoreResult."""
        self.kpts, self.fps, self.res = kpts, fps, res
        self._plot_signals(res, model_window)
        self._plot_scores(res)
        warns = list(res.warnings) if res.valid else [res.reason]
        if warns:
            self.note.set_text("\n".join(textwrap.fill("⚠ " + en(w_), 150) for w_ in warns if w_))
        self._start_replay()
        self._refresh()

    def _phase(self, t: int) -> str:
        r = self.res
        if r is None or not r.valid or r.phases.get("t0") is None:
            return ""
        t0, tp = r.phases["t0"], r.phases["t_peak"]
        te = r.phases.get("t_end")
        te = len(self.kpts) - 1 if te is None else te
        names = r.phase_names
        if t < t0:
            return names[0]
        if t < tp:
            return names[1]
        if t == tp:
            return names[2]
        if t <= te:
            return names[3]
        return names[4]

    def _plot_signals(self, r, model_window) -> None:
        ax = self.ax_sig
        ax.clear()
        s = r.series or {}
        if s.get("sig") is None:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "No signal", ha="center", va="center", transform=ax.transAxes)
            return
        T = len(s["sig"])
        x = np.arange(T)
        ax.plot(x, s["sig"], color="#d32f2f", lw=2)
        ax.set_ylabel(en(s.get("sig_name", "")), color="#d32f2f", fontsize=9)
        ax.tick_params(axis="y", colors="#d32f2f", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("frame", fontsize=9)
        ax.set_xlim(0, max(T - 1, 1))
        if s.get("speed") is not None:
            ax2 = ax.twinx()
            ax2.plot(x, s["speed"], color="#1565c0", lw=1.5, alpha=0.8)
            ax2.set_ylabel(en(s.get("speed_name", "")), color="#1565c0", fontsize=9)
            ax2.tick_params(axis="y", colors="#1565c0", labelsize=8)
        if model_window is not None:
            ax.axvspan(model_window[0], model_window[1], facecolor="none", edgecolor="#9e9e9e",
                       hatch="..", lw=0, zorder=0)
        if r.valid and r.phases.get("t0") is not None:
            t0, tp, te = r.phases["t0"], r.phases["t_peak"], r.phases.get("t_end")
            ax.axvspan(t0, tp, color="#ff9800", alpha=0.18)
            ax.axvspan(tp, te if te is not None else T - 1, color="#42a5f5", alpha=0.12)
            marks = [(t0, r.phase_labels[0]), (tp, r.phase_labels[1])]
            if te is not None:
                marks.append((te, r.phase_labels[2]))
            prev = None
            for t, lab in marks:
                ax.axvline(t, color="#424242", ls="--", lw=1)
                # 2 mốc sát nhau -> nhãn sau ghi vào trong khung (nền trắng) cho khỏi đè
                inside = prev is not None and abs(t - prev) < 0.1 * T
                ax.text(t, 0.97 if inside else 1.01, en(lab), transform=ax.get_xaxis_transform(), ha="center",
                        va="top" if inside else "bottom", fontsize=8.5,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1) if inside else None)
                prev = t
        title = "Phases  (orange = strike, blue = recovery"
        title += ", dots = model input)" if model_window is not None else ")"
        ax.set_title(title, fontsize=10, pad=14)
        self.cursor = ax.axvline(0, color="black", lw=2)

    def _plot_scores(self, r) -> None:
        ax = self.ax_bar
        ax.clear()
        if not r.valid:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "NOT SCORED\n\n" + textwrap.fill(en(r.reason), 60),
                    ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#c62828")
            return
        items = list(r.acc_items) + list(r.expr_items)
        ys = np.arange(len(items))[::-1]
        for y, it in zip(ys, items):
            ax.barh(y, it["maxi"], color="#f5f5f5", edgecolor="#bdbdbd", height=0.7)
            if it.get("got") is None:
                ax.barh(y, it["maxi"], color="none", edgecolor="#9e9e9e", hatch="//", height=0.7)
                ax.text(it["maxi"] + 0.04, y, "not assessed", va="center", fontsize=8, color="#757575")
            else:
                col = "#1e88e5" if it["category"] == "accuracy" else "#43a047"
                ax.barh(y, it["got"], color=col, height=0.7)
                ax.text(it["maxi"] + 0.04, y, f"{it['got']:.2f}/{it['maxi']:.1f}", va="center", fontsize=8)
        ax.set_yticks(ys, [f"{it['code']}  {textwrap.shorten(item_en(it), 32, placeholder='…')}" for it in items],
                      fontsize=8)
        ax.set_xlim(0, 2.6)
        ax.set_xticks([])
        n_acc = len(r.acc_items)
        if 0 < n_acc < len(items):
            ax.axhline(ys[n_acc - 1] - 0.5, color="#616161", lw=1)
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"Total {r.total:.2f}/10  =  Accuracy {r.acc:.2f}/4 (blue)  +  "
                     f"Expression {r.expr:.2f}/6 (green)", fontsize=10)

    # ---------- phát lại ----------
    def _start_replay(self) -> None:
        if not self.frames:
            return
        T = min(len(self.frames), self.kpts.shape[0])
        seq = list(range(T))
        tp = self.res.phases.get("t_peak") if self.res.valid else None
        if tp is not None and 0 <= tp < T:  # dừng lại ở đỉnh động tác cho dễ nhìn
            seq = seq[:tp] + [tp] * 12 + seq[tp + 1:]
        seq += [T - 1] * 15
        hl = self.res.highlight

        def update(t):
            img = self.frames[t].copy()
            _draw_skeleton(img, self.kpts[t], self.scale, hl)
            ph = self._phase(t)
            self._show_image(img, f"Replay (2× slower) — frame {t}" + (f"  ·  {en(ph)}" if ph else ""))
            if self.cursor is not None:
                self.cursor.set_xdata([t, t])
            return []

        update(tp if tp is not None else 0)
        if self.interactive:
            self.anim = FuncAnimation(self.fig, update, frames=seq, interval=2000.0 / max(self.fps, 1.0),
                                      repeat=True, cache_frame_data=False)

    # ---------- tiện ích ----------
    def _refresh(self) -> None:
        if not self.interactive or self.closed:
            return
        if not plt.fignum_exists(self.fig.number):
            self.closed = True  # người dùng đóng cửa sổ giữa chừng -> code vẫn chạy tiếp
            return
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save(self, path: str) -> None:
        self.fig.savefig(path, dpi=110)

    def wait_close(self) -> None:
        """Giữ cửa sổ mở (video phát lặp) tới khi người dùng đóng."""
        if self.interactive and not self.closed and plt.fignum_exists(self.fig.number):
            plt.ioff()
            plt.show()
        plt.close(self.fig)
