# scripts/score_app.py
"""
Ứng dụng chấm điểm: CHỌN động tác -> CHỌN video -> chấm theo luật của động tác đó (không dùng model nhận dạng).

  - Danh sách động tác lấy tự động từ action_scorers.py + scripts/scorers_*.py (thêm luật mới là có trong danh sách).
  - Mỗi lần chấm gọi infer_action.py --no_model --action <động tác> trong tiến trình riêng, nên giao diện không bị treo
    và cửa sổ pipeline (nếu bật) chạy độc lập.
  - Kết quả lưu ở outputs/app_scores/<video>__<động tác>__<thời gian>/ (result.json, keyframes.png, log.txt)
    và cộng dồn vào outputs/app_scores/history.csv.

Chạy:
  python code/scripts/score_app.py
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
PKG_ROOT = SCRIPTS_DIR.parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

INFER = SCRIPTS_DIR / "infer_action.py"
OUT_ROOT = PKG_ROOT / "outputs" / "app_scores"
VIDEO_TYPES = [("Video", "*.mp4 *.mov *.avi *.mkv"), ("Tất cả", "*.*")]

# nhóm hiển thị trong danh sách; động tác chưa có trong bảng này vào nhóm "Khác"
GROUPS: List[Tuple[str, List[str]]] = [
    ("Đá", ["front", "roundhouse", "axe", "side", "back", "hook_kick", "push", "side_low", "roundhouse_low"]),
    ("Đấm / tay", ["straight", "hook", "high_block", "low_block", "knife_block", "knife_strike", "spear_hand"]),
    ("Tấn / thủ thế", ["horse_stance", "front_stance", "duck", "knee_chamber"]),
    ("Phối hợp / bộ pháp", ["block_punch", "block_kick", "horse_punch", "footwork"]),
]
# tùy chọn nào có tác dụng với động tác nào (theo chữ ký các hàm chấm)
SIDE_ACTIONS = {"straight", "hook", "high_block", "low_block", "knife_block", "knife_strike", "spear_hand",
                "block_punch", "block_kick", "horse_punch"}
GUARD_ACTIONS = {"straight", "hook"}
KICK_ACTIONS = set(GROUPS[0][1]) | {"block_kick"}


def load_actions() -> List[Tuple[str, str, str]]:
    """-> [(nhóm, tên luật, nhãn tiếng Việt)] theo thứ tự hiển thị."""
    from infer_action import load_plugins
    load_plugins()
    from action_scorers import SCORERS
    out, seen = [], set()
    for g, names in GROUPS:
        for n in names:
            if n in SCORERS:
                out.append((g, n, SCORERS[n].label_vi))
                seen.add(n)
    out += [("Khác", n, s.label_vi) for n, s in SCORERS.items() if n not in seen]
    return out


def _fmt(x: Optional[float], nd: int = 2) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def append_history(video: Path, action: str, label: str, res: Dict, out_dir: Path) -> None:
    path = OUT_ROOT / "history.csv"
    new = not path.is_file()
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["time", "video", "action", "action_label", "valid", "total10", "acc4", "expr6",
                         "reason_or_warnings", "result_dir"])
        wr.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), str(video), action, label, int(bool(res.get("valid"))),
                     _fmt(res.get("total")), _fmt(res.get("acc")), _fmt(res.get("expr")),
                     " | ".join(res.get("warnings") or []) if res.get("valid") else res.get("reason", ""),
                     str(out_dir)])


class App(tk.Tk):
    def __init__(self, actions: List[Tuple[str, str, str]]):
        super().__init__()
        self.title("Chấm điểm động tác Taekwondo")
        self.geometry("1100x760")
        self.minsize(900, 600)
        self.actions = actions
        self.proc: Optional[subprocess.Popen] = None
        self.last_dir: Optional[Path] = None
        self._img = None  # giữ tham chiếu ảnh cho Tk

        self.var_action = tk.StringVar()
        self.var_video = tk.StringVar()
        self.var_side = tk.StringVar(value="Tự nhận")
        self.var_guard = tk.StringVar(value="chin")
        self.var_fixleg = tk.BooleanVar(value=False)
        self.var_window = tk.BooleanVar(value=False)
        self.var_status = tk.StringVar(value=f"Có {len(actions)} động tác có luật chấm.")
        self._build()

    # ---------- giao diện ----------
    def _build(self) -> None:
        pad = dict(padx=8, pady=4)
        top = ttk.Frame(self)
        top.pack(fill="x", **pad)

        # 1. chọn động tác: cây theo nhóm
        left = ttk.LabelFrame(top, text="1. Chọn động tác")
        left.pack(side="left", fill="y", padx=(0, 8))
        self.tree = ttk.Treeview(left, show="tree", height=14, selectmode="browse")
        self.tree.column("#0", width=330)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="y")
        sb.pack(side="left", fill="y")
        groups: Dict[str, str] = {}
        for g, name, label in self.actions:
            if g not in groups:
                groups[g] = self.tree.insert("", "end", text=g, open=True)
            self.tree.insert(groups[g], "end", iid=f"a:{name}", text=f"{label}   [{name}]")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        right = ttk.Frame(top)
        right.pack(side="left", fill="both", expand=True)

        box = ttk.LabelFrame(right, text="2. Chọn video")
        box.pack(fill="x")
        ttk.Entry(box, textvariable=self.var_video).pack(side="left", fill="x", expand=True, padx=6, pady=6)
        ttk.Button(box, text="Chọn video…", command=self._pick_video).pack(side="left", padx=6)

        opt = ttk.LabelFrame(right, text="Tùy chọn (chỉ hiện với động tác dùng đến)")
        opt.pack(fill="x", pady=6)
        self.fr_side = ttk.Frame(opt)
        ttk.Label(self.fr_side, text="Tay/bên ra đòn:").pack(side="left")
        ttk.Combobox(self.fr_side, textvariable=self.var_side, values=["Tự nhận", "Trái (L)", "Phải (R)"],
                     state="readonly", width=10).pack(side="left", padx=4)
        self.fr_guard = ttk.Frame(opt)
        ttk.Label(self.fr_guard, text="Kiểu thủ:").pack(side="left")
        ttk.Combobox(self.fr_guard, textvariable=self.var_guard, values=["chin", "waist"],
                     state="readonly", width=8).pack(side="left", padx=4)
        ttk.Label(self.fr_guard, text="(chin = tay ở cằm, waist = tay ở hông)").pack(side="left")
        self.fr_kick = ttk.Frame(opt)
        ttk.Checkbutton(self.fr_kick, text="Tự chọn đúng chân đá (--fix_leg)", variable=self.var_fixleg).pack(side="left")
        self.lbl_noopt = ttk.Label(opt, text="Chọn động tác để xem tùy chọn.", foreground="#777")
        self.lbl_noopt.pack(anchor="w", padx=6, pady=4)

        run = ttk.LabelFrame(right, text="3. Chấm điểm")
        run.pack(fill="x")
        ttk.Checkbutton(run, text="Mở cửa sổ pipeline (xem khung xương, các pha)",
                        variable=self.var_window).pack(side="left", padx=6)
        self.btn_run = ttk.Button(run, text="▶  Chấm điểm", command=self._run)
        self.btn_run.pack(side="left", padx=6, pady=6)
        self.btn_open = ttk.Button(run, text="Mở thư mục kết quả", command=self._open_dir, state="disabled")
        self.btn_open.pack(side="left", padx=6)
        self.pb = ttk.Progressbar(run, mode="indeterminate", length=160)
        self.pb.pack(side="left", padx=6)

        ttk.Label(self, textvariable=self.var_status, foreground="#1565c0").pack(fill="x", padx=10)

        # kết quả: chữ bên trái, ảnh 3 frame mốc bên dưới
        res = ttk.LabelFrame(self, text="Kết quả")
        res.pack(fill="both", expand=True, **pad)
        self.txt = tk.Text(res, wrap="word", font=("Consolas", 10), height=16)
        sb2 = ttk.Scrollbar(res, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb2.set)
        self.txt.pack(side="top", fill="both", expand=True)
        self.txt.tag_configure("big", font=("Segoe UI", 16, "bold"), foreground="#2e7d32")
        self.txt.tag_configure("bad", font=("Segoe UI", 12, "bold"), foreground="#c62828")
        self.txt.tag_configure("h", font=("Segoe UI", 10, "bold"))
        self.img_lbl = ttk.Label(res)
        self.img_lbl.pack(side="top", pady=4)

    def _selected_action(self) -> Optional[str]:
        sel = self.tree.selection()
        return sel[0][2:] if sel and sel[0].startswith("a:") else None

    def _on_select(self, _=None) -> None:
        a = self._selected_action()
        for fr in (self.fr_side, self.fr_guard, self.fr_kick):
            fr.pack_forget()
        self.lbl_noopt.pack_forget()
        if a is None:
            self.lbl_noopt.pack(anchor="w", padx=6, pady=4)
            return
        shown = False
        for fr, names in ((self.fr_side, SIDE_ACTIONS), (self.fr_guard, GUARD_ACTIONS), (self.fr_kick, KICK_ACTIONS)):
            if a in names:
                fr.pack(anchor="w", padx=6, pady=2)
                shown = True
        if not shown:
            self.lbl_noopt.configure(text="Động tác này không có tùy chọn thêm.")
            self.lbl_noopt.pack(anchor="w", padx=6, pady=4)

    def _pick_video(self) -> None:
        p = filedialog.askopenfilename(title="Chọn video động tác", filetypes=VIDEO_TYPES)
        if p:
            self.var_video.set(p)

    def _open_dir(self) -> None:
        if self.last_dir and self.last_dir.is_dir():
            os.startfile(str(self.last_dir))

    # ---------- chạy ----------
    def _run(self) -> None:
        action = self._selected_action()
        video = Path(self.var_video.get().strip().strip('"'))
        if action is None:
            messagebox.showwarning("Thiếu động tác", "Hãy chọn một động tác trong danh sách.")
            return
        if not video.is_file():
            messagebox.showwarning("Thiếu video", "Hãy chọn file video.")
            return

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = OUT_ROOT / f"{video.stem[:50]}__{action}__{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(INFER), str(video), "--no_model", "--action", action,
               "--out", str(out_dir / "result.json"), "--keyframes", str(out_dir / "keyframes.png"),
               "--window", "on" if self.var_window.get() else "off"]
        if action in SIDE_ACTIONS and self.var_side.get() != "Tự nhận":
            cmd += ["--side", "L" if self.var_side.get().startswith("Trái") else "R"]
        if action in GUARD_ACTIONS:
            cmd += ["--guard", self.var_guard.get()]
        if action in KICK_ACTIONS and self.var_fixleg.get():
            cmd += ["--fix_leg"]

        label = next(lbl for _, n, lbl in self.actions if n == action)
        self.last_dir = out_dir
        self.btn_run.configure(state="disabled")
        self.btn_open.configure(state="normal")
        self.pb.start(12)
        self.var_status.set(f"Đang chấm '{label}' cho {video.name} … (MediaPipe có thể mất vài chục giây)")
        self.txt.delete("1.0", "end")
        self.img_lbl.configure(image="")
        threading.Thread(target=self._worker, args=(cmd, video, action, label, out_dir), daemon=True).start()

    def _worker(self, cmd, video: Path, action: str, label: str, out_dir: Path) -> None:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        log_lines: List[str] = []
        shown = False
        try:
            self.proc = subprocess.Popen(cmd, cwd=str(PKG_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, encoding="utf-8", errors="replace", env=env,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in self.proc.stdout:
                log_lines.append(line)
                # JSON được ghi trước khi cửa sổ pipeline chờ đóng -> hiện kết quả ngay
                if not shown and "[SAVED]" in line and line.strip().endswith("result.json"):
                    shown = True
                    self.after(0, self._show, video, action, label, out_dir, "".join(log_lines), True)
            self.proc.wait()
        except Exception as e:  # noqa: BLE001
            log_lines.append(f"\n[LỖI] {e}\n")
        (out_dir / "log.txt").write_text("".join(log_lines), encoding="utf-8")
        if not shown:
            self.after(0, self._show, video, action, label, out_dir, "".join(log_lines), False)
        self.after(0, self._done)

    def _done(self) -> None:
        self.pb.stop()
        self.btn_run.configure(state="normal")

    def _show(self, video: Path, action: str, label: str, out_dir: Path, log: str, ok: bool) -> None:
        t = self.txt
        t.delete("1.0", "end")
        jp = out_dir / "result.json"
        if not ok or not jp.is_file():
            t.insert("end", "Không chạy được.\n\n", "bad")
            t.insert("end", log[-4000:])
            self.var_status.set("Lỗi — xem log bên dưới.")
            return
        r = json.loads(jp.read_text(encoding="utf-8"))["result"]
        append_history(video, action, label, r, out_dir)
        t.insert("end", f"{video.name}\nChấm theo: {label} [{action}]"
                        + (f" — {r['side_text']}" if r.get("side_text") else "") + "\n\n", "h")
        if not r.get("valid"):
            t.insert("end", f"KHÔNG CHẤM ĐƯỢC: {r.get('reason', '')}\n", "bad")
            self.var_status.set("Không chấm được — xem lý do.")
        else:
            t.insert("end", f"{r['total']:.2f} / 10\n", "big")
            t.insert("end", f"Chính xác {r['acc']:.2f} / 4   +   Biểu hiện {r['expr']:.2f} / 6\n\n")
            for title, items in (("ĐỘ CHÍNH XÁC", r.get("acc_items", [])), ("BIỂU HIỆN", r.get("expr_items", []))):
                t.insert("end", title + "\n", "h")
                for it in items:
                    got = "  — " if it.get("got") is None else f"{it['got']:.2f}"
                    t.insert("end", f"  [{it['code']}] {it['label']:<44s} {got} / {it['maxi']:.1f}   "
                                    f"{it.get('reason', '')}\n")
                t.insert("end", "\n")
            if r.get("top_issues"):
                t.insert("end", "CẦN CẢI THIỆN\n", "h")
                for it in r["top_issues"][:4]:
                    t.insert("end", f"  • {it['label']} (−{it['deduct']:.2f}): {it['tip']}\n")
                t.insert("end", "\n")
            self.var_status.set(f"Xong: {r['total']:.2f}/10 — đã lưu vào {out_dir.name}")
        for w in r.get("warnings") or []:
            t.insert("end", f"[!] {w}\n", "bad")
        self._show_image(out_dir / "keyframes.png")

    def _show_image(self, p: Path) -> None:
        if not p.is_file():
            return
        try:
            from PIL import Image, ImageTk
            im = Image.open(p)
            w = max(400, self.winfo_width() - 60)
            sc = min(w / im.width, 240 / im.height, 1.0)
            im = im.resize((int(im.width * sc), int(im.height * sc)))
            self._img = ImageTk.PhotoImage(im)
            self.img_lbl.configure(image=self._img)
        except Exception:  # noqa: BLE001  (thiếu Pillow thì chỉ không hiện ảnh)
            pass


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    actions = load_actions()
    App(actions).mainloop()


if __name__ == "__main__":
    main()
