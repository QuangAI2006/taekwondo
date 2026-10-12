"""Folder review and sequential batch scoring UI; launched by chamdiembatch.bat."""
from __future__ import annotations

import json
import os
import queue
import threading
import tkinter as tk
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from batch_scoring import BatchRunner, ROOT, scan_folder, validate_items
from score_app import load_actions


def score_text(value):
    return f"{value:.2f}" if isinstance(value, (int, float)) else "—"


class BatchResultsWindow(tk.Toplevel):
    """Display a snapshot of every video and its complete failure reason."""

    def __init__(self, parent, rows, output, open_path):
        super().__init__(parent)
        self.rows = [dict(row) for row in rows]
        self.title("Kết quả chấm điểm — Tất cả video trong batch")
        self.geometry("1200x680")
        self.minsize(850, 500)
        counts = Counter(row["status"] for row in rows)
        summary = "; ".join(f"{status}: {count}" for status, count in counts.items())
        ttk.Label(self, text=f"Tổng {len(rows)} video — {summary}", wraplength=1150,
                  font=("Segoe UI", 11, "bold")).pack(fill="x", padx=10, pady=8)
        table = ttk.Frame(self)
        table.pack(fill="both", expand=True, padx=10)
        cols = ("video", "action", "status", "total", "accuracy", "expression", "reason")
        self.tree = ttk.Treeview(table, columns=cols, show="headings", selectmode="browse")
        for column, title, width in zip(cols,
                ["Video", "Động tác", "Trạng thái", "Tổng / 10", "Chính xác / 4", "Biểu hiện / 6", "Lý do / cảnh báo"],
                [290, 250, 180, 85, 105, 105, 460]):
            self.tree.heading(column, text=title)
            self.tree.column(column, width=width, minwidth=70, stretch=False)
        self.tree.tag_configure("error", foreground="#b71c1c")
        self.tree.tag_configure("success", foreground="#2e7d32")
        self.tree.tag_configure("pending", foreground="#777777")
        vertical = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        horizontal = ttk.Scrollbar(table, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        for index, row in enumerate(self.rows):
            status = row["status"]
            tag = "success" if status == "Đã chấm" else "error" if status in {"Lỗi", "Không chấm được", "Chưa chấm (batch lỗi)"} else "pending"
            self.tree.insert("", "end", iid=str(index), values=(Path(row["video"]).name,
                             row["action_label"] or row["action"] or "Chưa chọn", status,
                             score_text(row["total10"]), score_text(row["acc4"]), score_text(row["expr6"]),
                             " ".join(row["reason"].splitlines())), tags=(tag,))
        ttk.Label(self, text="Chọn dòng để đọc đầy đủ lý do và đường dẫn video.").pack(anchor="w", padx=10, pady=4)
        details = ttk.Frame(self)
        details.pack(fill="both", padx=10)
        self.detail = tk.Text(details, wrap="word", height=8, state="disabled")
        scroll = ttk.Scrollbar(details, command=self.detail.yview)
        self.detail.configure(yscrollcommand=scroll.set)
        self.detail.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=10, pady=8)
        self.open_path = open_path
        ttk.Button(buttons, text="Mở kết quả video đang chọn", command=self._open_result).pack(side="left", padx=4)
        ttk.Button(buttons, text="Mở bảng điểm CSV", command=lambda: open_path(output / "summary.csv"),
                   state="normal" if (output / "summary.csv").is_file() else "disabled").pack(side="left", padx=4)
        ttk.Button(buttons, text="Đóng bảng tổng kết", command=self.destroy).pack(side="right", padx=4)
        self.tree.bind("<<TreeviewSelect>>", self._select_result)
        if self.rows:
            index = next((i for i, row in enumerate(self.rows) if row["status"] in {"Lỗi", "Không chấm được", "Chưa chấm (batch lỗi)"}), 0)
            self.tree.selection_set(str(index))
            self.tree.see(str(index))
            self._select_result()

    def _select_result(self, _event=None):
        selection = self.tree.selection()
        if not selection:
            return
        row = self.rows[int(selection[0])]
        text = (f"Video: {row['video']}\nĐộng tác: {row['action_label'] or row['action']}\n"
                f"Trạng thái: {row['status']}\nTổng: {score_text(row['total10'])} / 10; "
                f"Chính xác: {score_text(row['acc4'])} / 4; Biểu hiện: {score_text(row['expr6'])} / 6\n"
                f"Lý do / cảnh báo: {row['reason'] or 'Không có.'}\nKết quả: {row['result_dir'] or 'Chưa có.'}")
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("end", text)
        self.detail.configure(state="disabled")

    def _open_result(self):
        selection = self.tree.selection()
        if selection:
            path = self.rows[int(selection[0])]["result_dir"]
            if path:
                self.open_path(Path(path))


class BatchApp(tk.Tk):
    def __init__(self, actions):
        super().__init__()
        self.title("Chấm điểm Taekwondo — Batch video")
        self.geometry("1280x850")
        self.minsize(1000, 700)
        self.actions = actions
        self.labels = {name: label for _, name, label in actions}
        self.choices = {f"{label} [{name}]": name for _, name, label in actions}
        self.items = []
        self.outputs = {}
        self.results = {}
        self.report_rows = []
        self.report_output = None
        self._summary_window = None
        self.runner = None
        self.running = False
        self.events = queue.Queue()
        self.completed = 0
        self.folder = tk.StringVar()
        self.recursive = tk.BooleanVar(value=False)
        self.pipeline = tk.BooleanVar(value=True)
        self.preview = tk.StringVar(value="3")
        self.action = tk.StringVar()
        self.side = tk.StringVar(value="Tự nhận")
        self.guard = tk.StringVar(value="chin")
        self.fix_leg = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Chọn thư mục video để lập danh sách kiểm tra.")
        self.editor_buttons = []
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(100, self._poll)

    def _build(self):
        folder_box = ttk.LabelFrame(self, text="1. Chọn thư mục video")
        folder_box.pack(fill="x", padx=10, pady=6)
        ttk.Entry(folder_box, textvariable=self.folder, state="readonly").pack(
            side="left", fill="x", expand=True, padx=6, pady=6)
        self.btn_folder = ttk.Button(folder_box, text="Chọn thư mục…", command=self._pick_folder)
        self.btn_folder.pack(side="left", padx=6)
        self.check_recursive = ttk.Checkbutton(folder_box, text="Gồm thư mục con", variable=self.recursive)
        self.check_recursive.pack(side="left", padx=6)
        self.btn_scan = ttk.Button(folder_box, text="Quét lại", command=self._rescan)
        self.btn_scan.pack(side="left", padx=6)

        review = ttk.LabelFrame(self, text="2. Kiểm tra động tác trước khi chấm")
        review.pack(fill="both", expand=True, padx=10, pady=4)
        ttk.Label(review, text="Chọn một hoặc nhiều dòng để sửa/xác nhận. Video chưa rõ động tác cần chọn thủ công.").pack(
            anchor="w", padx=6, pady=4)
        table = ttk.Frame(review)
        table.pack(fill="both", expand=True)
        cols = ("include", "video", "action", "review", "status", "score", "reason")
        self.tree = ttk.Treeview(table, columns=cols, show="headings", selectmode="extended", height=12)
        for name, title, width in zip(cols, ["Chấm", "Video", "Động tác", "Kiểm tra", "Trạng thái", "Điểm / 10", "Lý do / cảnh báo"],
                                       [55, 330, 320, 115, 170, 85, 420]):
            self.tree.heading(name, text=title)
            self.tree.column(name, width=width, minwidth=45, stretch=name in {"video", "action"})
        self.tree.tag_configure("unreviewed", foreground="#a65c00")
        self.tree.tag_configure("error", foreground="#b71c1c")
        self.tree.tag_configure("skip", foreground="#777777")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        horizontal = ttk.Scrollbar(table, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=scroll.set, xscrollcommand=horizontal.set)
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<<TreeviewSelect>>", self._select)

        editor = ttk.Frame(review)
        editor.pack(fill="x", padx=6, pady=4)
        ttk.Label(editor, text="Động tác:").pack(side="left")
        ttk.Combobox(editor, textvariable=self.action, values=list(self.choices), state="readonly",
                     width=65).pack(side="left", padx=4)
        button = ttk.Button(editor, text="Áp dụng và xác nhận", command=self._apply)
        button.pack(side="left", padx=6)
        self.editor_buttons.append(button)
        options = ttk.Frame(review)
        options.pack(fill="x", padx=6, pady=4)
        ttk.Label(options, text="Tay/bên ra đòn:").pack(side="left")
        ttk.Combobox(options, textvariable=self.side, values=["Tự nhận", "Trái (L)", "Phải (R)"],
                     state="readonly", width=12).pack(side="left", padx=4)
        ttk.Label(options, text="Kiểu thủ:").pack(side="left")
        ttk.Combobox(options, textvariable=self.guard, values=["chin", "waist"],
                     state="readonly", width=7).pack(side="left", padx=4)
        ttk.Checkbutton(options, text="Tự chọn đúng chân đá", variable=self.fix_leg).pack(side="left", padx=6)
        ttk.Label(options, text="Tùy chọn chỉ áp dụng với động tác phù hợp.").pack(side="left", padx=6)
        buttons = ttk.Frame(review)
        buttons.pack(fill="x", padx=6, pady=4)
        for text, command in [("Xác nhận dòng chọn", self._confirm_selected),
                              ("Xác nhận toàn bộ lựa chọn", self._confirm_all),
                              ("Chấm / bỏ qua dòng chọn", self._toggle_included)]:
            button = ttk.Button(buttons, text=text, command=command)
            button.pack(side="left", padx=3)
            self.editor_buttons.append(button)
        ttk.Button(buttons, text="Mở video xem trước", command=self._preview_video).pack(side="left", padx=3)

        run = ttk.LabelFrame(self, text="3. Chấm lần lượt")
        run.pack(fill="x", padx=10, pady=4)
        ttk.Checkbutton(run, text="Mở pipeline từng video", variable=self.pipeline).pack(side="left", padx=6)
        ttk.Label(run, text="Giữ kết quả pipeline (giây):").pack(side="left")
        ttk.Spinbox(run, from_=0, to=60, textvariable=self.preview, width=5).pack(side="left", padx=4)
        self.btn_run = ttk.Button(run, text="▶ Bắt đầu batch", command=self._start)
        self.btn_run.pack(side="left", padx=6, pady=6)
        self.btn_stop = ttk.Button(run, text="Dừng sau video hiện tại", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        ttk.Button(run, text="Xem bảng tổng kết", command=self._show_summary).pack(side="left", padx=6)
        ttk.Button(run, text="Mở kết quả batch", command=self._open_batch).pack(side="left", padx=6)
        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=4)
        ttk.Label(self, textvariable=self.status, foreground="#1565c0", wraplength=1200).pack(
            fill="x", padx=10, pady=4)

        details = ttk.LabelFrame(self, text="Chi tiết / log của video đang chọn")
        details.pack(fill="both", expand=True, padx=10, pady=6)
        self.detail = tk.Text(details, height=9, wrap="word", state="disabled")
        log_scroll = ttk.Scrollbar(details, command=self.detail.yview)
        self.detail.configure(yscrollcommand=log_scroll.set)
        self.detail.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="left", fill="y")
        ttk.Button(details, text="Mở kết quả video", command=self._open_video_result).pack(side="left", padx=6)

    def _pick_folder(self):
        if self.running:
            return
        folder = filedialog.askdirectory(title="Chọn thư mục chứa video")
        if folder:
            self.folder.set(folder)
            self._rescan()

    def _rescan(self):
        if self.running or not self.folder.get():
            return
        from action_scorers import ALIASES
        try:
            items = scan_folder(Path(self.folder.get()), self.actions, ALIASES, self.recursive.get())
        except OSError as exc:
            messagebox.showerror("Không đọc được thư mục", str(exc))
            return
        self.items = items
        self.outputs = {}
        self.results = {}
        self.tree.delete(*self.tree.get_children())
        for index in range(len(items)):
            self.tree.insert("", "end", iid=str(index))
            self._refresh(index)
        self.status.set(f"Có {len(items)} video. Kiểm tra và xác nhận động tác trước khi bắt đầu.")
        self.progress["value"] = 0
        self._set_detail("")

    def _refresh(self, index, status=None, score="", reason=""):
        item = self.items[index]
        old = self.tree.item(str(index), "values")
        state = status if status is not None else (old[4] if old else "Chờ chấm")
        if not item.included:
            state = "Bỏ qua"
        tag = "skip" if not item.included else ("error" if state in {"Lỗi", "Không chấm được"}
                                               else "unreviewed" if not item.reviewed else "")
        try:
            name = str(Path(item.video).relative_to(Path(self.folder.get())))
        except ValueError:
            name = Path(item.video).name
        self.tree.item(str(index), values=("✓" if item.included else "—", name,
                       self.labels.get(item.action, "Chưa nhận diện"),
                       "Đã xác nhận" if item.reviewed else "Chưa xác nhận", state, score,
                       " ".join(reason.splitlines())),
                       tags=(tag,) if tag else ())

    def _indices(self):
        return [int(iid) for iid in self.tree.selection()]

    def _select(self, _event=None):
        selected = self._indices()
        if not selected:
            return
        index = selected[0]
        item = self.items[index]
        self.action.set(next((choice for choice, name in self.choices.items() if name == item.action), ""))
        self.side.set({"L": "Trái (L)", "R": "Phải (R)"}.get(item.side, "Tự nhận"))
        self.guard.set(item.guard)
        self.fix_leg.set(item.fix_leg)
        self._show_details(index)

    def _apply(self):
        if self.running:
            return
        selected = self._indices()
        action = self.choices.get(self.action.get())
        if not selected or not action:
            messagebox.showwarning("Chưa chọn", "Chọn dòng video và động tác cần áp dụng.")
            return
        for index in selected:
            item = self.items[index]
            item.action = action
            item.side = {"Trái (L)": "L", "Phải (R)": "R"}.get(self.side.get(), "")
            item.guard = self.guard.get()
            item.fix_leg = self.fix_leg.get()
            item.reviewed = True
            self._refresh(index)

    def _confirm(self, indices):
        if self.running:
            return
        missing = 0
        for index in indices:
            item = self.items[index]
            if not item.included:
                continue
            if item.action not in self.labels:
                missing += 1
                continue
            item.reviewed = True
            self._refresh(index)
        if missing:
            messagebox.showwarning("Cần chọn động tác", f"Còn {missing} video chưa nhận diện. Hãy chọn động tác hoặc bỏ qua.")

    def _confirm_selected(self):
        self._confirm(self._indices())

    def _confirm_all(self):
        self._confirm(range(len(self.items)))

    def _toggle_included(self):
        if not self.running:
            for index in self._indices():
                self.items[index].included = not self.items[index].included
                self._refresh(index, "Chờ chấm")

    def _preview_video(self):
        indices = self._indices()
        if indices:
            self._open_path(Path(self.items[indices[0]].video))

    def _open_path(self, path):
        try:
            if not path.exists():
                raise FileNotFoundError(str(path))
            os.startfile(str(path))
        except OSError as exc:
            messagebox.showerror("Không mở được", str(exc))

    def _open_video_result(self):
        indices = self._indices()
        if indices and indices[0] in self.outputs:
            self._open_path(self.outputs[indices[0]])

    def _open_batch(self):
        if self.runner and self.runner.output.exists():
            self._open_path(self.runner.output)

    def _show_summary(self):
        if not self.report_rows:
            return
        if self._summary_window and self._summary_window.winfo_exists():
            self._summary_window.lift()
            return
        self._summary_window = BatchResultsWindow(self, self.report_rows, self.report_output, self._open_path)

    def _set_detail(self, text):
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("end", text)
        self.detail.configure(state="disabled")

    def _show_details(self, index):
        item = self.items[index]
        text = f"{item.video}\nĐộng tác: {self.labels.get(item.action, 'Chưa chọn')}\n"
        row = self.results.get(index)
        if row:
            text += f"Trạng thái: {row['status']}\nĐiểm / 10: {row['total10']}\n"
            if row["reason"]:
                text += f"Lý do / cảnh báo:\n{row['reason']}\n"
        out = self.outputs.get(index)
        if out:
            try:
                result_file = out / "result.json"
                if result_file.is_file():
                    text += "\n" + json.dumps(json.loads(result_file.read_text(encoding="utf-8")),
                                              ensure_ascii=False, indent=2) + "\n"
                log = out / "log.txt"
                if log.is_file():
                    text += "\nLog:\n" + log.read_text(encoding="utf-8", errors="replace")[-6000:]
            except (OSError, ValueError) as exc:
                text += f"\nLỗi đọc kết quả: {exc}"
        self._set_detail(text)

    def _set_running(self, running):
        self.running = running
        for widget in [self.btn_folder, self.btn_scan, self.check_recursive, self.btn_run, *self.editor_buttons]:
            widget.configure(state="disabled" if running else "normal")
        self.btn_stop.configure(state="normal" if running else "disabled")

    def _start(self):
        if self.running:
            return
        errors = validate_items(self.items, self.labels)
        if errors:
            messagebox.showwarning("Chưa sẵn sàng", "\n".join(errors[:12]) +
                                   (f"\n… và {len(errors) - 12} vấn đề khác." if len(errors) > 12 else ""))
            return
        try:
            seconds = float(self.preview.get())
            if not 0 <= seconds <= 60:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Thời gian chưa đúng", "Nhập số giây từ 0 đến 60.")
            return
        output = ROOT / "outputs" / "batch_scores" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        self.runner = BatchRunner(self.items, self.labels, output,
                                  lambda *event: self.events.put(event), self.pipeline.get(), seconds)
        self.outputs = {}
        self.results = {}
        self.completed = 0
        self.progress.configure(maximum=sum(item.included for item in self.items), value=0)
        for index in range(len(self.items)):
            self._refresh(index, "Chờ chấm")
        self._set_running(True)
        self.status.set("Đang bắt đầu chấm lần lượt…")
        threading.Thread(target=self.runner.run, daemon=True).start()

    def _stop(self):
        if self.runner and self.running:
            self.runner.stop.set()
            self.btn_stop.configure(state="disabled")
            self.status.set("Sẽ dừng sau khi video hiện tại hoàn tất; các video còn lại chưa chấm.")

    def _poll(self):
        for _ in range(250):
            try:
                event, index, data = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "start":
                self.outputs[index] = Path(data)
                self._refresh(index, "Đang chấm")
                self.tree.see(str(index))
                self.status.set(f"Đang chấm: {Path(self.items[index].video).name} — {self.labels[self.items[index].action]}")
            elif event == "log":
                if self._indices()[:1] == [index]:
                    self.detail.configure(state="normal")
                    self.detail.insert("end", data)
                    if int(self.detail.index("end-1c").split(".")[0]) > 250:
                        self.detail.delete("1.0", "100.0")
                    self.detail.see("end")
                    self.detail.configure(state="disabled")
            elif event == "result":
                score = data["total10"]
                self.results[index] = dict(data)
                self._refresh(index, data["status"], f"{score:.2f}" if isinstance(score, (int, float)) else "", data["reason"])
                if data["status"] != "Chưa chấm (đã dừng)":
                    self.completed += 1
                    self.progress["value"] = self.completed
                if self._indices()[:1] == [index]:
                    self._show_details(index)
            elif event == "fatal":
                messagebox.showerror("Lỗi batch", data)
            elif event == "done":
                self._set_running(False)
                counts = Counter(row["status"] for row in self.runner.rows)
                summary = "; ".join(f"{status}: {count}" for status, count in counts.items())
                self.status.set(f"Kết thúc — {summary}. Kết quả: {data}")
                self.report_rows = [dict(row) for row in self.runner.rows]
                self.report_output = Path(data)
                if self._summary_window and self._summary_window.winfo_exists():
                    self._summary_window.destroy()
                self._show_summary()
        self.after(100, self._poll)

    def _close(self):
        if self.running:
            messagebox.showinfo("Batch đang chạy", "Dùng 'Dừng sau video hiện tại', đợi hoàn tất rồi đóng cửa sổ.")
            return
        self.destroy()


if __name__ == "__main__":
    BatchApp(load_actions()).mainloop()
