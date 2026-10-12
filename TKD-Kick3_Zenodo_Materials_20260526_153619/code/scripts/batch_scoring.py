"""Sequential batch scoring, independent of the single-video application."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

from filename_action import detect_filename_action

ROOT = Path(__file__).resolve().parents[2]
INFER = Path(__file__).with_name("infer_batch_action.py")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
SIDE_ACTIONS = {"straight", "hook", "high_block", "low_block", "knife_block", "knife_strike",
                "spear_hand", "block_punch", "block_kick", "horse_punch"}
KICK_ACTIONS = {"front", "roundhouse", "axe", "side", "back", "hook_kick", "push",
                "side_low", "roundhouse_low", "block_kick"}


@dataclass
class BatchItem:
    video: str
    action: str = ""
    included: bool = True
    reviewed: bool = False
    side: str = ""
    guard: str = "chin"
    fix_leg: bool = False


def scan_folder(folder: Path, actions, aliases, recursive=False) -> list[BatchItem]:
    paths = folder.rglob("*") if recursive else folder.iterdir()
    videos = sorted((p for p in paths if p.is_file() and p.suffix.lower() in VIDEO_EXTS),
                    key=lambda p: str(p.relative_to(folder)).casefold())
    return [BatchItem(str(p.resolve()), detect_filename_action(p.name, actions, aliases) or "")
            for p in videos]


def validate_items(items, allowed_actions) -> list[str]:
    errors = []
    selected = [item for item in items if item.included]
    if not selected:
        errors.append("Chưa có video nào được chọn để chấm.")
    for item in selected:
        name = Path(item.video).name
        if item.action not in allowed_actions:
            errors.append(f"{name}: chưa chọn động tác hợp lệ.")
        elif not item.reviewed:
            errors.append(f"{name}: chưa xác nhận động tác.")
        if not Path(item.video).is_file():
            errors.append(f"{name}: không tìm thấy video.")
    return errors


def build_command(item: BatchItem, out: Path, pipeline: bool, preview_seconds: float) -> list[str]:
    cmd = [sys.executable, "-u", str(INFER), item.video, "--no_model", "--action", item.action,
           "--out", str(out / "result.json"), "--keyframes", str(out / "keyframes.png"),
           "--window", "on" if pipeline else "off", "--batch-preview-seconds", str(preview_seconds)]
    if item.action in SIDE_ACTIONS and item.side:
        cmd += ["--side", item.side]
    if item.action in {"straight", "hook"}:
        cmd += ["--guard", item.guard]
    if item.action in KICK_ACTIONS and item.fix_leg:
        cmd += ["--fix_leg"]
    return cmd


class BatchRunner:
    """Own the subprocess in a worker thread; communicate via a UI event queue."""

    def __init__(self, items, labels, output: Path, emit: Callable, pipeline=False,
                 preview_seconds=3.0, execute=None):
        self.items = [replace(item) for item in items]
        self.labels = dict(labels)
        self.output = output
        self.emit = emit
        self.pipeline = pipeline
        self.preview_seconds = preview_seconds
        self.stop = threading.Event()  # Stop after the current video finishes.
        self.execute = execute or self._execute
        self.rows = [dict(index=i, video=item.video, action=item.action,
                          action_label=self.labels.get(item.action, ""),
                          status="Chờ chấm" if item.included else "Bỏ qua",
                          valid="", total10="", acc4="", expr6="",
                          reason="" if item.included else "Đã bỏ qua trước khi chạy.", result_dir="")
                     for i, item in enumerate(self.items)]

    def _execute(self, cmd, out, index):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        with (out / "log.txt").open("w", encoding="utf-8") as log:
            with subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                  errors="replace", env=env,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as proc:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    self.emit("log", index, line)
                return proc.wait()

    def _save_summary(self):
        # Replace atomically so an interrupted export cannot truncate the last summary.
        temp = self.output / "summary.tmp"
        with temp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        temp.replace(self.output / "summary.csv")

    def run(self):
        output_created = False
        try:
            errors = validate_items(self.items, self.labels)
            if errors:
                raise ValueError("\n".join(errors))
            self.output.mkdir(parents=True, exist_ok=False)
            output_created = True
            manifest = {"pipeline": self.pipeline, "preview_seconds": self.preview_seconds,
                        "items": [asdict(item) for item in self.items]}
            (self.output / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            self._save_summary()
            for index, item in enumerate(self.items):
                if not item.included:
                    continue
                if self.stop.is_set():
                    self.rows[index]["status"] = "Chưa chấm (đã dừng)"
                    self.rows[index]["reason"] = "Người dùng dừng batch trước khi chấm video này."
                    self.emit("result", index, dict(self.rows[index]))
                    continue
                row = self.rows[index]
                out = self.output / f"{index + 1:04d}__{Path(item.video).stem[:50]}__{item.action}"
                self.emit("start", index, str(out))
                try:
                    out.mkdir()
                    row["result_dir"] = str(out)
                    code = self.execute(build_command(item, out, self.pipeline, self.preview_seconds), out, index)
                    result_path = out / "result.json"
                    if not result_path.is_file():
                        raise RuntimeError(f"Không có kết quả (mã thoát {code}); xem log.txt.")
                    result = json.loads(result_path.read_text(encoding="utf-8"))["result"]
                    row.update(valid=int(bool(result["valid"])),
                               status="Đã chấm" if result["valid"] else "Không chấm được",
                               total10=result.get("total") if result["valid"] else "",
                               acc4=result.get("acc") if result["valid"] else "",
                               expr6=result.get("expr") if result["valid"] else "",
                               reason=" | ".join(result.get("warnings") or []) if result["valid"]
                               else result.get("reason") or "Bộ chấm không cung cấp lý do; xem log.txt.")
                    if code:
                        raise RuntimeError(f"Tiến trình thoát với mã {code}; xem log.txt.")
                except Exception as exc:
                    reason = str(exc)
                    log_path = out / "log.txt"
                    if log_path.is_file():
                        try:
                            tail = log_path.read_text(encoding="utf-8", errors="replace")[-1200:].strip()
                            if tail:
                                reason += "\nChi tiết log:\n" + tail
                        except OSError:
                            pass
                    row.update(status="Lỗi", reason=reason, valid="", total10="", acc4="", expr6="")
                    with (self.output / "batch_errors.log").open("a", encoding="utf-8") as log:
                        log.write(f"{item.video}: {exc}\n")
                self._save_summary()
                self.emit("result", index, dict(row))
            self._save_summary()
        except Exception as exc:
            for row in self.rows:
                if row["status"] == "Chờ chấm":
                    row.update(status="Chưa chấm (batch lỗi)", reason=str(exc))
            if output_created:
                try:
                    self._save_summary()
                except OSError:
                    pass
            self.emit("fatal", -1, str(exc))
        finally:
            self.emit("done", -1, str(self.output))
