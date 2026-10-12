"""Run manually on Windows: hidden Tk UI checks with synthetic scoring results."""
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import score_batch_app
from batch_scoring import BatchRunner


def main():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        videos = root / "videos"
        videos.mkdir()
        for filename in ["đá trước - trái.mp4", "gạt thấp - đấm thẳng - gạt phải.mp4", "unknown.mp4", "đá chẻ - trái.mp4"]:
            (videos / filename).touch()
        app = score_batch_app.BatchApp(score_batch_app.load_actions())
        app.withdraw()
        warnings = []
        calls = []

        def execute(cmd, out, index):
            calls.append(index)
            action = cmd[cmd.index("--action") + 1]
            if action == "straight":
                (out / "log.txt").write_text("Synthetic GUI fixture: video cannot be read", encoding="utf-8")
                return 1
            (out / "log.txt").write_text("Synthetic GUI fixture", encoding="utf-8")
            (out / "result.json").write_text(json.dumps({"result": {
                "valid": action != "block_punch", "total": 8.0, "acc": 3.0, "expr": 5.0,
                "reason": "Synthetic fixture: Không phát hiện được người trong video." if action == "block_punch" else "",
            }}), encoding="utf-8")
            return 0

        def runner_factory(items, labels, output, emit, pipeline, seconds):
            return BatchRunner(items, labels, root / "batch", emit, pipeline, seconds, execute)

        try:
            with patch.object(score_batch_app.messagebox, "showwarning", side_effect=lambda *args: warnings.append(args)), \
                 patch.object(score_batch_app, "BatchRunner", side_effect=runner_factory):
                app.folder.set(str(videos))
                app._rescan()
                assert len(app.items) == 4
                assert {item.action for item in app.items} == {"front", "block_punch", "axe", ""}
                app._start()
                assert not app.running and warnings  # Preflight must block unreviewed rows.
                unknown = next(i for i, item in enumerate(app.items) if not item.action)
                app.tree.selection_set(str(unknown))
                app.update()
                app.action.set(next(choice for choice, name in app.choices.items() if name == "straight"))
                app.side.set("Trái (L)")
                app._apply()
                assert app.items[unknown].action == "straight"
                assert app.items[unknown].side == "L" and app.items[unknown].reviewed
                app._confirm_all()
                assert all(item.reviewed for item in app.items)
                app._start()
                assert app.running and str(app.btn_run["state"]) == "disabled"
                deadline = time.monotonic() + 15
                while app.running and time.monotonic() < deadline:
                    app.update()
                    time.sleep(0.02)
                assert not app.running
                assert calls == [0, 1, 2, 3]
                assert (root / "batch" / "summary.csv").is_file()
                expected = {"front": "Đã chấm", "axe": "Đã chấm", "straight": "Lỗi", "block_punch": "Không chấm được"}
                for index, item in enumerate(app.items):
                    values = app.tree.item(str(index), "values")
                    assert values[4] == expected[item.action]
                    if item.action in {"straight", "block_punch"}:
                        assert values[6]
                report = app._summary_window
                assert report and report.winfo_exists()
                report.withdraw()
                assert len(report.tree.get_children()) == 4
                for index, row in enumerate(app.report_rows):
                    values = report.tree.item(str(index), "values")
                    assert values[2] == row["status"]
                    report.tree.selection_set(str(index))
                    report._select_result()
                    detail = report.detail.get("1.0", "end")
                    assert row["video"] in detail
                    if row["status"] != "Đã chấm":
                        assert values[3] == "—" and row["reason"] in detail
                    else:
                        assert values[3:6] == ("8.00", "3.00", "5.00")
                report.destroy()
                app._show_summary()
                assert app._summary_window.winfo_exists()
                app._summary_window.withdraw()
                app._select()
                assert "Synthetic GUI fixture" in app.detail.get("1.0", "end")
                assert str(app.btn_run["state"]) == "normal"
        finally:
            app.destroy()
    print("GUI review, axe detection, mixed success/failure reasons, automatic summary, and report reopening passed.")


if __name__ == "__main__":
    main()
