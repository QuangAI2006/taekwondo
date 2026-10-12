"""Batch orchestration tests using synthetic results, not study measurements."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from batch_scoring import BatchItem, BatchRunner, build_command, scan_folder, validate_items


class BatchScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.labels = {"front": "Đá trước", "block_punch": "Gạt thấp – đấm thẳng", "straight": "Đấm thẳng"}
        self.actions = [("", key, label) for key, label in self.labels.items()]

    def video(self, name, action="front", reviewed=True):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return BatchItem(str(path), action, reviewed=reviewed)

    def write_synthetic_result(self, out, valid=True):
        (out / "result.json").write_text(json.dumps({"result": {
            "valid": valid, "total": 8.0 if valid else None,
            "acc": 3.0 if valid else None, "expr": 5.0 if valid else None,
            "reason": "Synthetic invalid-video fixture" if not valid else "",
        }}), encoding="utf-8")

    def test_scan_autodetection_recursion_and_validation(self):
        self.video("đá trước - trái.MP4")
        self.video("gạt thấp - đấm thẳng - gạt phải.mov")
        self.video("nested/unknown.avi")
        (self.root / "notes.txt").touch()
        items = scan_folder(self.root, self.actions, {})
        self.assertEqual(len(items), 2)
        self.assertEqual({item.action for item in items}, {"front", "block_punch"})
        self.assertTrue(all(not item.reviewed for item in items))
        self.assertTrue(validate_items(items, self.labels))
        recursive = scan_folder(self.root, self.actions, {}, True)
        self.assertEqual(len(recursive), 3)
        unknown = next(item for item in recursive if not item.action)
        unknown.included = False
        for item in recursive:
            item.reviewed = True
        self.assertEqual(validate_items(recursive, self.labels), [])
        Path(items[0].video).unlink()
        self.assertTrue(any("không tìm thấy" in error for error in validate_items(items, self.labels)))

    def test_sequential_processing_continues_after_failure_and_exports(self):
        items = [self.video(f"{index}.mp4") for index in range(4)]
        events, calls = [], []

        def execute(cmd, out, index):
            if calls:
                self.assertTrue((output / "summary.csv").is_file())
                self.assertEqual(runner.rows[calls[-1]]["status"],
                                 ["Đã chấm", "Lỗi", "Không chấm được"][calls[-1]])
            calls.append(index)
            (out / "log.txt").write_text("Synthetic fixture\n", encoding="utf-8")
            if index == 1:
                return 1  # Deliberately missing output; next video must still run.
            self.write_synthetic_result(out, valid=index != 2)
            return 0

        output = self.root / "batch"
        runner = BatchRunner(items, self.labels, output, lambda *event: events.append(event), execute=execute)
        items[0].action = "straight"  # The runner must keep the reviewed snapshot.
        runner.run()
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertEqual([row["status"] for row in runner.rows],
                         ["Đã chấm", "Lỗi", "Không chấm được", "Đã chấm"])
        with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["total10"], "8.0")
        self.assertEqual(rows[1]["total10"], "")
        self.assertIn("Không có kết quả", rows[1]["reason"])
        self.assertIn("Synthetic fixture", rows[1]["reason"])
        self.assertEqual(rows[2]["reason"], "Synthetic invalid-video fixture")
        self.assertEqual(rows[0]["action"], "front")
        self.assertTrue((output / "manifest.json").is_file())
        self.assertEqual(events[-1][0], "done")
        self.assertFalse(any(event[0] == "fatal" for event in events))

    def test_stop_after_current_and_skip(self):
        items = [self.video(f"stop_{i}.mp4") for i in range(4)]
        items[1].included = False
        calls = []

        def execute(cmd, out, index):
            calls.append(index)
            self.write_synthetic_result(out)
            runner.stop.set()
            return 0

        runner = BatchRunner(items, self.labels, self.root / "stopped", lambda *args: None, execute=execute)
        runner.run()
        self.assertEqual(calls, [0])
        self.assertEqual([row["status"] for row in runner.rows],
                         ["Đã chấm", "Bỏ qua", "Chưa chấm (đã dừng)", "Chưa chấm (đã dừng)"])
        self.assertTrue(all(row["reason"] for row in runner.rows[1:]))

    def test_validation_blocks_execution(self):
        events, calls = [], []
        item = self.video("not_reviewed.mp4", reviewed=False)
        runner = BatchRunner([item], self.labels, self.root / "blocked",
                             lambda *event: events.append(event), execute=lambda *args: calls.append(args))
        runner.run()
        self.assertEqual(calls, [])
        self.assertFalse(runner.output.exists())
        self.assertEqual([event[0] for event in events], ["fatal", "done"])

    def test_options_only_sent_to_applicable_actions(self):
        item = self.video("options.mp4")
        item.side, item.fix_leg, item.guard = "L", True, "waist"
        cmd = build_command(item, self.root, True, 3)
        self.assertIn("--fix_leg", cmd)
        self.assertNotIn("--side", cmd)
        self.assertNotIn("--guard", cmd)
        self.assertEqual(cmd[cmd.index("--window") + 1], "on")
        item.action = "straight"
        cmd = build_command(item, self.root, False, 0)
        self.assertNotIn("--fix_leg", cmd)
        self.assertEqual(cmd[cmd.index("--side") + 1], "L")
        self.assertEqual(cmd[cmd.index("--guard") + 1], "waist")


if __name__ == "__main__":
    unittest.main()
