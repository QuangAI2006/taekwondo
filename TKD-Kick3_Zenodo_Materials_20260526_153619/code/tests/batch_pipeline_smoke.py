"""Real subprocess/pipeline smoke test with temporary blank videos (no study data)."""
import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cv2
import numpy as np
from batch_scoring import BatchItem, BatchRunner


def main():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        items = []
        for index in range(2):
            video = root / f"blank_{index}.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
            assert writer.isOpened()
            for _ in range(5):
                writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            writer.release()
            items.append(BatchItem(str(video), "front", reviewed=True))
        events = []
        runner = BatchRunner(items, {"front": "Đá trước"}, root / "batch",
                             lambda *event: events.append(event), pipeline=True, preview_seconds=0.2)
        runner.run()
        assert not any(event[0] == "fatal" for event in events), events
        assert len([event for event in events if event[0] == "start"]) == 2
        assert all(row["status"] == "Không chấm được" for row in runner.rows), runner.rows
        for row in runner.rows:
            out = Path(row["result_dir"])
            result = json.loads((out / "result.json").read_text(encoding="utf-8"))["result"]
            assert not result["valid"] and not result["reason"].startswith("Lỗi:"), result
            assert (out / "log.txt").is_file()
        with (runner.output / "summary.csv").open(encoding="utf-8-sig", newline="") as file:
            assert len(list(csv.DictReader(file))) == 2
    print("Two real scoring subprocesses with auto-closing pipeline completed sequentially; blank videos correctly not scored.")


if __name__ == "__main__":
    main()
