# scripts/infer_video.py
"""
Chấm điểm ĐÁ từ video (model nhận dạng của tác giả + luật chấm scoring_api.py) — lớp bọc của infer_action.py,
giữ nguyên các tham số cũ (--kick được hiểu là --action). Khi chấm 1 video sẽ mở cửa sổ pipeline.

dụ:
  python code/scripts/infer_video.py path/to/video.mp4
  python code/scripts/infer_video.py video.mp4 --model gcn --kick axe --out result.json
  python code/scripts/infer_video.py video.mp4 --window off
Xem thêm tham số: python code/scripts/infer_action.py -h
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_action import main  # noqa: E402

if __name__ == "__main__":
    main(["--action" if a == "--kick" else a for a in sys.argv[1:]])
