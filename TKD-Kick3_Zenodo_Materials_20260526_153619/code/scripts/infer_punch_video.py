# scripts/infer_punch_video.py
"""
Chấm điểm ĐẤM từ video (luật punch_scoring.py) — lớp bọc của infer_action.py, giữ nguyên tham số cũ
(--punch được hiểu là --action). Mặc định không dùng model nhận dạng (model của tác giả chỉ có đòn đá);
nếu bạn đã train model có lớp đấm thì truyền --ckpt để có thêm bước nhận dạng.

Ví dụ:
  python code/scripts/infer_punch_video.py "clip.mp4" --punch hook
  python code/scripts/infer_punch_video.py "clip.mp4" --punch hook --keyframes kf.jpg --out result.json
  python code/scripts/infer_punch_video.py "E:/martial arts/clips" --punch hook --csv punch_scores.csv
  python code/scripts/infer_punch_video.py "vovinam.mp4" --punch hook --guard waist
Xem thêm tham số: python code/scripts/infer_action.py -h
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_action import main  # noqa: E402

if __name__ == "__main__":
    argv = ["--action" if a == "--punch" else a for a in sys.argv[1:]]
    if "--ckpt" not in argv and "--no_model" not in argv:
        argv.append("--no_model")
    main(argv)
