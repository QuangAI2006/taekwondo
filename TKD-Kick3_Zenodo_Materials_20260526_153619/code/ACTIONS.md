# Pipeline chung: đá, đấm và động tác train thêm

```
video ─► MediaPipe Pose (2D + 3D) ─► chọn người & cắt đoạn ─► model nhận dạng (LSTM/GCN)
      ─► luật chấm của động tác (action_scorers.py) ─► điểm 10 = Chính xác 4 + Biểu hiện 6
```

| File (`scripts/`) | Vai trò |
|---|---|
| `infer_action.py` | **Lệnh chạy chung** (1 video hoặc thư mục), mở cửa sổ pipeline |
| `infer_video.py` | Lệnh cũ cho đá — lớp bọc của `infer_action.py` (`--kick` = `--action`) |
| `infer_punch_video.py` | Lệnh cũ cho đấm — lớp bọc, mặc định không dùng model (`--punch` = `--action`) |
| `action_pose.py` | Trích pose cả video (2D + 3D) + các cách cắt đoạn đưa vào model |
| `action_classifier.py` | Nạp checkpoint bất kỳ; số và tên động tác đọc từ checkpoint |
| `action_scorers.py` | Bảng tra "động tác → luật chấm" (đá, đấm có sẵn) |
| `scorers_*.py` | Luật chấm các động tác còn lại (đá sau, đỡ, gạt, chém, tấn, đòn phối hợp, bộ pháp...) — xem [OTHER_TECHNIQUES_SCORING.md](OTHER_TECHNIQUES_SCORING.md) |
| `score_segments.py` | Chấm **từng lần thực hiện** trong các đoạn của video dài (vd video bài giảng 30 s/bài) → CSV |
| `pipeline_viewer.py` | Cửa sổ pipeline |
| `prepare_action_data.py` | Video xếp theo thư mục động tác → dữ liệu train |
| `train_actions.py` | Train model nhận dạng với số động tác bất kỳ |

## Chạy

```powershell
cd "E:\martial arts\tkd\TKD-Kick3_Zenodo_Materials_20260526_153619"

# Model đá của tác giả (LSTM ×3) -> nhận dạng front/roundhouse/axe -> chấm theo đòn nhận dạng
python code/scripts/infer_action.py "video_da.mp4"

# Chỉ định động tác cần chấm (bỏ qua nhận dạng nếu model chưa học động tác đó)
python code/scripts/infer_action.py "video_dam.mp4" --action hook

# Model tự train
python code/scripts/infer_action.py "video.mp4" --ckpt "models/checkpoints/my_actions/*.pt"

# Cả thư mục -> CSV (không mở cửa sổ, trừ khi thêm --window on)
python code/scripts/infer_action.py "E:/martial arts/clips" --ckpt "models/checkpoints/my_actions/*.pt" --csv diem.csv
```

| Tham số | Ý nghĩa |
|---|---|
| `--action` | Chỉ định động tác cần chấm. Không có → chấm theo kết quả nhận dạng |
| `--ckpt` | Checkpoint nhận dạng: `a.pt,b.pt`, glob `dir/*.pt` hoặc thư mục. Nhiều file = ensemble |
| `--model/--align/--seeds` | Khi không có `--ckpt`: chọn checkpoint của tác giả (mặc định `lstm`, `uniform`, `0,1,2`) |
| `--no_model` | Không nhận dạng (cần `--action`) |
| `--guard chin/waist`, `--side L/R` | Tùy chọn cho luật đấm |
| `--kick_window ankle/none`, `--fix_leg` | Tùy chọn cho luật đá: đoạn cắt của tác giả hay cả video; tự chọn đúng chân đá (xem [OTHER_TECHNIQUES_SCORING.md](OTHER_TECHNIQUES_SCORING.md)) |
| `--window auto/on/off` | Cửa sổ pipeline (`auto`: mở khi chấm 1 video) |
| `--save_dashboard`, `--keyframes`, `--out`, `--csv`, `--save_pose` | Lưu ảnh dashboard / 3 frame mốc / JSON / CSV / pose |

### Cửa sổ pipeline

6 bước: **Đọc video → MediaPipe Pose → Chọn người & cắt đoạn → Nhận dạng (chỉ với đá) → Chia pha & đo chỉ số → Chấm điểm**.
Bước đang chạy tô cam, xong tô xanh, lỗi tô đỏ, bỏ qua tô xám; dưới mỗi bước là kết quả của bước đó.
Bên dưới: video (khung xương vẽ trực tiếp lúc MediaPipe chạy, chấm xong thì phát lại chậm 2×), xác suất
từng đòn đá, tín hiệu chia pha (vùng chấm = đoạn đưa vào model) và điểm từng hạng mục.

Khi model trả về động tác có luật chấm, điểm được tính tự động. Động tác **chưa có luật** vẫn được nhận dạng
và hiện xác suất, ô chấm điểm báo "chưa có luật chấm". Nếu xác suất cao nhất < 50%, có cảnh báo model không chắc.

## Thêm động tác mới

### Bước 1 — Quay video và xếp thư mục

```
my_videos/
  hook/       *.mp4      <- tên thư mục = tên động tác
  straight/   *.mp4
  elbow/      *.mp4
```

- Mỗi video **một lần** thực hiện động tác, cắt ngắn 1–3 giây, thấy toàn thân, góc chính diện hoặc chéo ~45°.
- Đặt tên thư mục **trùng tên luật chấm** (`hook`, `straight`, `front`, `roundhouse`, `axe`, `side`, ...)
  để được chấm điểm tự động. Tên tiếng Việt không dấu như `dam_moc`, `dam_thang` cũng được nhận
  (bảng `ALIASES` trong `action_scorers.py`).
- Nên có **≥ 30–50 video mỗi động tác, từ nhiều người khác nhau**.
- **Chia train/val theo người**: nếu cùng một người có video ở cả train và val, độ chính xác đo được sẽ cao
  ảo. Muốn tự chia thì dùng cấu trúc `my_videos/train/<động tác>/` và `my_videos/val/<động tác>/`;
  nếu không, script tự chia ngẫu nhiên 20% làm val.

### Bước 2 — Chuẩn bị dữ liệu

```powershell
python code/scripts/prepare_action_data.py --videos "E:/martial arts/my_videos" --out data/my_sequences --include_public_kicks
```

`--include_public_kicks` gộp thêm 765 mẫu đá public (front/roundhouse/axe) để model mới vẫn nhận ra đòn đá.

### Bước 3 — Train

```powershell
python code/scripts/train_actions.py --data data/my_sequences --out models/checkpoints/my_actions --seeds 0,1,2
```

In ra accuracy / macro-F1 / ma trận nhầm lẫn trên val; lưu `lstm_seed0.pt`, `lstm_seed1.pt`, ... (dùng chung
làm ensemble). Dùng `--model gcn` để train GCN.

### Bước 4 — Chạy

```powershell
python code/scripts/infer_action.py "video.mp4" --ckpt "models/checkpoints/my_actions/*.pt"
```

### Kết quả chạy thử quy trình

Thêm lớp `hook` từ 26 clip "Dam moc" vào 765 mẫu đá public → model 4 lớp (LSTM, 20 epoch, ~70s trên CPU):
val accuracy 73%, macro-F1 0.79; `hook` 5/5 đúng. Phần đá xấp xỉ mức 75% của tác giả. **Con số `hook` không có
ý nghĩa đánh giá**: cả 26 clip là cùng một người cùng một buổi quay, nên train và val gần như giống nhau. Lần
chạy này chỉ để kiểm tra quy trình hoạt động. Chạy `infer_action.py` với checkpoint này: nhận dạng
"Đấm móc 99%" rồi tự chấm bằng luật đấm móc.

## Thêm luật chấm cho động tác mới

Tạo file `scripts/scorers_<tên>.py` — `infer_action.py` tự nạp mọi file `scorers_*.py`. Ví dụ "đánh chỏ":

```python
# scripts/scorers_elbow.py — VÍ DỤ luật chấm cho động tác mới "elbow" (đánh chỏ)
import numpy as np

from action_scorers import ScoreResult, make_item, register_scorer

LS, RS, LE, RE, LW, RW = 1, 2, 3, 4, 5, 6  # thứ tự 13 khớp: xem action_pose.py


def _angle(a, b, c):
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


@register_scorer("elbow", "Đánh chỏ")
def score_elbow(pose, action, **opts):
    k, w, fps = pose["kpts"], pose["world"], float(pose["fps"])  # world: tọa độ 3D (mét)
    if len(k) < 8:
        return ScoreResult(action=action, valid=False, reason="Video quá ngắn")
    S, E, W = RS, RE, RW  # tay phải
    speed = np.r_[0.0, np.linalg.norm(np.diff(w[:, E], axis=0), axis=1)] * fps  # m/s
    t_peak = int(np.argmax(speed))
    t0 = max(0, t_peak - int(0.3 * fps))
    t_end = min(len(k) - 1, t_peak + int(0.5 * fps))
    flex = float(_angle(w[t_peak, S], w[t_peak, E], w[t_peak, W]))

    acc = [make_item("accuracy", "EL_A1", "Gập khuỷu khi đánh", 4.0, 4.0 * np.clip((120 - flex) / 60, 0, 1),
                     f"góc khuỷu≈{flex:.0f}°", "Gập chặt tay, mũi chỏ dẫn đường.")]
    expr = [make_item("expression", "EL_E1", "Tốc độ chỏ", 6.0, 6.0 * np.clip(speed[t_peak] / 5.0, 0, 1),
                      f"≈{speed[t_peak]:.1f} m/s", "Xoay hông để tăng tốc.")]
    a, e = acc[0]["got"], expr[0]["got"]
    issues = sorted([x for x in acc + expr if x["deduct"] > 0], key=lambda x: -x["deduct"])
    return ScoreResult(
        action=action, valid=True, total=a + e, acc=a, expr=e, acc_items=acc, expr_items=expr,
        top_issues=issues[:3],
        phases={"t0": t0, "t_peak": t_peak, "t_end": t_end},
        phase_labels=("Bắt đầu", "Chạm", "Kết thúc"),
        phase_names=("Thủ", "Ra đòn", "ĐIỂM CHẠM", "Thu về", "Kết thúc"),
        series={"sig": speed, "sig_name": "Tốc độ khuỷu (m/s)"},
        highlight=(S, E, W), side_text="tay phải", status=f"khuỷu {flex:.0f}°",
    )
```

Đây chỉ là **khung mẫu** (đã chạy thử được); ngưỡng và hạng mục cần thiết kế lại theo kỹ thuật thật, như đã
làm cho đấm trong [PUNCH_SCORING.md](PUNCH_SCORING.md).

- `pose` có: `kpts` (T,13,2) pixel, `vis` (T,13), `world` (T,13,3) mét, `fps` — của **cả video**.
- `make_item(category, code, label, maxi, got, reason, tip)`: `category` là `"accuracy"` hoặc `"expression"`;
  `got=None` = không đánh giá được.
- `phases` dùng index frame của cả video; `series["sig"]` là tín hiệu vẽ ở ô chia pha (có thể thêm
  `"speed"`, `"speed_name"` cho trục phải); `highlight` là các khớp tô đỏ khi phát lại.
- Không bắt buộc theo khung 4 + 6, nhưng nên giữ để so sánh được với đá/đấm.
- Cửa sổ pipeline hiển thị **tiếng Anh**: đặt thêm `"label_en"` cho từng hạng mục (hoặc thêm bản dịch tên
  động tác, nhãn pha, cảnh báo vào `scripts/viewer_text_en.py`); chuỗi chưa có bản dịch sẽ hiện nguyên văn.

## Cắt đoạn đưa vào model

| Chế độ | Cách cắt | Dùng cho |
|---|---|---|
| `ankle` | Hàm `_compute_rel_speed_window` của tác giả (cổ chân so với hông) | Checkpoint của tác giả — cho kết quả **giống hệt** `infer_video.py` cũ (đã kiểm: 54.0 / 38.4 / 7.6%, điểm 4.30) |
| `motion` | Đoạn chuyển động mạnh nhất của cả cơ thể (4 đầu chi) + đệm 0.4s | **Mặc định cho model tự train** — không cần biết trước là đá hay đấm |
| `wrist` | Thuật toán của tác giả áp cho cổ tay so với vai | Model chỉ gồm động tác tay |
| `none` | Cả video | Video đã cắt sẵn rất sát |

Chế độ được lưu trong checkpoint (`window_mode`) và `prepare_action_data.py` dùng **đúng hàm** đó, nên dữ
liệu train và lúc chạy luôn cắt giống nhau. Không dùng cách "tự đoán đá hay đấm để cắt theo chân/tay":
thử trên dữ liệu thì 10–15% video đá bị đoán nhầm (người tập vung tay mạnh khi đá).
Với `motion`: chứa trọn cú đấm ở 25/25 clip đấm móc, giữ ~70% đoạn cắt của tác giả với đá (cả pha nâng gối và thu chân).

## Giới hạn

- Model nhận dạng là **tập đóng**: luôn chọn một trong các động tác đã học. Video động tác lạ vẫn ra một
  nhãn — xem xác suất và cảnh báo "model không chắc".
- Chất lượng model phụ thuộc dữ liệu: cần nhiều người, nhiều góc quay, và chia train/val theo người.
- Luật chấm đá (của tác giả) và đấm đều **chưa đối chiếu đủ với trọng tài**; xem PUNCH_SCORING.md.
