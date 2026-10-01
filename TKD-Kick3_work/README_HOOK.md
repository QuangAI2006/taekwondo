# Mở rộng: Đấm móc phải (hook_right)

Module này bổ sung kỹ thuật **đấm móc phải** vào hệ thống nhận diện + chấm điểm hiện có.
**Không file cũ nào bị sửa**: training cũ (3 lớp đá), checkpoint `models/checkpoints/sample_level/`,
dữ liệu `data/sequences/`, `predict_video.py`, `scoring_api.py` giữ nguyên và vẫn chạy như trước.

## Thành phần mới

| File | Vai trò |
|---|---|
| `scripts/hook_pose.py` | Trích skeleton. Dùng lại tracking của `extract_pose_mediapipe.py`, thêm cửa sổ hành động theo **cổ tay** (extractor cũ cắt theo cổ chân nên cắt lệch cú đấm). Một lần chạy MediaPipe cho ra 2 view: *kick view* (giống hệt extractor cũ) và *punch view*. |
| `scripts/hook_build_dataset.py` | 24 clip trong `E:/martial arts/clips` → `data/sequences_hook/{train,val,test}/hook_right/*.npz` (16/4/4, seed 0) + `outputs/hook/split_manifest.csv`. |
| `scripts/train_kick4_hook.py` | Train model 4 lớp `front, roundhouse, axe, hook_right`: dữ liệu đá cũ (chỉ đọc) + dữ liệu móc. Cùng model/tiền xử lý/siêu tham số với `train_kick3_baselines.py` (LSTM, uniform, 40 epoch, batch 8, lr 1e-3, dropout 0.2, seed 0/1/2, sampler cân bằng lớp). Chọn checkpoint theo *balanced accuracy* trên val (vì val chỉ có 4 clip móc so với 104 clip đá). |
| `scripts/hook_scoring.py` | Chấm điểm đấm móc, thang 10 = Độ chính xác 4 + Biểu hiện 6 (giống `scoring_api.py`). `--calibrate` tạo `outputs/hook/hook_score_ranges.json`. |
| `scripts/predict_video_kick4.py` | Nhận diện 4 lớp + chấm điểm cho video mới. |

Kết quả / checkpoint mới: `models/checkpoints/kick4_hook/`, `outputs/hook/`, `outputs/predictions_kick4/`.

## Cách chạy

```powershell
# 1) Trích skeleton từ clips (chạy lại khi thêm clip mới vào E:\martial arts\clips)
.\.venv\Scripts\python.exe scripts\hook_build_dataset.py

# 2) Train model 4 lớp (3 seed, ~20 phút trên CPU)
.\.venv\Scripts\python.exe scripts\train_kick4_hook.py

# 3) Hiệu chỉnh ngưỡng chấm điểm móc từ clip mẫu
.\.venv\Scripts\python.exe scripts\hook_scoring.py --calibrate

# 4) Dự đoán + chấm điểm video mới (đá hoặc đấm móc)
.\.venv\Scripts\python.exe scripts\predict_video_kick4.py "D:\video\test.mp4"
.\.venv\Scripts\python.exe scripts\predict_video_kick4.py "D:\video\test.mp4" --kick hook_right   # bỏ qua nhận diện

# 5) Video trực quan: bbox, keypoint 2D (33 + 13 điểm), 3D keypoint, cửa sổ động tác, xác suất 4 lớp, điểm
.\.venv\Scripts\python.exe scripts\visualize_video_kick4.py "D:\video\test.mp4"
#    -> outputs/predictions_kick4/<tên video>/visualization.mp4, peak.png, result.json
```

## Quy tắc nhận diện

Mỗi video được trích một lần, tạo 2 view. Model 4 lớp chạy trên cả hai:
`p_hook` = xác suất hook_right trên *punch view*, `p_kick` = xác suất lớp đá cao nhất trên *kick view*.
Nếu `p_hook > p_kick` → hook_right, ngược lại → lớp đá đó. Đá được chấm bằng `scoring_api.py` (không đổi),
đấm móc được chấm bằng `hook_scoring.py`.

## Tiêu chí chấm đấm móc phải

Tọa độ 2D, chuẩn hóa theo **chiều dài thân** (tâm vai → tâm hông). Không dùng độ rộng vai vì khi xoay người đấm móc, hai vai chồng lên nhau làm độ rộng vai co về gần 0.

| Mã | Mục | Điểm tối đa | Chỉ số |
|---|---|---|---|
| HK_A1 | Góc khuỷu & tay ngang | 1.4 | góc khuỷu lúc chạm (tối ưu 75–115°), khuỷu ngang vai, cẳng tay nằm ngang |
| HK_A2 | Xoay vai – hông | 1.2 | độ co hình chiếu vai/hông (proxy xoay thân) |
| HK_A3 | Tay trái thủ | 0.8 | khoảng cách cổ tay trái – đầu |
| HK_A4 | Thu tay & giữ trụ | 0.6 | mức thu tay về thế thủ, dao động tâm hông |
| EX_E1 | Tốc độ | 2.2 | p90 tốc độ cổ tay phải |
| EX_E2 | Độ bộc phát | 1.8 | 0.7·tốc độ + 0.3·gia tốc (proxy, **không phải** công suất cơ sinh học) |
| EX_E3 | Phối hợp xoay người | 1.0 | xoay vai + hông |
| EX_E4 | Độ mượt | 1.0 | mức "một đỉnh" của biên độ tốc độ đến lúc chạm |

Ngưỡng kỹ thuật (góc khuỷu, độ cao khuỷu, tay thủ) cố định trong code. Ngưỡng tốc độ/xoay/mượt/thu tay được
hiệu chỉnh từ clip mẫu: **đạt trung vị clip mẫu = điểm tối đa**. Tất cả là **tạm thời, cần HLV/chuyên gia xác nhận**.

## Kết quả (LSTM uniform, 3 seed)

Nguồn: `outputs/hook/results/kick4_lstm_uniform_summary.csv` và các file JSON theo từng seed.

| Seed | Val acc (108) | Val balanced acc | Recall hook_right val / test | Acc chỉ lớp đá trên val |
|---|---|---|---|---|
| 0 | 0.676 | 0.753 | 1.00 / 1.00 | 0.663 |
| 1 | 0.741 | 0.799 | 1.00 / 1.00 | 0.731 |
| 2 | 0.741 | 0.799 | 1.00 / 1.00 | 0.731 |

- Không có clip đá nào bị nhận nhầm thành hook_right (val).
- Model 3 lớp cũ trên cùng val đá: acc 0.779 / 0.740 / 0.740 (seed 0/1/2). Model 4 lớp thấp hơn nhẹ ở seed 0.
- Tập test đá cũ (13 file, dạng x,y,z) cho kết quả kém với **cả model cũ** (acc 0.15–0.38) lẫn model mới. Đây là vấn đề sẵn có của tập test đó.
- End-to-end (`predict_video_kick4.py`) trên các clip móc test/val: nhận đúng 5/5, điểm 7.85–9.52.

## Giới hạn cần biết

- Cả 24 clip được cắt từ **cùng một video hướng dẫn, cùng một người tập, cùng góc máy (chính diện)**. Kết quả
  nhận diện/chấm điểm trên người khác, góc quay khác chưa được kiểm chứng. Nên bổ sung clip của nhiều người,
  nhiều góc máy, và cả clip đấm móc sai kỹ thuật.
- Chia tập ở mức clip (sample-level), **không** độc lập theo người tập (không có metadata người tập).
- Chỉ có 4 clip móc ở val và 4 ở test: chỉ số của lớp hook_right có độ bất định rất lớn.
- Không có video gốc của dữ liệu đá (chỉ có npz), nên quy tắc 2-view chưa được kiểm tra end-to-end trên video đá;
  model thì đã được đánh giá trên npz đá val/test.
- 13 file `data/sequences/test` cũ có kpts dạng (x,y,z); script mới chỉ dùng (x,y), đúng quy ước của `scoring_api.py`.
