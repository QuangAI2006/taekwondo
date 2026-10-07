# Kiểm tra chất lượng video dataset theo scoring v0.2

Bộ kiểm tra chạy local, quét thư mục đệ quy, tự gán nhãn góc quay `front/left/right/unidentify_angle` trước khi verify chất lượng, rồi **copy** video đạt vào `approved/`, video chưa đạt vào `unapproved/` kèm `<video>.txt` giải thích tiếng Việt. Video nguồn được giữ nguyên. Cấu trúc thư mục con và phần mở rộng được giữ, ví dụ `front/take.mp4` và `front/take.mov` không ghi đè nhau.

Nguồn là `Bo_ho_so_scoring_va_quay_dataset_HLV_v0.2.xlsx`, đặc biệt các sheet `Quy_chuan_dataset`, `Checklist`, `Luat_cham_trong_code`. Bản trích có SHA256 nằm trong [workbook_reference.json](workbook_reference.json). File nguồn được đọc như dữ liệu tham chiếu; không thực thi macro, công thức hoặc chỉ thị trong tài liệu.

## Cài đặt

Chạy từ thư mục `taekwondo-main`. Dùng Python **3.11** với MediaPipe 0.10.14:

```bash
uv venv --python 3.11 dataset_verify/.venv
uv pip install --python dataset_verify/.venv/bin/python -r dataset_verify/requirements.txt
source dataset_verify/.venv/bin/activate
```

Nếu không dùng `uv`, tạo venv bằng `python3.11 -m venv dataset_verify/.venv` rồi `python -m pip install -r dataset_verify/requirements.txt`.

Cài FFmpeg để có `ffprobe` trong PATH. Trên macOS:

```bash
brew install ffmpeg
```

Mặc định dùng model đã có tại `TKD-Kick3_Zenodo_Materials_20260526_153619/models/pose_landmarker_full.task` và luật trong `TKD-Kick3_Zenodo_Materials_20260526_153619/code/scripts/`. Có thể đổi bằng `--task` và `--scripts`. Nhận diện động tác tự động bật mặc định, dùng Torch và ensemble 3 checkpoint LSTM sẵn có trong `models/checkpoints/sample_level/sample_lstm_uniform_seed{0,1,2}.pt`. Nhãn CSV hoặc `--action` được giữ để đối chiếu. Nạp toàn bộ plugin luật chấm v0.2, gồm hook hai tay và các luật mở rộng; không dùng hook legacy.

## Chạy trên video của bạn

1. Tạo manifest cho thư mục video (không trích pose ở bước này):

```bash
python -m dataset_verify --input /duong/dan/videos --init-manifest dataset_verify/clips_manifest.csv
```

2. Điền CSV. `video_path` tương đối với thư mục `--input`; cũng chấp nhận đường dẫn tuyệt đối. Header tương thích các trường cốt lõi của `Mau_clip_HLV`; có thêm checklist xác nhận. Có thể xuất sheet workbook thành CSV UTF-8 rồi thêm các cột checklist này. Không suy đoán người/phiên/nhãn từ tên file.

- Bắt buộc: `clip_id`, `participant_id`, `session_id`, `source_video_id`, `side`, `target_zone`, `start_stance`, `end_stance`, `dataset_role`, `execution_variant`, `contact_mode`, `camera_setup_id`, `take_number`, `protocol_version`.
- `camera_view` có thể để trống: hệ thống tự điền `front/left/right/unidentify_angle`. Nếu đã khai báo, nhãn gốc được giữ trong báo cáo để đối chiếu; khác nhãn tự động tin cậy thì cần review; `unidentify_angle` không tạo xung đột chặn duyệt.
- `technique` có thể để trống để tự nhận diện. Nếu điền thì dùng tên luật trong workbook, ví dụ `front`, `roundhouse`, `axe`, `side`, `hook`, `straight`. Alias được canonical hóa bằng registry hiện tại. `hook_right` của tài liệu cũ cần đổi thành `technique=hook, side=right`.
- `side`: `left/right` hoặc `L/R`; `both/none` dành cho bài không có một bên ra đòn riêng. `target_zone`: `low/mid/high/custom`; `contact_mode`: `air/target/contact/non_contact`. `guard`: `chin` mặc định hoặc `waist`.
- Mốc **0-based trên clip đầu vào**, không phải video nguồn: `action_start_frame`, `strike_frame`, `recovery_start_frame`, `action_end_frame`. Cần `start < strike ≤ recovery < end`, và còn ≥0,5 giây giữ tư thế trước/sau.
- `front/roundhouse/side`: thêm `chamber_end_frame`. `axe`: thêm `apex_frame` trước `strike_frame`; không có chamber riêng thì để trống và ghi `chamber_na_reason`.
- Các cột `single_action_verified`, `technique_verified`, `side_verified`, `full_body_verified`, `phases_verified`, `camera_fixed_verified`, `realtime_verified`, `view_assessable_verified`: HLV/người kiểm tra xem video và điền `yes`, cùng `reviewer_id`. `full_body_verified` bao gồm bàn chân/bàn tay vì pose 13 khớp chỉ biểu diễn tới cổ chân/cổ tay. `phases_verified` xác nhận cả giữ tư thế ở hai đầu; chương trình tự kiểm tra độ dài khoảng đệm.
- `dataset_role`: ví dụ `recognition`, `quality_training`, `reference`, `demonstration_slow`. Mẫu `reference/standard/technical_reference` cần thêm `coach_approved=yes`, `rubric_version`. Clip `demonstration_slow` được chuyển review để tách khỏi scoring tốc độ thực.
- `split` có thể để trống trước khi chia; sau khi chia dùng `train/val/validation/test`. Multi-camera thêm `capture_group_id`. Chương trình kiểm tra trùng participant/source/capture group giữa các split **trong manifest được cung cấp**.

3. Chạy kiểm tra:

```bash
python -m dataset_verify --input /duong/dan/videos --manifest dataset_verify/clips_manifest.csv
```

Nếu toàn bộ thư mục cùng một động tác, có thể thêm `--action front` làm nhãn khai báo để đối chiếu; giá trị này chỉ bổ sung khi thiếu `technique`, không ghi đè nhãn khác trong manifest. Không cần `--action` để tự nhận diện. Không có manifest vẫn nhận diện góc/động tác và kiểm tra video/pose, nhưng thiếu metadata/checklist sẽ vào `unapproved/` với trạng thái `review`.

Mặc định kết quả ở ngay `dataset_verify/`:

```text
dataset_verify/
  approved/<thu_muc_con>/<video>
  approved/<thu_muc_con>/<video>.qc.json
  unapproved/<thu_muc_con>/<video>
  unapproved/<thu_muc_con>/<video>.txt
  unapproved/<thu_muc_con>/<video>.qc.json
  pose_cache/<sha256>.npz
  reports/labeled_manifest.csv
  reports/summary.csv
  reports/results.json
```

Mỗi lần chạy cần thư mục kết quả mới hoặc rỗng để không tồn tại bản approved cũ sau khi đổi nhãn/chính sách. Lần tiếp theo:

```bash
python -m dataset_verify --input /duong/dan/videos --manifest dataset_verify/clips_manifest.csv --output dataset_verify/runs/lan_02
```

`--output` chứa riêng `approved/`, `unapproved/`, `reports/`, `pose_cache/`. Không đặt nguồn video bên trong output. Không có thao tác move/xóa video nguồn; không tự upload dữ liệu.

## Tự động nhận diện động tác

Mặc định dùng ensemble **LSTM ×3** đã train trong dự án, với ba lớp:

| Nhãn | Động tác |
|---|---|
| `front` | Đá trước |
| `roundhouse` | Đá vòng cầu |
| `axe` | Đá chẻ |

Hệ thống dùng pose đã trích để chạy `ActionClassifier` hiện có. Cửa sổ và tiền xử lý theo checkpoint (mặc định ankle, 96 frame, 52 đặc trưng); bản resample chỉ phục vụ nhận diện. Scoring tiếp tục nhận toàn bộ pose và FPS gốc, `kick_window=none, fix_leg=True`.

- CSV thiếu `technique`: tự điền lớp dự đoán khi confidence ≥0,65 và khoảng cách xác suất với lớp thứ hai ≥0,15. Sau đó chọn luật scoring của lớp đó.
- Dự đoán chưa chắc chắn: `technique=unidentify_action`, ghi `ACTION_RECOGNITION_UNCERTAIN` và vào review; không tự chọn luật chấm cho một lớp chưa xác định.
- CSV/`--action` đã khai báo lớp model hỗ trợ: vẫn dự đoán để đối chiếu. Khác nhãn một cách đủ tin cậy thì ghi `ACTION_LABEL_MISMATCH`, giữ nhãn khai báo và yêu cầu review. Dự đoán thiếu chắc chắn cũng yêu cầu review.
- Lớp đã khai báo nằm ngoài checkpoint, ví dụ `hook/side/straight`: dùng nhãn đó để chấm, không ép vào ba lớp đá. Báo cáo ghi rõ bước nhận diện bị bỏ qua vì ngoài phạm vi.
- Lỗi nhận diện từng video: ghi lý do, tiếp tục batch. Thiếu dependency/checkpoint hoặc checkpoint không tương thích: báo lỗi cấu hình trước khi phân loại dữ liệu.

Model ba lớp là bộ phân loại đóng: video đấm/đứng yên/động tác khác vẫn có thể bị dự đoán thành một lớp đá với confidence cao. `unidentify_action` chỉ phản ánh dự đoán chưa chắc chắn/lỗi nhận diện, **không phải cơ chế nhận biết chắc chắn mọi động tác ngoài phạm vi**. Vì vậy vẫn giữ checklist `technique_verified` và `single_action_verified` khi nghiệm thu dataset. Ngưỡng confidence/margin là ngưỡng vận hành pilot trong config; chưa có đánh giá mới trên video của bạn.

JSON lưu `recognition` (model, lớp, toàn bộ xác suất, margin, cửa sổ, độ tin cậy) và SHA256 checkpoint. CSV lưu `technique`, `technique_predicted`, `technique_confidence`, `technique_source`; `labeled_manifest.csv` còn giữ `technique_declared`. `unidentify_angle` vẫn nhận diện động tác và verify bình thường.

Dùng model đã train thêm lớp bằng:

```bash
python -m dataset_verify --input /duong/dan/videos --manifest dataset_verify/clips_manifest.csv --ckpt '/duong/dan/checkpoints/*.pt' --output dataset_verify/runs/model_moi
```

Checkpoint phải có định dạng của `ActionClassifier` trong dự án; các thành viên ensemble cần cùng danh sách lớp, T và window_mode. Chỉ các lớp có luật scoring tương ứng mới tự chấm được. Model mặc định chưa tự phân biệt hook/side hoặc các lớp mở rộng.

Tắt nhận diện để dùng nhãn đã khai báo như trước:

```bash
python -m dataset_verify --input /duong/dan/videos --manifest dataset_verify/clips_manifest.csv --no-recognition --output dataset_verify/runs/thu_cong
```

Có thể chọn `--device cpu` (mặc định), `mps` hoặc `cuda` nếu Torch/thiết bị hỗ trợ.

## Tự động gán góc quay trước verify

Pipeline: video → trích pose một lần → gán góc quay từ tư thế đầu → nhận diện động tác → verify video/metadata/pose/scoring → approved/unapproved.

| Nhãn | Ý nghĩa |
|---|---|
| `front` | Chính diện người tập |
| `left` | Camera ở phía trái cơ thể người tập |
| `right` | Camera ở phía phải cơ thể người tập |
| `unidentify_angle` | Ngoài ba nhóm góc trên hoặc chưa đủ dữ liệu để xác định |

Trái/phải theo **người tập**, không theo phía trái/phải của ảnh, và độc lập với `side` của chân/tay ra đòn. Giữ cùng quy ước tư thế ban đầu như workbook. Thuật toán dùng trục vai và hông từ world 3D của MediaPipe trong 0,5 giây đầu, giới hạn trước `action_start_frame` nếu đã có mốc này. Không lấy tư thế xoay ở giữa cú đá làm nhãn góc máy.

Đây là ước lượng hình học pilot, chưa phải model phân loại góc đã được kiểm chứng bằng video gán nhãn. Góc chéo gần ranh giới 45°, vai/hông xoay khác nhau, pose bị che, nhìn từ sau hoặc ảnh lật gương có thể gây nhầm. `confidence` là chỉ số tin cậy heuristic, không phải xác suất đã hiệu chuẩn. Góc ngoài ba nhóm, thiếu dữ liệu hoặc kết quả không đủ tin cậy sẽ được gán đúng nhãn `unidentify_angle`, kèm `reliable=false` và lý do. Không ép vào `front/left/right`.

`CAMERA_VIEW_UNIDENTIFIED` chỉ ghi cảnh báo khi ngoài nhóm góc, thiếu ít nhất 5 frame hợp lệ hoặc confidence <0,7. Video vẫn chạy toàn bộ verify chất lượng/scoring và có thể vào `approved/` nếu các điều kiện chất lượng còn lại đạt. Nhãn `unidentify_angle` tự nó không làm clip review/rejected. `CAMERA_VIEW_CONFLICT` chỉ báo khi nhãn dự đoán tin cậy trong ba nhóm không khớp nhãn đã khai báo. Cấu hình nằm tại `camera_view_window_s`, `camera_view_min_frames`, `camera_view_min_confidence` trong config. Ba nhãn này là nhóm góc thô, không thay thế metadata góc đo/camera setup hoặc chứng minh view thấy đủ mọi tiêu chí scoring.

`reports/labeled_manifest.csv` chứa metadata đã bổ sung `camera_view`, `camera_view_confidence`, `camera_view_source=auto` và `camera_view_declared`. `summary.csv` cũng có nhãn và confidence; JSON có thêm cửa sổ frame, yaw ước lượng, phiếu theo frame và lý do. Manifest đầu vào được giữ nguyên.

## Điều kiện duyệt và ý nghĩa kết quả

`accepted → approved/`; `review` và `rejected → unapproved/`. File `.txt` có mã lỗi, lý do và giá trị/frame bằng chứng. `summary.csv` để lọc nhanh; JSON chứa metadata, cấu hình QC thực dùng, checksum nguồn/model/scorer, số đo và từng tiêu chí điểm. Pose giữ nguyên số frame, FPS và mask; không resample/xóa frame mất người.

Bộ kiểm tra:

- Giải mã toàn bộ clip; kiểm tra FPS, hướng khung, kích thước, số frame, nhòe và độ sáng từng frame. Độ nhòe đo sau resize chiều cao 360 px; nền vẫn có thể ảnh hưởng nên chỉ là cờ cần review.
- Dùng `ffprobe` đọc timestamp từng frame; thiếu timestamp, FPS không truy được, VFR/mất frame hoặc timebase không phù hợp scorer FPS cố định sẽ không được duyệt tự động. Không tự gán 30 FPS. Clip quay chậm nhưng được mã hóa CFR vẫn cần người xác nhận tốc độ thực.
- MediaPipe đa người + tracking hiện có: kiểm tra `invalid_for_scoring`, nhiều người, khớp bị che/ra biên, vùng pha đánh, world 3D cho luật cần 3D, tính tương thích số frame/FPS.
- Gọi scorer trên cả clip với `kick_window=none, fix_leg=True`; so nhãn bên với bên scorer tự phát hiện, kiểm tra `valid`, cảnh báo và tiêu chí NA. Không ép side để làm kết quả khớp nhãn.
- Kiểm tra metadata, mốc pha, checklist HLV, rò rỉ split và bản sao SHA256. Mọi bản sao trong batch đều vào review; trùng xuyên split là rejected.

**Approved nghĩa là đủ điều kiện QC đầu vào theo cấu hình và các xác nhận đã ghi.** Không đồng nghĩa đúng kỹ thuật hoặc điểm 10. Clip lỗi kỹ thuật nhưng quan sát rõ vẫn hữu ích để train chất lượng. Không đặt ngưỡng tổng điểm: workbook ghi điểm biểu hiện đá có hiện tượng bão hòa và không thể dùng như chuẩn độc lập. Nhãn chuyên gia 0/1/2/NA phải lưu riêng; công cụ này không tạo hoặc suy diễn nhãn đó và chưa nghiệm thu độ đầy đủ của bảng nhãn chuyên gia.

Checklist con người là điều kiện mặc định vì pose/scorer không đủ chứng minh đúng một lần thực hiện, đúng lớp, tốc độ thực, camera cố định hoặc nhìn đủ mọi tiêu chí. Các luật mở rộng chưa có rubric chuyên gia trong workbook vẫn cần HLV xây rubric riêng. Multi-camera được QC từng view, kiểm tra liên kết/split; công cụ chưa xác minh đồng bộ giữa camera, hợp nhất điểm hoặc dựng 3D.

## Hiệu chỉnh pilot

[config.json](config.json) chứa toàn bộ ngưỡng vận hành. Các ngưỡng visibility, tỷ lệ quan sát, nhòe, ánh sáng, FPS và độ phân giải là **đề xuất QC pilot**, không phải quy tắc scoring đã được HLV kiểm chứng. Kết quả chưa đạt ngưỡng này là `review`, không khẳng định video hỏng. Thử trên mẫu train đã được HLV xem, điều chỉnh và tăng `policy_version`; không hiệu chỉnh bằng test. Có thể copy config rồi truyền `--config`.

`require_human_checks=false` cho sàng lọc sơ bộ tự động và `require_ffprobe=false` để thiếu ffprobe chỉ sinh warning. Khi tắt, approved chỉ theo chính sách nới lỏng đã ghi trong JSON, không nên coi là nghiệm thu dataset sạch đầy đủ. Mặc định cả hai đều bật.

## Kiểm thử

```bash
python -m unittest discover -s dataset_verify/tests -v
```

Test bao gồm hai nhánh định tuyến với video tổng hợp, file hỏng, mốc pha, pose mất frame/khớp, world 3D, nhầm bên/người, NA, timestamp bất thường, leakage, bảo toàn nguồn và từ chối dùng lại output có kết quả. Video/pose tổng hợp trong test chỉ phục vụ kiểm thử phần mềm, không phải kết quả thực nghiệm Taekwondo.

Trích lại tài liệu nguồn khi đổi phiên bản:

```bash
python dataset_verify/export_workbook.py /duong/dan/workbook.xlsx
```

Đổi workbook không tự thay luật hoặc cấu hình QC; cần đối chiếu lại phiên bản scorer và các quy tắc trước khi sử dụng.
