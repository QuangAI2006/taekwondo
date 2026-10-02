# Luật chấm điểm động tác ĐẤM

Mở rộng của luật chấm đá (`scripts/scoring_api.py`) sang động tác đấm, dùng **cùng khung 10 điểm**
mượn từ cấu trúc chấm Poomsae của World Taekwondo:

> **Tổng 10 = Chính xác 4 + Biểu hiện 6**

Giống luật đá gốc, đây là **luật viết tay** (công thức + ngưỡng), không phải model học máy. Ngưỡng điểm
**chưa được đối chiếu với điểm của trọng tài/chuyên gia** (xem phần Giới hạn).

| File | Vai trò |
|---|---|
| `scripts/punch_scoring.py` | Đo chỉ số + chấm điểm (`build_punch_metrics`, `score_punch`, ngưỡng trong `PunchRanges`) |
| `scripts/infer_punch_video.py` | Chạy từ video (1 file hoặc cả thư mục) — lớp bọc của `infer_action.py` |
| `scripts/action_pose.py` | Trích pose MediaPipe (2D + 3D) dùng chung cho mọi động tác |

Pipeline chung cho đá / đấm / động tác train thêm: xem [ACTIONS.md](ACTIONS.md).

## Cách chạy

```powershell
# 1 video, kèm ảnh 3 khung mốc để kiểm tra bằng mắt
python code/scripts/infer_punch_video.py "clip.mp4" --punch hook --keyframes kf.jpg --out result.json

# cả thư mục -> bảng CSV
python code/scripts/infer_punch_video.py "E:/martial arts/clips" --punch hook --csv punch_scores.csv
```

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `--punch` | `hook` / `straight` | Loại đấm (bắt buộc — không có model nhận dạng đấm) |
| `--guard` | `chin` (mặc định) / `waist` | Thủ ở cằm (đối kháng) hay nắm tay ở hông (Poomsae, Vovinam) |
| `--side` | `L` / `R` | Tay đấm; mặc định tự nhận |
| `--window` | `auto` (mặc định) / `on` / `off` | Cửa sổ pipeline: `auto` = mở khi chấm 1 video, tắt khi chấm thư mục |
| `--save_dashboard` | `file.png` | Lưu ảnh cửa sổ pipeline (dùng được cả khi `--window off`) |

### Cửa sổ pipeline (`scripts/pipeline_viewer.py`)

Khi chấm 1 video, một cửa sổ tự mở và cập nhật trực tiếp theo code:

- **Hàng trên** — sơ đồ 6 bước: Đọc video → MediaPipe Pose → Chọn người & cắt đoạn → Nhận dạng (chỉ với đá, với đấm hiện "bỏ qua") → Chia pha & đo chỉ số →
  Chấm điểm. Bước đang chạy tô cam, xong tô xanh, lỗi tô đỏ; dưới mỗi bước là kết quả của bước đó.
- **Trái** — video; lúc MediaPipe chạy thì vẽ khung xương từng frame, chấm xong thì phát lại chậm 2×
  (tay đấm tô đỏ, ghi pha Thủ / Ra đòn / Điểm chạm / Thu tay, dừng lại ở điểm chạm).
- **Phải trên** — tín hiệu dùng để chia pha (độ nâng khuỷu hoặc độ vươn tay + tốc độ cổ tay), tô màu pha ra đòn
  và thu tay, con trỏ chạy theo video.
- **Phải dưới** — điểm từng hạng mục; mục không đánh giá được hiện gạch chéo.

Kết quả dạng chữ vẫn in ra PowerShell như cũ; đóng cửa sổ để kết thúc lệnh.

Cần file `models/pose_landmarker_full.task` (MediaPipe).

## Quy trình

1. **Pose**: MediaPipe PoseLandmarker → 13 khớp 2D (như luật đá) **+ tọa độ 3D** (`pose_world_landmarks`).
   Dùng 3D cho góc khuỷu tay và độ xoay vai/hông, vì quay chính diện thì ảnh 2D làm méo các góc này rất nặng
   (thử trên clip: góc khuỷu 2D nhảy 20–150° trong một cú đấm, 3D liên tục và ổn định hơn nhiều).
2. **Chia pha** (làm mượt 5 frame trước):
   - *Điểm chạm* — đấm móc: lúc khuỷu tay nâng lên cao nhất (khi thủ khuỷu khép sát sườn); đấm thẳng: lúc tay
     vươn xa nhất. Chọn cú đấm có mức tăng lớn nhất trong 1s, lấy frame đầu tiên đến đích.
   - *Bắt đầu ra đòn* — lùi từ đỉnh tốc độ cổ tay tới khi tốc độ < 20% đỉnh. Nếu có pha lấy đà (hạ tay
     xuống trước khi vung), mốc này rơi vào cuối pha lấy đà.
   - *Về thế thủ* — tay trở lại cấu hình thủ (khuỷu hạ xuống / tay co về) trong 1.2s sau điểm chạm.
3. **Đo chỉ số → chấm điểm theo bảng dưới.**

Đơn vị khoảng cách là **chiều dài thân** (vai–hông) để không phụ thuộc chiều cao người và khoảng cách camera.

## Bảng điểm

Ký hiệu: `map(x, a, b)` = 0 khi x≤a, 1 khi x≥b (a>b thì càng nhỏ càng tốt);
`band(x, a, b, c, d)` = 1 khi b≤x≤c, giảm dần về 0 ở a và d.

### Chính xác (4 điểm)

**Đấm móc (`hook`)**

| Mã | Hạng mục | Điểm | Công thức |
|---|---|---:|---|
| HK_A1 | Hình dạng tay khi chạm | 1.6 | 0.40·band(góc khuỷu, 55, 80, 130, 155°) + 0.35·band(nâng cánh tay, 45, 75, 135, 160°) + 0.25·band(nắm tay cao hơn vai, −0.15, 0.15, 0.65, 0.90) |
| HK_A2 | Xoay hông – vai theo đòn | 1.2 | 0.60·band(xoay vai, 10, 40, 90, 130°) + 0.40·band(xoay hông, 5, 30, 90, 130°) |

**Đấm thẳng (`straight`)** — *chưa có video để kiểm thử*

| Mã | Hạng mục | Điểm | Công thức |
|---|---|---:|---|
| ST_A1 | Duỗi tay & đường đấm thẳng | 1.6 | 0.60·map(góc khuỷu, 120, 160°) + 0.40·map(độ thẳng quỹ đạo, 0.70, 0.93) |
| ST_A2 | Xoay vai & tầm đấm | 1.2 | 0.50·band(xoay vai, 5, 30, 75, 110°) + 0.50·band(nắm tay so với vai, −0.60, −0.40, 0.30, 0.50) |

**Chung**

| Mã | Hạng mục | Điểm | Công thức |
|---|---|---:|---|
| PU_A3 | Thu tay về thế thủ | 0.8 | map(tỉ lệ trở về, 0.30, 0.80) |
| PU_A4 | Tay thủ & thăng bằng | 0.4 | 0.50·map(tay còn lại cách vị trí thủ, 0.95, 0.55) + 0.25·map(thủ ban đầu, 0.90, 0.55) + 0.25·thăng bằng (nghiêng thân 35→15°, lắc hông 0.20→0.08) |

Vị trí thủ: `chin` = cằm, `waist` = hông cùng bên.

### Biểu hiện (6 điểm) — chung cho mọi loại đấm

| Mã | Hạng mục | Điểm | Công thức |
|---|---|---:|---|
| PX_E1 | Tốc độ | 2.0 | map(tốc độ cổ tay lớn nhất, 3.5, 8.0 thân/giây) |
| PX_E2 | Lực (xoay thân truyền lực) | 1.6 | map(tốc độ xoay vai lớn nhất, 80, 300 °/s) |
| PX_E3 | Dứt khoát | 1.4 | map(thời gian bắt đầu → điểm chạm, 0.90, 0.35 s) |
| PX_E4 | Độ mượt | 1.0 | map(2·v_đỉnh / tổng biến thiên tốc độ, 0.35, 0.80) |

Tỉ trọng 2.0 / 1.6 / 1.4 / 1.0 giữ nguyên như phần Biểu hiện của luật đá. Như tác giả luật đá đã lưu ý với
`power_raw`, "Lực" ở đây chỉ là **chỉ số động học** (tốc độ xoay vai), không phải lực đo được.

### Hạng mục không đánh giá được

Khi video không ghi lại đủ, hạng mục được đánh dấu `assessable = False` (không chấm 0), và điểm thành phần
được **quy đổi theo tỉ lệ** trên các hạng mục còn lại:

- Video kết thúc < 0.4s sau điểm chạm mà chưa thấy thu tay → bỏ PU_A3.
- Video bắt đầu giữa động tác (không thấy lúc tay đứng yên trước khi ra đòn) → bỏ xoay thân (HK_A2, PX_E2),
  thời gian ra đòn (PX_E3) và phần "thủ ban đầu" trong PU_A4.
- Không thấy cú đấm trọn vẹn → **không chấm** (`valid = False`).

## Kết quả thử trên `E:/martial arts/clips`

| Bộ clip | Kết quả |
|---|---|
| 24 clip "Dam moc phai" (1 võ sinh đai đen, đối kháng, quay chéo) | Chấm được 24/24; điểm 5.95–9.35, trung vị ≈ 8.6. Ảnh khung mốc kiểm tra bằng mắt khớp với video (clip 11, 17). Rep thấp nhất (clip 11) đúng là rep chậm (0.67s so với 0.2–0.5s). |
| Vovinam "test (1)" | Không chấm được: cú đấm ở 2–3 frame cuối video, người nghiêng hẳn → MediaPipe 3D không bắt được tay. |
| Vovinam "test (2)" | Chấm được nhưng kèm cảnh báo: người xoay ~120° quay lưng lại camera, số đo kém tin cậy. |

## Giới hạn

1. **Chưa đối chiếu với trọng tài.** Ngưỡng đặt từ hiểu biết kỹ thuật + phân bố số đo của **một** võ sinh.
   Điểm phù hợp để so sánh tương đối / theo dõi tiến bộ, chưa thay được điểm chấm chính thức.
2. **Đấm thẳng chưa được kiểm thử** trên video thật.
3. **Góc quay**: tốt nhất là chính diện hoặc chéo ~45°, thấy toàn thân, mỗi video một cú đấm và có ~0.3s thủ
   trước và ~0.5s sau cú đấm. Quay nghiêng hẳn hoặc quay lưng thì MediaPipe dễ sai/nhầm trái–phải.
4. Tọa độ 3D của MediaPipe là **ước lượng từ một camera**; góc khuỷu 3D có xu hướng lớn hơn thực tế
   (clip 17: nhìn khung hình ~90° nhưng đo ~115°) — ngưỡng HK_A1 đã nới ra để tính đến điều này.
5. Đấm móc trong clip là kiểu có lấy đà (hạ tay rồi vung lên). Luật **không trừ điểm** pha lấy đà; nếu
   muốn chấm theo kiểu móc ngắn đối kháng (không lấy đà) cần thêm hạng mục "lộ ý đồ".

## Hiệu chỉnh

- Sửa ngưỡng trong `PunchRanges` (`scripts/punch_scoring.py`) — mọi con số trong bảng trên đều nằm ở đó.
- Để hiệu chỉnh nghiêm túc: thu video nhiều người/nhiều trình độ, nhờ ≥2 chuyên gia chấm cùng thang 4+6, rồi
  đo tương quan / MAE / Bland–Altman như tác giả đã làm với luật đá
  (`outputs/expert_agreement/expert_agreement_report.md`).
