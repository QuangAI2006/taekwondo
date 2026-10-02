# Luật chấm các động tác còn lại

Mở rộng luật chấm đá của tác giả (`scripts/scoring_api.py`) và luật đấm ([PUNCH_SCORING.md](PUNCH_SCORING.md)) sang
toàn bộ động tác võ trong 11 video bài giảng "Tự học Taekwondo tại nhà" (VinKungfu). Mọi luật giữ **cùng khung
10 điểm của tác giả**:

> **Tổng 10 = Chính xác 4 (A1 1.6 · A2 1.2 · A3 0.8 · A4 0.4) + Biểu hiện 6 (E1 2.0 · E2 1.6 · E3 1.4 · E4 1.0)**

- A1 = điểm kỹ thuật cốt lõi của động tác, A2 = điểm kỹ thuật thứ hai, A3 = thu về / trở về, A4 = thăng bằng & tay
  thủ — đúng thứ tự ưu tiên trong luật đá của tác giả (FR/RH/SD/AX_A1–A4).
- Điểm mục = điểm tối đa × `map(chỉ số, ngưỡng dưới, ngưỡng trên)` (0 khi chưa tới ngưỡng dưới, 1 khi qua ngưỡng
  trên; ngưỡng dưới > ngưỡng trên nghĩa là "càng nhỏ càng tốt"); `band(x, a, b, c, d)` = 1 trong [b, c].
- Hạng mục không đo được (video bắt đầu/kết thúc giữa động tác) → **không chấm**, điểm thành phần quy đổi theo tỉ lệ
  (như luật đấm).
- Đây là **luật viết tay**, ngưỡng đặt từ kỹ thuật chuẩn, **chưa đối chiếu với trọng tài** (xem Giới hạn).

| File (`scripts/`) | Động tác |
|---|---|
| `scorers_kicks_extra.py` | đá sau, đá móc gập, đá đẩy, đá ngang thấp, đá vòng cầu thấp — dùng **chính chỉ số và phần Biểu hiện của tác giả** |
| `scorers_hands.py` + `hand_scoring.py` | đỡ cao, gạt thấp, chém cạnh 2 tay, chém đao 1 tay, chọc mũi tay — cách đo của luật đấm (3D) |
| `scorers_stances.py` | trung bình tấn, tấn trước dài |
| `scorers_combos.py` | gạt thấp – đấm thẳng, gạt thấp – đá trước, đấm trong trung bình tấn — ghép từ các luật trên |
| `scorers_drills.py` | nhảy tiến – lùi, thủ thế thụt người, thủ co gối |
| `score_segments.py` | chấm từng lần thực hiện trong các đoạn 30 s của video bài giảng |

`infer_action.py` tự nạp mọi `scorers_*.py`, nên các luật mới dùng được ngay: `--action back`, `--action high_block`, ...

## Bảng ánh xạ bài tập → luật

| Bài trong video | Video | Luật (`--action`) | Nguồn luật |
|---|---|---|---|
| Đá trước cao; Nhảy lùi – đá trước; Nhảy (đá) hất chân; Co đá song phi | #2, #4, #5, #7, #9 | `front` | tác giả |
| Đá vòng cầu; Xoay cổ (chân) đá vòng cầu | #1, #2, #6, #11 | `roundhouse` | tác giả |
| Đá ngang; Đá ngang thấp – cao | #6, #8, #10 | `side` | tác giả |
| Đấm móc; Đấm vòng | #2, #3, #4 | `hook` | luật đấm |
| Đấm trái – phải (thẳng) | #6 | `straight` | luật đấm |
| Đá sau | #3, #5 | `back` | mới — chỉ số tác giả |
| Đá gập vòng | #3, #4 | `hook_kick` | mới — chỉ số tác giả |
| Đá ức ngang; Đá ức chân; Ngả người – đá ức chân | #4, #5, #8 | `push` | mới — chỉ số tác giả |
| Đá cạnh thấp | #7 | `side_low` | tác giả, đổi mục độ cao |
| Đá vòng cầu thấp | #7 | `roundhouse_low` | tác giả, đổi mục độ cao |
| Đỡ cao | #2 | `high_block` | mới |
| Bước gạt thấp | #3, #6 | `low_block` | mới |
| Bước mở chân – chém cạnh 2 tay | #2 | `knife_block` | mới |
| Bước mở chân – chém đao 1 tay | #2 | `knife_strike` | mới |
| Bước chọc mũi tay | #10 | `spear_hand` | mới |
| Gạt thấp – đấm thẳng | #7 | `block_punch` | ghép |
| Gạt thấp – đá trước | #10 | `block_kick` | ghép |
| Rộng chân – đấm thẳng | #8 | `horse_punch` | ghép |
| Đứng trung bình tấn | #9 | `horse_stance` | mới |
| Tấn trước dài | #11 | `front_stance` | mới |
| Nhảy tiến – lùi | #9 | `footwork` | mới |
| Thủ thế thụt người | #11 | `duck` | mới |
| Thủ co gối | #11 | `knee_chamber` | mới |

Không chấm: khởi động, giãn cơ, thể lực (jumping jack, ép/xoạc chân, hít đất, chống chữ V, đứng ngồi, chạy nâng
gối...). "Đấm móc (phải)" ở video #1 đã chấm trước đó (24 clip "Dam moc phai", xem PUNCH_SCORING.md).

## 1. Đá — `scorers_kicks_extra.py`

Làm đúng như `_score_kick` của tác giả: cắt đoạn bằng `action_window(..., "ankle")`, đo bằng
`build_metrics_from_sequence()`, phần **Biểu hiện EX_E1–E4 chép nguyên** công thức và ngưỡng `ScoreRanges` của tác giả.
A3 (thu chân, `map(recovery_ratio, 0.25, 0.75)`) và A4 (trọng tâm & tay thủ) cũng là công thức của tác giả.
Chỉ A1/A2 là mới, và dùng lại ngưỡng của tác giả khi có thể.

| Mã | Hạng mục | Điểm | Công thức |
|---|---|---:|---|
| BK_A1 | Đẩy gót thẳng ra sau | 1.6 | 0.6·map(độ thẳng, 0.72, 0.96) + 0.4·map(−cos(hướng mặt, hướng đá), 0, 0.6) — hướng đo bằng 3D |
| BK_A2 | Độ cao & duỗi gối | 1.2 | 0.55·map(độ cao, 0.15, 0.65) + 0.45·map(góc gối lớn nhất, 135, 175°) (= FR_A2 của tác giả) |
| HKK_A1 | Duỗi chân rồi gập móc gót | 1.6 | 0.5·map(góc gối lớn nhất, 135, 175°) + 0.5·map(gập gối sau đỉnh, 30, 90°) |
| HKK_A2 | Độ cao & quỹ đạo vòng | 1.2 | 0.7·map(độ cao, 0.15, 0.65) + 0.3·clip((0.9 − độ thẳng)/0.4) ("arc" của tác giả) |
| PK_A1 | Co gối cao rồi đẩy thẳng | 1.6 | 0.6·map(nâng gối, 0.04, 0.12) + 0.4·map(độ thẳng, 0.72, 0.96) (= FR_A1 của tác giả) |
| PK_A2 | Chân duỗi ngang khi đẩy | 1.2 | 0.45·map(góc gối, 135, 175°) + 0.55·map(\|cẳng chân so với phương ngang\|, 45, 15°) (3D) |
| *_A3 | Thu chân về | 0.8 | map(tỉ lệ thu về, 0.25, 0.75) — tác giả |
| *_A4 | Trọng tâm & tay thủ | 0.4 | 0.65·map(ổn định, 0.35, 0.80) + 0.35·(1 − tay thủ hạ/0.25) — tác giả |
| EX_E1–E4 | Tốc độ / Lực / Độ cao / Độ mượt | 6.0 | nguyên luật tác giả |

**Đá thấp** (`side_low`, `roundhouse_low`): chạy nguyên luật `side` / `roundhouse` của tác giả, chỉ thay phần độ cao
(SD_A2/RH_A2 và EX_E3) bằng `band(cổ chân so với hông, −1.6, −1.0, 0.05, 0.45)` — đòn đá thấp cố ý không bị trừ điểm vì
"không cao".

**Đầu vào cho luật đá** (mọi luật đá, cả của tác giả) — hai tùy chọn, không đổi công thức hay ngưỡng nào:

| Tùy chọn | Mặc định `infer_action.py` | `score_segments.py` | Lý do |
|---|---|---|---|
| `kick_window` | `ankle` — đoạn cắt của tác giả, như `predict_video.py` | `none` — cả clip của 1 lần đá | Trên đoạn cắt, chân chưa kịp thu về nên A3 (thu chân) gần như luôn = 0. Kiểm trên 240 mẫu public của tác giả: trung vị A3 = 0.43/0.37/0.58 (front/roundhouse/axe) trên cả clip, **0.00** trên đoạn cắt. Luật lấy 10% frame đầu/cuối làm mốc đứng yên nên được viết cho cả clip. |
| `fix_leg` | tắt | bật | Hàm chọn chân của tác giả lấy "cổ chân xa tâm hông nhất", nhưng chân trụ đứng thẳng cũng xa hông ngang chân đá duỗi thẳng → hay chọn nhầm chân trụ (thử trên video: "độ cao" −5, điểm ~1.5 cho mọi cú đá chân trái). Bật: chọn chân có cổ chân nâng cao hơn cổ chân kia, rồi đặt cổ chân trụ trùng tâm hông để hàm của tác giả lấy đúng chân. Cổ chân trụ không dùng trong hạng mục nào. |

`infer_action.py` có thêm `--kick_window none --fix_leg` để chấm 1 video theo cùng cách.

## 2. Kỹ thuật tay — `scorers_hands.py`

Đo như luật đấm: tọa độ 3D, làm mượt 5 frame, chuẩn hóa theo chiều dài thân. "Điểm đến" tìm bằng tín hiệu riêng của
từng kỹ thuật (lần tăng lớn nhất trong 1 s, như điểm chạm của cú đấm):

| Kỹ thuật | Tín hiệu | A1 (1.6) | A2 (1.2) |
|---|---|---|---|
| `high_block` đỡ cao | cổ tay cao hơn mũi | 0.35·band(cổ tay trên mũi, −0.05, 0.10, 0.55, 0.80) + 0.25·band(khuỷu trên vai) + 0.20·band(góc khuỷu, 55, 85, 145, 170°) + 0.20·map(\|cẳng tay so với ngang\|, 50, 20°) | 0.5·map(biên độ nâng tay, 0.5, 1.1 thân) + 0.5·tay còn lại ở cằm/hông |
| `low_block` gạt thấp | cổ tay thấp hơn vai | 0.5·map(góc khuỷu, 125, 160°) + 0.5·band(cổ tay so với hông, −0.85, −0.60, −0.05, 0.15) | 0.35·tay lấy đà cao ở vai + 0.25·lấy đà từ bên kia thân + 0.40·tay còn lại ở hông/cằm |
| `knife_block` chém cạnh 2 tay | cổ tay vươn ngang | 0.35·band(góc khuỷu tay trước, 95, 120, 170, 181°) + 0.30·band(cổ tay ngang vai) + 0.35·map(tay sau cách giữa ngực, 0.70, 0.25) | 0.5·lấy đà từ vai đối diện + 0.5·bước mở tấn (rộng ≥ 1.8 vai, gối ≤ 145°) |
| `knife_strike` chém đao 1 tay | cổ tay vươn ngang | 0.40·map(góc khuỷu, 125, 160°) + 0.30·band(cổ tay ngang vai/cổ) + 0.30·map(độ vươn, 0.70, 0.90) | 0.4·lấy đà + 0.3·tay kia kéo về hông + 0.3·bước mở tấn |
| `spear_hand` chọc mũi tay | độ vươn (tay ở tầm ngực trở lên) | 0.6·map(góc khuỷu, 120, 160°) + 0.4·map(độ thẳng quỹ đạo, 0.70, 0.93) (= ST_A1 luật đấm) | 0.5·band(tầm thượng vị–mặt) + 0.5·max(tay kia dưới khuỷu, tay kia ở hông) |

Tín hiệu "cổ tay vươn ngang" = cổ tay cách giữa vai theo trục vai về phía cùng bên (× rộng vai), trừ khi tay thấp hơn
vai quá 0.3 thân. Không dùng độ vươn thuần vì tay buông xuôi bên hông cũng duỗi thẳng.

A3 = thu tay về `map(tỉ lệ trở về, 0.30, 0.80)` — riêng gạt thấp A3 = **dừng dứt khoát ở điểm gạt**
`map(cổ tay trôi trong 0.25 s, 0.30, 0.10 thân)`, vì bài bước gạt giữ tay ở dưới rồi mới lấy đà cho lần sau.
A4 = thân thẳng & hông không lắc (như PU_A4), cộng độ mở tấn với động tác có bước. Biểu hiện HX_E1–E4 **dùng đúng ngưỡng của luật đấm** (`PunchRanges`): tốc độ cổ tay 3.5→8.0 thân/s,
tốc độ xoay vai 80→300°/s, thời gian ra đòn 0.90→0.35 s, độ mượt 0.35→0.80.

## 3. Tấn — `scorers_stances.py`

Đo trong đoạn đứng yên dài nhất (tốc độ khớp trung bình < 0.6 thân/s), mỗi cửa sổ 3 s là một lần chấm. Không chấm
(thay vì cho điểm thấp) khi: không có đoạn đứng yên ≥ 0.5 s; gối gần thẳng (đã ra khỏi tấn); trung bình tấn mà người
quay nghiêng (rộng vai/dài thân 2D < 0.40 — hai chân chồng theo chiều sâu, MediaPipe đo sai).

Góc quay: quay chính diện → góc gối 3D (2D bị co ngắn); quay nghiêng (tấn trước thường quay từ bên cạnh) → góc gối,
độ nghiêng thân và chiều dài tấn đo bằng 2D (chính xác trong mặt phẳng dọc; 3D sai độ sâu), chiều dài tấn tính theo
chiều dài thân `band(…, 0.7, 1.05, 2.1, 2.5)`, và FS_A4 (bề ngang 2 chân) không chấm.

| Mã | Trung bình tấn (`horse_stance`) | Điểm | Tấn trước dài (`front_stance`) |
|---|---|---:|---|
| A1 | band(góc 2 gối, 90, 105, 145, 165°) | 1.6 | 0.6·band(gối trước, 80, 95, 135, 155°) + 0.4·map(gối sau, 145, 168°) |
| A2 | 0.6·band(2 chân rộng, 1.3, 1.7, 2.9, 3.5 vai) + 0.4·map(gối/cổ chân, 0.65, 0.85) | 1.2 | band(dài tấn theo hướng mặt, 1.0, 1.5, 3.0, 3.6 vai) |
| A3 | map(thân nghiêng, 20, 7°) | 0.8 | như bên trái |
| A4 | map(\|gối trái − gối phải\|, 25, 8°) | 0.4 | band(bề ngang 2 chân, 0.1, 0.3, 1.2, 1.6 vai) |
| E1 | map(lắc tâm hông, 0.06, 0.015 thân) — độ vững | 2.0 | như bên trái |
| E2 | map(chiều cao hông / dài chân, 0.95, 0.80) — trọng tâm thấp | 1.6 | map(…, 0.97, 0.85) |
| E3 | map(tỉ lệ thời gian đạt tấn, 0.4, 0.9) | 1.4 | như bên trái |
| E4 | map(dao động góc thân, 6, 1.5°) | 1.0 | như bên trái |

## 4. Đòn phối hợp — `scorers_combos.py`

Không đặt ngưỡng mới, chỉ ghép luật đã có (mỗi phần đã là thang 4 + 6):

- `block_punch` = 0.5 × `low_block` (tay gạt, mặc định trái) + 0.5 × `straight` (tay kia, thủ ở hông).
- `block_kick` = 0.5 × `low_block` + 0.5 × `front` của tác giả.
- `horse_punch` ("Rộng chân – đấm thẳng") = luật `straight` (thủ ở hông), mục PU_A4 thành HP_A4 = 0.5·PU_A4 +
  0.5·mở tấn (rộng `map(1.1, 1.8 vai)`, gối chùng `map(172, 145°)`) — bài là tấn rộng, không phải trung bình tấn sâu.

Nếu phần sau tới đích trước phần đầu → cảnh báo "Thứ tự chưa đúng".

## 5. Bộ pháp & phòng thủ — `scorers_drills.py`

| Mã | `footwork` nhảy tiến – lùi | `duck` thủ thế thụt người | `knee_chamber` thủ co gối |
|---|---|---|---|
| A1 1.6 | giữ tấn: 0.5·band(2 chân hẹp nhất, 0.6, 1.0, 2.6, 3.2 vai) + 0.5·map(gối, 178, 160°) | hạ bằng gối: 0.5·map(đầu hạ, 0.15, 0.50 thân) + 0.5·map(gối, 170, 125°) | 0.6·map(gối so với hông, −0.70, −0.05 thân) + 0.4·map(góc gối, 120, 70°) |
| A2 1.2 | tay thủ: map(độ cao trung bình 2 cổ tay so với vai, −0.90, −0.35 thân) | tay thủ (như trái) | tay thủ (như trái) |
| A3 0.8 | thân thẳng: map(nghiêng, 25, 10°) | bật về thế thủ: map(trở về, 0.3, 0.8) | hạ chân về: map(trở về, 0.3, 0.8) |
| A4 0.4 | về lại độ rộng tấn: map(lệch, 50%, 15%) | lưng không gập: map(nghiêng, 55, 25°) | thăng bằng: nghiêng thân + lắc hông |
| E1 2.0 | tốc độ hông 0.8→3.0 thân/s (2D) | tốc độ hạ đầu 0.8→2.5 | tốc độ gối 1.5→4.5 (3D) |
| E2 1.6 | biên độ bước 0.10→0.45 thân | tốc độ bật lên 0.6→2.0 | biên độ nâng gối 0.40→0.90 thân |
| E3 1.4 | thời gian bước 0.90→0.35 s | thời gian hạ 0.80→0.30 s | thời gian nâng 0.70→0.25 s |
| E4 1.0 | độ mượt 0.35→0.80 | như trái | như trái |

## Chấm video bài giảng

```powershell
cd "E:\martial arts\tkd\TKD-Kick3_Zenodo_Materials_20260526_153619"
python code/scripts/score_segments.py --segments code/tutorial_segments.csv --root "E:/martial arts" `
       --out outputs/tutorial_scores --pose_cache outputs/tutorial_scores/pose_cache --workers 8 --transcode
```

- `code/tutorial_segments.csv`: 82 đoạn tập (41 bài × 2 hiệp), mốc lấy từ chữ tên bài trên video; mỗi đoạn 27 s
  (bỏ 2–3 s cuối có thể đã sang phần nghỉ, có ảnh nhỏ bài kế tiếp).
- Mỗi đoạn: cắt sang H.264 bằng ffmpeg (`--transcode`; video YouTube là AV1, OpenCV giải mã ~15 frame/s) → MediaPipe →
  tách từng lần thực hiện → chấm từng lần. Tách lần: đá = đỉnh chênh độ cao 2 cổ chân; đỡ/gạt/chém/chọc = đỉnh tín hiệu
  của chính kỹ thuật; đấm xen kẽ = đỉnh |độ vươn tay trái − phải| (tay ra đòn = tay vươn xa hơn); đòn phối hợp = mỗi
  lần gạt; tấn = cửa sổ 3 s.
- Kết quả: `reps.csv` (từng lần), `summary.csv` (trung vị theo bài và video), `results.json` (kèm chỉ số thô từng lần).
  `--pose_cache` lưu pose nên chạy lại (vd đổi ngưỡng) không cần MediaPipe; đoạn ngắn hơn dùng lại pose của đoạn dài.
- `--author_window`: chấm đá trên đoạn cắt của tác giả (như `infer_action.py` mặc định).

### Kết quả (HLV trong video, 2 hiệp gộp lại)

Trung vị điểm các lần thực hiện hợp lệ. Hai hiệp của cùng một bài cho trung vị chênh nhau không quá 0.5 điểm ở 38/41 bài
(lệch nhất: "Đá ngang (phải)" video #8 8.23 / 7.47, "Bước chọc mũi tay" 5.70 / 5.07, "Nhảy hất chân" 7.16 / 7.70).

| Video | Bài | Luật | Lần hợp lệ | Điểm /10 | Chính xác /4 | Biểu hiện /6 |
|---|---|---|---:|---:|---:|---:|
| #1 | Xoay cổ chân đá (phải) | `roundhouse` | 21/21 | 6.18 | 1.36 | 4.78 |
| #2 | Đỡ cao (trái) | `high_block` | 25/25 | 7.59 | 3.19 | 4.32 |
| #2 | Đá trước cao (phải) | `front` | 31/31 | 8.51 | 3.51 | 5.00 |
| #2 | Đấm móc (trái) | `hook` | 36/36 | 7.97 | 3.32 | 4.76 |
| #2 | Xoay cổ đá vòng cầu (trái) | `roundhouse` | 24/24 | 6.41 | 1.41 | 5.00 |
| #2 | Bước mở chân – chém cạnh 2 tay | `knife_block` | 26/26 | 8.37 | 3.09 | 5.27 |
| #2 | Bước mở chân – chém đao 1 tay | `knife_strike` | 25/25 | 8.27 | 3.39 | 4.89 |
| #3 | Đấm vòng (phải) | `hook` | 45/46 | 8.42 | 3.63 | 5.03 |
| #3 | Đá sau (phải) | `back` | 32/32 | 8.54 | 3.54 | 5.00 |
| #3 | Bước gạt thấp (trái) | `low_block` | 20/25 | 7.42 | 3.39 | 4.05 |
| #3 | Đá gập vòng (phải) | `hook_kick` | 23/23 | 7.28 | 2.28 | 5.00 |
| #3 | Đấm vòng (trái) | `hook` | 58/60 | 8.98 | 3.60 | 5.34 |
| #4 | Đá gập vòng (trái) | `hook_kick` | 24/24 | 7.47 | 2.47 | 5.00 |
| #4 | Đá ức ngang (phải) | `push` | 27/27 | 7.39 | 2.39 | 5.00 |
| #4 | Đấm móc (trái) | `hook` | 57/57 | 7.38 | 3.08 | 4.49 |
| #4 | Nhảy đá hất chân (phải) | `front` | 24/24 | 7.16 | 2.16 | 5.00 |
| #5 | Đá sau (trái) | `back` | 28/28 | 8.43 | 3.43 | 5.00 |
| #5 | Nhảy hất chân (trái) | `front` | 31/31 | 7.19 | 2.55 | 5.00 |
| #5 | Đá ức chân (trái) | `push` | 33/33 | 6.65 | 1.65 | 5.00 |
| #6 | Đá vòng cầu (phải) | `roundhouse` | 21/21 | 7.17 | 2.17 | 5.00 |
| #6 | Đấm trái – phải (thẳng) | `straight` | 25/31 | 7.31 | 2.84 | 4.47 |
| #6 | Bước gạt thấp (phải) | `low_block` | 19/29 | 7.84 | 2.88 | 4.59 |
| #6 | Đá ngang (phải) | `side` | 23/23 | 7.14 | 2.14 | 5.00 |
| #7 | Gạt thấp (trái) – đấm thẳng (phải) | `block_punch` | 20/31 | 7.01 | 2.39 | 4.44 |
| #7 | Đá vòng cầu thấp (phải) | `roundhouse_low` | 28/28 | 6.98 | 1.98 | 5.00 |
| #7 | Đá cạnh thấp (phải) | `side_low` | 25/25 | 7.06 | 2.16 | 5.00 |
| #7 | Co đá song phi (phải) | `front` | 26/26 | 8.15 | 3.15 | 5.00 |
| #8 | Rộng chân – đấm thẳng | `horse_punch` | 30/39 | 6.67 | 3.26 | 3.95 |
| #8 | Đá ngang (phải) | `side` | 20/20 | 7.83 | 2.83 | 5.00 |
| #8 | Ngả người – đá ức chân (trái) | `push` | 26/26 | 7.67 | 2.67 | 5.00 |
| #9 | Đứng trung bình tấn | `horse_stance` | 11/18 | 9.84 | 3.86 | 6.00 |
| #9 | Nhảy tiến – lùi | `footwork` | 58/58 | 7.02 | 3.06 | 4.00 |
| #9 | Nhảy lùi – đá trước (phải) | `front` | 17/17 | 7.96 | 2.96 | 5.00 |
| #10 | Đá ngang thấp – cao (phải) | `side` | 22/22 | 8.54 | 3.54 | 5.00 |
| #10 | Bước chọc mũi tay (trái) | `spear_hand` | 14/24 | 5.56 | 2.78 | 3.17 |
| #10 | Gạt thấp (trái) – đá trước (phải) | `block_kick` | 26/26 | 6.80 | 2.40 | 4.39 |
| #11 | Đá vòng cầu (phải) | `roundhouse` | 24/24 | 7.89 | 2.89 | 5.00 |
| #11 | Thủ thế thụt người | `duck` | 38/38 | 9.25 | 3.48 | 5.81 |
| #11 | Tấn trước dài (trái) | `front_stance` | 9/18 | 6.24 | 1.09 | 4.63 |
| #11 | Tấn trước dài (phải) | `front_stance` | 18/18 | 9.36 | 3.55 | 5.85 |
| #11 | Thủ co gối (trái) | `knee_chamber` | 29/29 | 9.43 | 3.81 | 5.53 |

Đọc kết quả:
- **Đá**: Biểu hiện luôn 5.00 (đặc điểm của luật tác giả, xem Giới hạn 2), điểm khác nhau chỉ ở phần Chính xác. Đá
  vòng cầu quay chính diện mất nhiều ở RH_A1 (lệch mặt phẳng vung chân, đo bằng 2D) → 6.2–6.4; quay chéo/nghiêng 7.2–7.9.
- **Kỹ thuật tay**: mất điểm chủ yếu ở "Lực (xoay thân)" — tốc độ xoay vai thấp so với ngưỡng của luật đấm
  (80→300°/s); các bài đỡ/gạt trong video làm tại chỗ, ít xoay hông.
- **Chọc mũi tay** thấp vì nửa đầu bài HLV làm chậm (tốc độ HX_E1 = 0 ở các lần đó); các lần làm nhanh đạt 8.3–9.2.
- **Tấn trước dài (trái)**: HLV vừa đi vừa xoay người, chỉ 9/18 cửa sổ có đoạn đứng yên đủ lâu → kém tin cậy.
- **Trung bình tấn**: chỉ chấm nửa đầu bài (quay chính diện); nửa sau HLV quay nghiêng nên không chấm.
- Tấn, thụt người, co gối được 9.2–9.8 một phần vì ngưỡng các luật này rộng — cần trọng tài đối chiếu.

## Giới hạn

1. **Chưa đối chiếu với trọng tài.** Ngưỡng của luật mới đặt từ kỹ thuật chuẩn; người tập trong video là HLV đai đen
   nên điểm cao là điều mong đợi, nhưng chưa chứng minh được điểm khớp với chuyên gia.
2. **Phần Biểu hiện của luật đá tác giả gần như không phân biệt**: chạy lại trên chính dữ liệu public của tác giả
   (180 mẫu train), EX_E1–E3 luôn đạt tối đa (2.0/1.6/1.4) và EX_E4 luôn ≈ 0 (độ mượt tính bằng px/s², rất nhỏ) — nên
   Biểu hiện ≈ 5.0/6 với mọi cú đá; điểm đá chỉ phân biệt ở phần Chính xác. Các luật đá mới giữ nguyên đặc điểm này để
   "đúng luật tác giả"; các luật tay/tấn/bộ pháp dùng thang đo vật lý của luật đấm nên phân biệt được.
3. Luật đá của tác giả tìm "đỉnh đá" bằng khoảng cách cổ chân–hông lớn nhất: chân duỗi thẳng lúc đứng cũng xa hông,
   nên thỉnh thoảng đỉnh rơi vào lúc đứng thủ (thấy ở vài lần đá đẩy) — các lần đó điểm thấp bất thường, trung vị ít bị
   ảnh hưởng.
4. Chuẩn hóa 2D của tác giả theo **bề rộng vai/hông**: khi quay nghiêng hẳn (vai chồng nhau) số đo độ cao/tốc độ bị
   phóng to. MediaPipe 3D sai độ sâu khi quay nghiêng — đã xử lý cho tấn, chưa xử lý cho kỹ thuật tay.
5. Tách lần thực hiện tự động có thể bắt nhầm cử động chuẩn bị; lần nào không thấy đủ động tác thì `valid = 0` và
   không tính vào trung vị (tỉ lệ hợp lệ thấp nhất: gạt thấp phải 19/29, đấm trong tấn rộng 30/39, chọc mũi tay 14/24).
6. Luật **không kiểm tra đúng tên động tác** (`--action` ép kiểu): vd một cú đấm móc chấm bằng luật đỡ cao vẫn ra 8.1
   vì tay cũng lên cao. Muốn biết người tập làm đúng bài cần model nhận dạng (train thêm, xem ACTIONS.md).
7. Video chỉ có **một người** (HLV), một góc quay mỗi bài: kết quả dùng để kiểm tra luật chạy hợp lý trên video thật,
   không phải để đánh giá độ chính xác của luật.
