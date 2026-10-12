# Chấm nhiều video

Chạy `chamdiembatch.bat` trong thư mục này.

1. Chọn thư mục chứa video. Bật **Gồm thư mục con** trước khi chọn hoặc nhấn **Quét lại** để cập nhật.
2. Kiểm tra động tác tự nhận theo tên video trong bảng. Nhấn **Mở video xem trước** để xem video đang chọn.
3. Để sửa, chọn dòng, chọn động tác và tùy chọn bên ra đòn, rồi nhấn **Áp dụng và xác nhận**. Có thể chọn nhiều dòng bằng Ctrl/Shift để áp dụng cùng động tác.
4. Dùng **Xác nhận dòng chọn** hoặc **Xác nhận toàn bộ lựa chọn** sau khi kiểm tra. Video chưa rõ động tác phải được chọn thủ công hoặc **bỏ qua**.
5. Nhấn **Bắt đầu batch**. Các video được chấm lần lượt theo thứ tự trong bảng. Video lỗi hoặc không chấm được được ghi nhận và ứng dụng tiếp tục video sau.

**Mở pipeline từng video** hiển thị khung xương và các bước xử lý. Sau khi có kết quả, cửa sổ giữ lại mặc định 3 giây rồi tự đóng để chuyển video tiếp theo. Có thể chỉnh thời gian từ 0 đến 60 giây. Tắt lựa chọn này để chạy batch không mở pipeline.

**Dừng sau video hiện tại** giữ kết quả video đang chạy và dừng các video còn lại. Chờ batch kết thúc trước khi đóng ứng dụng.

Mỗi lần chạy tạo một thư mục riêng trong `outputs/batch_scores/`, gồm:

- `summary.csv`: bảng điểm và trạng thái của toàn bộ danh sách, kể cả video bỏ qua hoặc chưa chấm do dừng.
- `manifest.json`: video, động tác và tùy chọn đã xác nhận khi bắt đầu.
- Thư mục từng video: `result.json`, `log.txt`, và `keyframes.png` khi có đủ kết quả chấm.
- `batch_errors.log` nếu có lỗi xử lý.

Chọn dòng để xem chi tiết/log; dùng **Mở kết quả video** hoặc **Mở kết quả batch** để mở thư mục tương ứng.

Khi batch kết thúc, bảng tổng kết tự mở và hiển thị tất cả video: điểm tổng, điểm chính xác, điểm biểu hiện, trạng thái và lý do/cảnh báo. Video lỗi hoặc không chấm được được tô đỏ. Chọn dòng để đọc đầy đủ lý do; nhấn **Xem bảng tổng kết** để mở lại bảng sau khi đóng.

Tên video chứa **đá chẻ** (hoặc `da_che`) được nhận diện là **đá bổ / axe**.

Ứng dụng batch dùng chung môi trường Python và luật chấm hiện có. `Cham diem.bat` và ứng dụng chấm từng video vẫn giữ nguyên.
