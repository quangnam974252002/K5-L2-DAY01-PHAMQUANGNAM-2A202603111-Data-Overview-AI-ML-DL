# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
`class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về cả bức ảnh. Lý do là dựa vào ảnh biểu đồ sinh ra từ record này, nó đánh giá độ tin cậy của mô hình và để đánh giá được độ tin cậy cho nhiều vật thể của mô hình thì nó phải nhìn cả bức ảnh, nên nó record này ko nói về "một vật thể riêng".

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Là đi kèm taxonomy/checkpoint. 

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Class id là để phân biệt các version của từng record, class name là để người đọc dễ dàng nhận dạng và taxonomy_name là để biết nguồn file đi kèm.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần quy định quy tắc chọn chủ thể chính (dominant subject) để người gán nhãn chỉ chọn một lớp duy nhất cho mỗi ảnh.

6. Vì sao không được lấy `score` của model làm ground truth?
vì score chỉ thể hiện mức độ tự tin của model đối với dự đoán, chứ không chứng minh dự đoán đó đúng hay không của người dán.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…

class_name`=bus ; `score`=0.912557 ; `bbox_xyxy`=[93.17, 187.95, 223.01, 320.91] ; `bbox_width`=129.84 ; `bbox_height`=136.96

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Sử dụng record của câu 1 trên. Góc trên trái ở xy(93.17,187.95), góc dưới-phải ở xy(223.01,320.91), rộng là 129.84 ; cao là 136.96

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Ví dụ có 2 ngưỡng là 0.20 và 0.60 và có 100 predictions.
Ngưỡng 0.20 sẽ có trung bình 80 predictions và ngưỡng 0.60 sẽ có 40 predictions
Vì ngưỡng càng thấp, số predictions đưa ra bởi model sẽ càng cao vì score
mà model đưa ra có thể được giao động từ 0.20 đên 1.00.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Người reviewer sẽ phải kiểm tra predictions nhiều hơn do hạ ngưỡng và sẽ phải kiểm tra và phát hiện nhiều predictions sai, thừa, thiếu hoặc cả trên. Đồng thời phải sửa nhãn hoặc loại bỏ prediction đấy.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Quy tắc bõ:
1/Vẽ một bounding box hình chữ nhật bao quanh mỗi đối tượng thuộc class được yêu cầu.
2/Box phải ôm sát ranh giới bên ngoài của vật thể, không để khoảng trống quá lớn.
3/Không vẽ box cho bóng, phản chiếu hoặc các vật thể không thuộc danh sách class được yêu cầu.
4/Nếu vật thể bị che khuất một phần nhưng vẫn nhận diện được, vẽ box theo phần quy định trong guideline (ví dụ: phạm vi toàn bộ vật thể ước lượng hoặc phần nhìn thấy).
5/Nếu vật thể bị cắt bởi mép ảnh, vẽ box đến sát mép ảnh, không tự mở rộng ra ngoài khung hình.


6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trong ảnh kitchen, có cái 1 cái oven là bị che khuất. 
Tôi sẽ viết guideline. Vì vật này có thể này là 1 "oven" mà model không thể nhận dạng do bị 1 người che khuất, thế nên tôi sẽ thêm quy tắc và điều kiện vào guideline mặc dù có thể nó sai.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
`instance_id`=traffic-001 ; `class_name`=bus ; `score`=0.925745 ; số điểm polygon=`polygon_point_count`=120 ; 3–5 điểm đầu của `polygon_xy`=
[148.0,189.0],[147.0,190.0],[145.0,190.0],[143.0,192.0],[142.0,192.0]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon cho thêm chính xác thông tin về vật thể và hình dạng của vật thể ở mức đơn mầu và dễ giúp model nhận dạng vật thể hơn mà tránh được vùng không phải là vật thể 

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance id dùng để vừa nhận dạng vật và vừa đánh dấu số. Instance ID là giống với việc mình tạo 1 class trong code thì mỗi lần tạo new class là 1 new instance mà sẽ được lưu lại trong bộ nhớ.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Vẽ mask theo đường viền thực tế của đối tượng, không vẽ theo hình chữ nhật như Bounding Box.
Mask phải bao phủ toàn bộ vùng pixel nhìn thấy thuộc về đối tượng.
Cố gắng giữ chính xác các chi tiết như tai, chân và đuôi nếu chúng nhìn thấy rõ.
Với phần vật thể bị che khuất, chỉ đánh dấu phần nhìn thấy được, không tự suy đoán và tô phần bị che.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Có vật là dining table là đang biên mờ, tôi sẽ dùng guideline. Vì vấn đề này có thể được giải quyết bằng viết guideline chặt hơn. 

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một class label cho toàn ảnh, kèm class ID/name theo taxonomy đã chọn. | `traffic` trong `classification_predictions.json`: top-1 `cab`, class ID `468`, score `0.510915`, taxonomy `ImageNet-1K`; top-2 `minibus`, score `0.164284`. Đây là dự đoán cấp ảnh, không phải nhãn chuẩn. | Xem toàn ảnh, chọn lớp theo tiêu chí chủ đề chính trong guideline; ghi đúng taxonomy và class; escalation nếu ảnh có nhiều chủ đề ngang nhau hoặc không rõ lớp đại diện. | Kiểm tra nhãn có mô tả toàn ảnh và đúng taxonomy không. Nếu chọn lớp không phù hợp guideline, trả lại để annotator xem xét/sửa; không chọn lớp chỉ vì model có score cao. |
| Phát hiện vật thể | Với mỗi vật thể: class label và bounding box `xyxy` theo pixel. | Output Detection mới của `traffic`: `person`, class ID `0`, score `0.769676`, box `[0.08, 256.79, 18.39, 313.12]`; box sát mép trái ảnh 640 × 428. Record này là prediction của model tại threshold `0.35`. | Gán từng vật thể nhìn thấy theo guideline; vẽ box cho đúng một instance; xử lý vật thể cắt mép theo quy tắc đã được xác nhận, không tự đặt ngưỡng nhìn thấy. | Kiểm tra lớp, box có bao đúng phần nhìn thấy không, box có quá rộng/hẹp hoặc trùng lặp không, có vật thể bị bỏ sót không. Với box sát mép trái, xác minh cách áp dụng quy tắc truncated object; nếu guideline chưa rõ thì escalation, không tự kết luận. |
| Instance segmentation | Với mỗi instance: class label, mask/polygon theo pixel và `instance_id` riêng. | `segmentation_predictions.json`: record `kitchen-002` là `bowl`, score `0.735743`, polygon có 67 điểm; `kitchen-005` là `dining table` với 598 điểm polygon. PNG segmentation cho thấy vùng bàn và các vật đặt trên bàn nằm gần nhau; các mask sát mép ảnh cũng cần được quan sát cẩn thận. | Vẽ mask riêng cho từng vật thể, bám biên nhìn thấy; giữ ID riêng cho từng instance; không gộp hai vật chỉ vì cùng lớp; escalation khi biên bị che hoặc không thể phân định. | Kiểm tra mask có lấn nền/vật thể khác không, biên có bám đúng vật thể không, instance có bị gộp/tách sai không và ID có phân biệt đúng instance không. Trả lại mask lệch biên để sửa; ca mơ hồ chuyển Lab Coach quyết định và cập nhật guideline. |


## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Không viết hoặc đưa dữ liệu bảo mật và cá nhân.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và tôi sẽ "Escalation".

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS