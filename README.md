# BÀI TẬP LỚN – PHÂN TÍCH DỮ LIỆU VÀ TRỰC QUAN HÓA  
## Hệ thống cảnh báo sớm hạn hán và ngập lụt tỉnh Long An

## 1. Giới thiệu
Repository này lưu trữ toàn bộ **dữ liệu và mã nguồn** phục vụ bài tập lớn
môn Phân tích dữ liệu, với mục tiêu xây dựng và đánh giá **hệ thống cảnh báo sớm
hạn hán và ngập lụt cho tỉnh Long An** dựa trên dữ liệu khí tượng và viễn thám.

Nghiên cứu tập trung vào việc tích hợp dữ liệu mưa, chỉ số hạn khí tượng
(SPI-3M) và chỉ số thảm thực vật (NDVI), kết hợp phân tích thống kê,
dự báo chuỗi thời gian và trực quan hóa kết quả.



## 2. Dữ liệu sử dụng
- **Dữ liệu khí tượng (lượng mưa):**
  - Nguồn: NASA POWER
  - Dữ liệu mưa theo ngày, tổng hợp theo tháng và 3 tháng

- **Dữ liệu viễn thám (NDVI):**
  - Nguồn: MODIS (Google Earth Engine)
  - NDVI trung bình theo tháng cho khu vực tỉnh Long An

- **Dữ liệu không gian:**
  - Ranh giới hành chính tỉnh Long An (shapefile/geojson)



## 3. Nội dung và quy trình thực hiện
Các bước chính trong bài tập lớn bao gồm:

1. Thu thập và tiền xử lý dữ liệu mưa và NDVI  
2. Tính toán **chỉ số hạn khí tượng SPI-3M** từ lượng mưa lũy tích 3 tháng  
3. Phân tích thống kê mối quan hệ giữa **SPI-3M và NDVI**  
4. Dự báo ngắn hạn (1–3 tháng) bằng **mô hình chuỗi thời gian SARIMA**  
5. Trực quan hóa kết quả bằng **Power BI** và xây dựng ứng dụng truy cập nhanh bằng **Power Apps**



## 4. Mã nguồn
Repository bao gồm các file mã nguồn Python phục vụ:
- Xử lý và tổng hợp dữ liệu mưa
- Truy vấn và xử lý dữ liệu NDVI
- Phân tích thống kê (mô tả và suy luận)
- Xây dựng và đánh giá mô hình dự báo SARIMA
- Xuất dữ liệu phục vụ trực quan hóa Power BI

Các file được đặt trong cùng một thư mục nhằm đảm bảo
**khả năng chạy ổn định của mã nguồn** khi thực thi.



## 5. Kết quả chính
- Xây dựng được bộ dữ liệu tích hợp khí tượng – viễn thám cho tỉnh Long An  
- Phân tích cho thấy mối quan hệ giữa SPI-3M và NDVI chưa thể hiện
  sự khác biệt thống kê rõ rệt giữa các mức hạn
- Mô hình SARIMA cho phép dự báo SPI-3M và NDVI trong ngắn hạn (1–3 tháng)
- Kết quả được trực quan hóa thông qua dashboard Power BI và ứng dụng Power Apps



## 6. Trực quan hóa và ứng dụng
- **Power BI:** Dashboard trực quan hóa chuỗi thời gian SPI-3M, NDVI và kết quả dự báo
- **Power Apps:** Ứng dụng hỗ trợ truy cập nhanh hệ thống cảnh báo trên thiết bị di động

(Chi tiết link Power BI và Power Apps được trình bày trong báo cáo bài tập lớn)



## 7. Công cụ và thư viện sử dụng
- Ngôn ngữ: Python  
- Thư viện chính: pandas, numpy, matplotlib, statsmodels  
- Trực quan hóa: Power BI  
- Ứng dụng: Power Apps  




