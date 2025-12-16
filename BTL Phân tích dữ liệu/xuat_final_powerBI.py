import pandas as pd
import numpy as np

#1. Đọc dữ liệu lịch sử 
hist = pd.read_csv("NDVI_SPI3M_merged_v2.csv")
hist["Date"] = pd.to_datetime(hist["Date"])

#file này có cột:
# 'Date', 'Monthly_NDVI', 'Rain3mo', 'SPI_3M'
print("Lịch sử:", hist.head())

# Thêm cột forecast (ban đầu để trống) + loại dữ liệu
hist["SPI_Forecast"]  = np.nan
hist["NDVI_Forecast"] = np.nan
hist["Data_Type"]     = "Observed"

#2. Đọc dữ liệu dự báo 3 tháng
fc = pd.read_csv("forecast_3months.csv")
fc["Date"] = pd.to_datetime(fc["Date"])

# file forecast_3months.csv đang có:
# 'Date', 'SPI_Forecast', 'NDVI_Forecast'
print("Forecast:", fc.head())

# Tạo các cột còn thiếu để cùng cấu trúc với hist
fc["Monthly_NDVI"] = np.nan
fc["Rain3mo"]      = np.nan
fc["SPI_3M"]       = np.nan
fc["Data_Type"]    = "Forecast"

# Sắp xếp lại thứ tự cột cho giống nhau
cols = ["Date", "Data_Type",
        "Monthly_NDVI", "SPI_3M", "Rain3mo",
        "SPI_Forecast", "NDVI_Forecast"]

hist = hist[cols]
fc   = fc[cols]

# 3. Ghép lịch sử + forecast 
final = pd.concat([hist, fc], ignore_index=True)
final = final.sort_values("Date")

print("\nDữ liệu tổng hợp:")
print(final.tail(10))

# 4. Xuất CSV cuối cùng
final.to_csv("FINAL_for_PowerBI.csv", index=False)
print("\nĐÃ LƯU FILE: FINAL_for_PowerBI.csv")
