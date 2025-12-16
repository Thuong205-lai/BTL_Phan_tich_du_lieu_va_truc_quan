import ee
import pandas as pd
import numpy as np

ee.Initialize(project='data-bolt-479209-m5')
# ee.Authenticate()  

POINT = ee.Geometry.Point([106.24, 10.76])
START_DATE = '2000-01-01'
END_DATE   = '2025-10-01'
OUTPUT_FILENAME = 'NDVI_Monthly_LongAn_2000_2025.csv'

NDVI_COLLECTION = 'MODIS/006/MOD13A2'
SCALE_FACTOR = 0.0001

def extract_ndvi(image):
    ndvi = image.select('NDVI')
    value = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=POINT,
        scale=1000,
        bestEffort=True
    ).get('NDVI')

    return image.set({
        'NDVI_point': ee.Number(value).multiply(SCALE_FACTOR)
    })

# 1) Lọc collection
ndvi_col = (ee.ImageCollection(NDVI_COLLECTION)
            .filterDate(START_DATE, END_DATE)
            .filterBounds(POINT)
            .map(extract_ndvi))

# 2) Lấy mảng Date và NDVI 
dates = ndvi_col.aggregate_array('system:time_start').getInfo()
ndvis = ndvi_col.aggregate_array('NDVI_point').getInfo()

# 3) Đưa về DataFrame
data = []
for d, v in zip(dates, ndvis):
    if v is not None:
        data.append({
            'Date': pd.to_datetime(d, unit='ms'),
            'NDVI': v
        })

df_ndvi = pd.DataFrame(data).set_index('Date')

# 4) Tính trung bình theo tháng
monthly_ndvi = df_ndvi['NDVI'].resample('M').mean().to_frame(name='Monthly_NDVI')

# 5) Xuất CSV
monthly_ndvi.to_csv(OUTPUT_FILENAME)

print("\n--- THÀNH CÔNG ---")
print(f"File NDVI đã được lưu: {OUTPUT_FILENAME}")
print(monthly_ndvi.head())
