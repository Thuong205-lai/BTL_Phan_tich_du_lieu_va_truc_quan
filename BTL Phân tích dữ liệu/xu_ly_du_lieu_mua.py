import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
from scipy.stats import gamma, norm

# 1) đọc file mưa
rain = pd.read_csv("mua_3_month.csv")

# 2) đổi tên cột cho đúng
rain = rain.rename(columns={
    "Monthly Rainfall (mm)": "Rain_month",
    "Rain 3mo (mm)": "Rain3mo"
})

# 3) tạo Date cuối tháng từ Year + Month 
rain["Date"] = pd.to_datetime(
    dict(year=rain["Year"], month=rain["Month"], day=1)
) + MonthEnd(0)

rain = rain.sort_values("Date").reset_index(drop=True)

# 4) tính lại Rain3mo chuẩn (không reset qua năm)
rain["Rain3mo"] = rain["Rain_month"].rolling(3).sum()

# 5) tính SPI-3M từ Rain3mo
rain["month_of_year"] = rain["Date"].dt.month
spi_vals = np.full(len(rain), np.nan)

for m in range(1, 13):
    sub = rain[rain["month_of_year"] == m]
    x = sub["Rain3mo"].values.astype(float)

    p0 = np.mean(x == 0)
    x_pos = x[x > 0]
    if len(x_pos) < 3:
        continue

    shape, loc, scale = gamma.fit(x_pos, floc=0)
    cdf = p0 + (1 - p0) * gamma.cdf(x, a=shape, loc=loc, scale=scale)
    spi = norm.ppf(cdf)

    spi_vals[sub.index] = spi

rain["SPI_3M"] = spi_vals

print(rain.head(12))

# 6) lưu ra file 
rain.to_csv("rain_SPI3M.csv", index=False)

ndvi = pd.read_csv("NDVI_Monthly_LongAn_2000_2025.csv")
ndvi["Date"] = pd.to_datetime(ndvi["Date"])
ndvi = ndvi.sort_values("Date")

merged = pd.merge(
    ndvi[["Date", "Monthly_NDVI"]],
    rain[["Date", "Rain3mo", "SPI_3M"]],
    on="Date", how="inner"
)

merged.to_csv("NDVI_SPI3M_merged_v2.csv", index=False)

