import warnings
warnings.filterwarnings("ignore")

import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# 0. DOC DU LIEU
df = pd.read_csv("NDVI_SPI3M_merged_v2.csv")

# Chuyen Date sang datetime va lam index
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")

# Tach 2 chuoi quan tam
spi_ts  = df["SPI_3M"].dropna()
ndvi_ts = df["Monthly_NDVI"].dropna()

print("SPI range :", spi_ts.index.min(), "->", spi_ts.index.max(), " n =", len(spi_ts))
print("NDVI range:", ndvi_ts.index.min(), "->", ndvi_ts.index.max(), " n =", len(ndvi_ts))


# 1. HÀM GRID SEARCH CHO SARIMAX
def sarimax_grid_search(train_series, s=12):
    """
    Tim bo tham so (p,d,q) va (P,D,Q,s) toi uu theo AIC
    train_series: pandas Series (index datetime)
    s: so ky vong mua (12 cho du lieu theo thang)
    """

    p = q = range(0, 3)   # 0,1,2
    d_range = [0, 1]
    P = Q = range(0, 2)   # 0,1
    D_range = [0, 1]

    best_aic = np.inf
    best_order = None
    best_seasonal = None

    for order in itertools.product(p, d_range, q):
        for seasonal in itertools.product(P, D_range, Q):
            seasonal_order = (seasonal[0], seasonal[1], seasonal[2], s)
            try:
                model = sm.tsa.statespace.SARIMAX(
                    train_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                res = model.fit(disp=False)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = order
                    best_seasonal = seasonal_order
            except Exception:
                continue

    return best_order, best_seasonal, best_aic


# 2. BACKTEST 12 THÁNG CUỐI
test_size = 12

# SPI BACKTEST 
train_spi = spi_ts[:-test_size]
test_spi  = spi_ts[-test_size:]

print("\n=== BACKTEST SPI-3M ===")
best_order_spi, best_seasonal_spi, best_aic_spi = sarimax_grid_search(train_spi, s=12)
print("Best order SPI       :", best_order_spi)
print("Best seasonal SPI    :", best_seasonal_spi)
print("Best AIC SPI         :", round(best_aic_spi, 3))

model_spi_bt = sm.tsa.statespace.SARIMAX(
    train_spi,
    order=best_order_spi,
    seasonal_order=best_seasonal_spi,
    enforce_stationarity=False,
    enforce_invertibility=False
)
res_spi_bt = model_spi_bt.fit()

pred_spi_test = res_spi_bt.get_forecast(steps=test_size).predicted_mean
compare_spi = pd.DataFrame({
    "Observed_SPI": test_spi,
    "Predicted_SPI": pred_spi_test
})
mae_spi = (compare_spi["Observed_SPI"] - compare_spi["Predicted_SPI"]).abs().mean()
print(compare_spi.head())
print("MAE_SPI_test =", round(mae_spi, 3))

# NDVI BACKTEST 
train_ndvi = ndvi_ts[:-test_size]
test_ndvi  = ndvi_ts[-test_size:]

print("\n=== BACKTEST NDVI ===")
best_order_ndvi, best_seasonal_ndvi, best_aic_ndvi = sarimax_grid_search(train_ndvi, s=12)
print("Best order NDVI      :", best_order_ndvi)
print("Best seasonal NDVI   :", best_seasonal_ndvi)
print("Best AIC NDVI        :", round(best_aic_ndvi, 3))

model_ndvi_bt = sm.tsa.statespace.SARIMAX(
    train_ndvi,
    order=best_order_ndvi,
    seasonal_order=best_seasonal_ndvi,
    enforce_stationarity=False,
    enforce_invertibility=False
)
res_ndvi_bt = model_ndvi_bt.fit()

pred_ndvi_test = res_ndvi_bt.get_forecast(steps=test_size).predicted_mean
compare_ndvi = pd.DataFrame({
    "Observed_NDVI": test_ndvi,
    "Predicted_NDVI": pred_ndvi_test
})
mae_ndvi = (compare_ndvi["Observed_NDVI"] - compare_ndvi["Predicted_NDVI"]).abs().mean()
print(compare_ndvi.head())
print("MAE_NDVI_test =", round(mae_ndvi, 3))


# 3. FORECAST 3 THÁNG TƯƠNG LAI
steps_future = 3

# SPI FORECAST 
print("\n=== FORECAST FUTURE SPI-3M ===")
final_model_spi = sm.tsa.statespace.SARIMAX(
    spi_ts,
    order=best_order_spi,
    seasonal_order=best_seasonal_spi,
    enforce_stationarity=False,
    enforce_invertibility=False
)
final_res_spi = final_model_spi.fit()

fc_spi = final_res_spi.get_forecast(steps=steps_future)
spi_future_mean = fc_spi.predicted_mean
spi_future_ci = fc_spi.conf_int()  # columns: 'lower SPI_3M', 'upper SPI_3M'

print("SPI future mean:")
print(spi_future_mean)
print("\nSPI future 95% CI:")
print(spi_future_ci)

# NDVI FORECAST 
print("\n=== FORECAST FUTURE NDVI ===")
final_model_ndvi = sm.tsa.statespace.SARIMAX(
    ndvi_ts,
    order=best_order_ndvi,
    seasonal_order=best_seasonal_ndvi,
    enforce_stationarity=False,
    enforce_invertibility=False
)
final_res_ndvi = final_model_ndvi.fit()

fc_ndvi = final_res_ndvi.get_forecast(steps=steps_future)
ndvi_future_mean = fc_ndvi.predicted_mean
ndvi_future_ci = fc_ndvi.conf_int()  # columns: 'lower Monthly_NDVI', 'upper Monthly_NDVI'

print("NDVI future mean:")
print(ndvi_future_mean)
print("\nNDVI future 95% CI:")
print(ndvi_future_ci)


# 4. LƯU FORECAST THÀNH CSV
forecast_df = pd.DataFrame({
    "Date": spi_future_mean.index,
    "SPI_Forecast": spi_future_mean.values,
    "NDVI_Forecast": ndvi_future_mean.reindex(spi_future_mean.index).values
})
forecast_df.to_csv("forecast_3months.csv", index=False)
print("\nSaved forecast to forecast_3months.csv")


# 5. VẼ BIỂU ĐỒ 

try:
    plt.style.use("seaborn-v0_8")
except Exception:
    pass

# FIGURE 1: Time series NDVI & SPI 
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(ndvi_ts.index, ndvi_ts.values, label="NDVI", linewidth=1.8)
ax1.set_ylabel("NDVI", fontsize=12)
ax1.set_xlabel("Năm", fontsize=12)

ax2 = ax1.twinx()
ax2.plot(spi_ts.index, spi_ts.values, label="SPI-3M", linestyle="--", linewidth=1.3)
ax2.set_ylabel("SPI-3M", fontsize=12)
ax2.axhline(0, linestyle=":", linewidth=1)

fig.suptitle("Chuỗi thời gian NDVI và SPI-3M tại Long An", fontsize=14, fontweight="bold")
fig.tight_layout()

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper right")

plt.savefig("Fig1_TimeSeries_NDVI_SPI.png", dpi=300, bbox_inches="tight")
plt.show()


# FIGURE 2a: Backtest SPI (12 tháng cuối) 
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(compare_spi.index, compare_spi["Observed_SPI"], marker="o",
        label="SPI quan sát", linewidth=1.5)
ax.plot(compare_spi.index, compare_spi["Predicted_SPI"], marker="s",
        label="SPI dự báo", linewidth=1.5)

ax.axhline(0, linestyle=":", linewidth=1)
ax.set_xlabel("Thời gian", fontsize=11)
ax.set_ylabel("SPI-3M", fontsize=11)
ax.set_title("Backtest SPI-3M – 12 tháng cuối", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Fig2a_Backtest_SPI.png", dpi=300, bbox_inches="tight")
plt.show()


# FIGURE 2b: Backtest NDVI (12 tháng cuối) 
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(compare_ndvi.index, compare_ndvi["Observed_NDVI"], marker="o",
        label="NDVI quan sát", linewidth=1.5)
ax.plot(compare_ndvi.index, compare_ndvi["Predicted_NDVI"], marker="s",
        label="NDVI dự báo", linewidth=1.5)

ax.set_xlabel("Thời gian", fontsize=11)
ax.set_ylabel("NDVI", fontsize=11)
ax.set_title("Backtest NDVI – 12 tháng cuối", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Fig2b_Backtest_NDVI.png", dpi=300, bbox_inches="tight")
plt.show()


# FIGURE 3a: Forecast SPI 3 tháng tới 
last_years_spi = spi_ts[spi_ts.index >= (spi_ts.index.max() - pd.DateOffset(years=3))]

spi_fc = spi_future_mean
spi_fc_lower = spi_future_ci["lower SPI_3M"]
spi_fc_upper = spi_future_ci["upper SPI_3M"]

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(last_years_spi.index, last_years_spi.values, label="SPI lịch sử", linewidth=1.5)
ax.plot(spi_fc.index, spi_fc.values, marker="o", label="SPI forecast", linewidth=1.5)
ax.fill_between(spi_fc.index, spi_fc_lower, spi_fc_upper, alpha=0.2,
                label="Khoảng tin cậy 95%")

ax.axhline(0, linestyle=":", linewidth=1)
ax.set_xlabel("Thời gian", fontsize=11)
ax.set_ylabel("SPI-3M", fontsize=11)
ax.set_title("Dự báo SPI-3M cho 3 tháng tiếp theo", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Fig3a_Forecast_SPI.png", dpi=300, bbox_inches="tight")
plt.show()


# FIGURE 3b: Forecast NDVI 3 tháng tới 
last_years_ndvi = ndvi_ts[ndvi_ts.index >= (ndvi_ts.index.max() - pd.DateOffset(years=3))]

ndvi_fc = ndvi_future_mean
ndvi_fc_lower = ndvi_future_ci["lower Monthly_NDVI"]
ndvi_fc_upper = ndvi_future_ci["upper Monthly_NDVI"]

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(last_years_ndvi.index, last_years_ndvi.values, label="NDVI lịch sử", linewidth=1.5)
ax.plot(ndvi_fc.index, ndvi_fc.values, marker="o", label="NDVI forecast", linewidth=1.5)
ax.fill_between(ndvi_fc.index, ndvi_fc_lower, ndvi_fc_upper, alpha=0.2,
                label="Khoảng tin cậy 95%")

ax.set_xlabel("Thời gian", fontsize=11)
ax.set_ylabel("NDVI", fontsize=11)
ax.set_title("Dự báo NDVI cho 3 tháng tiếp theo", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Fig3b_Forecast_NDVI.png", dpi=300, bbox_inches="tight")
plt.show()
