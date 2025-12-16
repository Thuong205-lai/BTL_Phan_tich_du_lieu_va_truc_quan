import pandas as pd
import statsmodels.api as sm
import itertools
import warnings
warnings.filterwarnings("ignore")

# 0) READ DATA
df = pd.read_csv("NDVI_SPI3M_merged_v2.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")

spi_ts  = df["SPI_3M"].dropna()
ndvi_ts = df["Monthly_NDVI"].dropna()

print("SPI range :", spi_ts.index.min(), "->", spi_ts.index.max(), "n=", len(spi_ts))
print("NDVI range:", ndvi_ts.index.min(), "->", ndvi_ts.index.max(), "n=", len(ndvi_ts))

# 1) GRID SEARCH SARIMAX (seasonal => SARIMA)
def sarimax_grid_search(train_series, s=12):
    p = q = range(0, 3)      # 0,1,2
    d_range = [0, 1]
    P = Q = range(0, 2)      # 0,1
    D_range = [0, 1]

    best_aic = float("inf")
    best_order, best_seasonal = None, None

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
            except:
                continue

    return best_order, best_seasonal, best_aic


def fit_sarimax(series, order, seasonal_order):
    model = sm.tsa.statespace.SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)


def backtest(series, test_size=12, s=12, name="SERIES"):
    train = series.iloc[:-test_size]
    test  = series.iloc[-test_size:]

    best_order, best_seasonal, best_aic = sarimax_grid_search(train, s=s)
    res = fit_sarimax(train, best_order, best_seasonal)

    pred = res.get_forecast(steps=test_size).predicted_mean
    pred = pd.Series(pred.values, index=test.index, name=f"Predicted_{name}")

    compare = pd.DataFrame({
        f"Observed_{name}": test,
        f"Predicted_{name}": pred
    })
    mae = (compare[f"Observed_{name}"] - compare[f"Predicted_{name}"]).abs().mean()

    return best_order, best_seasonal, best_aic, compare, mae


def forecast_future(series, order, seasonal_order, steps=3, name="SERIES"):
    res_full = fit_sarimax(series, order, seasonal_order)
    fc = res_full.get_forecast(steps=steps)

    mean = fc.predicted_mean
    mean = pd.Series(mean.values, index=mean.index, name=f"{name}_future_mean")

    ci = fc.conf_int()
    # đặt tên cột CI cho rõ
    ci.columns = [f"lower_{name}", f"upper_{name}"]

    return mean, ci


# A) BACKTEST (SPI, NDVI)
test_size = 12
season_len = 12

print("\n=== BACKTEST SPI-3M ===")
best_order_spi, best_seasonal_spi, best_aic_spi, compare_spi, mae_spi = backtest(
    spi_ts, test_size=test_size, s=season_len, name="SPI"
)
print("Best order SPI:", best_order_spi)
print("Best seasonal SPI:", best_seasonal_spi)
print("Best AIC SPI:", round(best_aic_spi, 3))
print(compare_spi.head())
print("MAE_SPI_test =", round(mae_spi, 3))

print("\n=== BACKTEST NDVI ===")
best_order_ndvi, best_seasonal_ndvi, best_aic_ndvi, compare_ndvi, mae_ndvi = backtest(
    ndvi_ts, test_size=test_size, s=season_len, name="NDVI"
)
print("Best order NDVI:", best_order_ndvi)
print("Best seasonal NDVI:", best_seasonal_ndvi)
print("Best AIC NDVI:", round(best_aic_ndvi, 3))
print(compare_ndvi.head())
print("MAE_NDVI_test =", round(mae_ndvi, 3))


# B) FORECAST FUTURE 1-3 MONTHS
steps_future = 3

print("\n=== FORECAST FUTURE SPI-3M ===")
spi_future_mean, spi_future_ci = forecast_future(
    spi_ts, best_order_spi, best_seasonal_spi, steps=steps_future, name="SPI_3M"
)
print("SPI future mean:")
print(spi_future_mean)
print("\nSPI future 95% CI:")
print(spi_future_ci)

print("\n=== FORECAST FUTURE NDVI ===")
ndvi_future_mean, ndvi_future_ci = forecast_future(
    ndvi_ts, best_order_ndvi, best_seasonal_ndvi, steps=steps_future, name="Monthly_NDVI"
)
print("NDVI future mean:")
print(ndvi_future_mean)
print("\nNDVI future 95% CI:")
print(ndvi_future_ci)


# SAVE OUTPUT
out_future = pd.concat([spi_future_mean, ndvi_future_mean], axis=1)
out_future.to_csv("forecast_3months.csv")
print("\nSaved forecast to forecast_3months.csv")
