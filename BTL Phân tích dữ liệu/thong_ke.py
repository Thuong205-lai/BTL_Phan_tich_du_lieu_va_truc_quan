import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


merged = pd.read_csv("NDVI_SPI3M_merged_v2.csv")
merged["Date"] = pd.to_datetime(merged["Date"])

# describe tổng quát
desc_all = merged.describe().round(3)
print(desc_all)

# theo năm
merged["Year"] = merged["Date"].dt.year
stats_year = merged.groupby("Year").agg(
    NDVI_mean=("Monthly_NDVI", "mean"),
    NDVI_std =("Monthly_NDVI", "std"),
    NDVI_min =("Monthly_NDVI", "min"),
    NDVI_max =("Monthly_NDVI", "max"),
    SPI_mean=("SPI_3M", "mean"),
    SPI_std =("SPI_3M", "std"),
    SPI_min =("SPI_3M", "min"),
    SPI_max =("SPI_3M", "max"),
    Rain3mo_mean=("Rain3mo", "mean"),
    Rain3mo_sum=("Rain3mo", "sum"),
    n_months=("Monthly_NDVI", "count")
).round(3)

print(stats_year.head())

desc_all.to_csv("desc_overall.csv")
stats_year.to_csv("stats_by_year.csv")


# PHÂN TÍCH THỐNG KÊ SUY LUẬN (T-test / ANOVA)

# làm sạch dữ liệu cho suy luận
merged_inf = merged.dropna(subset=["Monthly_NDVI", "SPI_3M"]).copy()
print("\n--- Inferential statistics ---")
print("Total samples =", len(merged_inf))

#  T-TEST: so sánh NDVI giữa 2 nhóm Hạn vs Không hạn
# Quy ước:
#   Hạn: SPI_3M <= -1.0
#   Không hạn/Bình thường: SPI_3M > -1.0
drought = merged_inf[merged_inf["SPI_3M"] <= -1.0]["Monthly_NDVI"]
normal  = merged_inf[merged_inf["SPI_3M"] >  -1.0]["Monthly_NDVI"]

print("\n[T-test] NDVI drought vs normal")
print("n drought =", len(drought), "mean =", round(drought.mean(), 4))
print("n normal  =", len(normal),  "mean =", round(normal.mean(), 4))

t_stat, p_val = stats.ttest_ind(drought, normal, equal_var=False, nan_policy="omit")
print("Welch t-test: t =", round(t_stat, 3), "p =", round(p_val, 5))

# Effect size: Cohen's d
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((nx-1)*sx + (ny-1)*sy) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / s_pooled

d = cohens_d(drought, normal)
print("Cohen's d =", round(d, 3))

# 2) ANOVA: so sánh NDVI giữa 3 mức SPI (Hạn - Bình thường - Ẩm)
def spi_class(spi):
    if spi <= -1.0:
        return "Drought"
    elif spi >= 1.0:
        return "Wet"
    else:
        return "Normal"

merged_inf["SPI_class"] = merged_inf["SPI_3M"].apply(spi_class)

group_drought = merged_inf[merged_inf["SPI_class"] == "Drought"]["Monthly_NDVI"]
group_normal  = merged_inf[merged_inf["SPI_class"] == "Normal"]["Monthly_NDVI"]
group_wet     = merged_inf[merged_inf["SPI_class"] == "Wet"]["Monthly_NDVI"]

print("\n[ANOVA] NDVI among Drought / Normal / Wet")
print("n drought =", len(group_drought),
      "| n normal =", len(group_normal),
      "| n wet =", len(group_wet))

F_stat, p_anova = stats.f_oneway(group_drought, group_normal, group_wet)
print("ANOVA: F =", round(F_stat, 3), "p =", round(p_anova, 5))

# 3) Post-hoc: pairwise Welch t-test + Bonferroni
pairs = [
    ("Drought", group_drought, "Normal", group_normal),
    ("Drought", group_drought, "Wet", group_wet),
    ("Normal",  group_normal,  "Wet", group_wet)
]

alpha = 0.05
alpha_bonf = alpha / len(pairs)

print("\n[Post-hoc pairwise Welch t-tests + Bonferroni]")
for a_name, a, b_name, b in pairs:
    t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    sig = "SIGNIFICANT" if p < alpha_bonf else "ns"
    print(f"{a_name} vs {b_name}: t={t:.3f}, p={p:.5f} -> {sig} (alpha={alpha_bonf:.3f})")

# TƯƠNG QUAN SPI ↔ NDVI

df_corr = merged.dropna(subset=["Monthly_NDVI", "SPI_3M"]).copy()

# tương quan Pearson tổng quát
corr_val = df_corr["Monthly_NDVI"].corr(df_corr["SPI_3M"])
print("\n[Correlation]")
print("Pearson corr (NDVI vs SPI_3M) =", round(corr_val, 3))

# thử tương quan có độ trễ 1–2 tháng
df_corr = df_corr.sort_values("Date")
df_corr["SPI_lag1"] = df_corr["SPI_3M"].shift(1)
df_corr["SPI_lag2"] = df_corr["SPI_3M"].shift(2)

print("corr NDVI vs SPI(t)   =", round(df_corr["Monthly_NDVI"].corr(df_corr["SPI_3M"]), 3))
print("corr NDVI vs SPI(t-1) =", round(df_corr["Monthly_NDVI"].corr(df_corr["SPI_lag1"]), 3))
print("corr NDVI vs SPI(t-2) =", round(df_corr["Monthly_NDVI"].corr(df_corr["SPI_lag2"]), 3))

# scatter SPI vs NDVI
plt.figure()
plt.scatter(df_corr["SPI_3M"], df_corr["Monthly_NDVI"])
plt.xlabel("SPI-3M")
plt.ylabel("Monthly NDVI")
plt.title("Scatter: SPI-3M vs NDVI")
plt.show()

# time series NDVI và SPI
plt.figure()
plt.plot(df_corr["Date"], df_corr["Monthly_NDVI"], label="NDVI")
plt.plot(df_corr["Date"], df_corr["SPI_3M"], label="SPI-3M")
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Time series: NDVI and SPI-3M")
plt.legend()
plt.show()
