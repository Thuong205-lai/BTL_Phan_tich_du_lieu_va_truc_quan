import geopandas as gpd

# đọc cấp tỉnh (level 1)
gdf = gpd.read_file("gadm41_VNM_1.shp")

# xem thử tên cột
print(gdf.columns)
print(gdf[["NAME_1"]].head())

# lọc Long An
longan = gdf[gdf["NAME_1"].str.contains("Long An", case=False, na=False)]
print(longan)

# xuất ra geojson riêng
longan.to_file("LongAn_boundary.geojson", driver="GeoJSON")

import matplotlib.pyplot as plt

# vẽ bản đồ ranh giới
ax = longan.plot(edgecolor="black", facecolor="none", linewidth=1.5)
ax.set_title("Boundary of Long An (GADM)")
ax.set_axis_off()
plt.show()
