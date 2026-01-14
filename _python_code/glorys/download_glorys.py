
import copernicusmarine

# Nares Strait Region
LAT_MIN, LAT_MAX = 75.00, 85.00
LON_MIN, LON_MAX = -85.32, -43.42

print("Starting download for Nares Strait region...")
print(f"Lat: {LAT_MIN} to {LAT_MAX}")
print(f"Lon: {LON_MIN} to {LON_MAX}")

copernicusmarine.subset(
  dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
  variables=["uo", "vo"],
  minimum_longitude=LON_MIN,
  maximum_longitude=LON_MAX,
  minimum_latitude=LAT_MIN,
  maximum_latitude=LAT_MAX,
  start_datetime="2024-06-08T00:00:00",
  end_datetime="2024-06-21T00:00:00",
  minimum_depth=0.49402499198913574,
  maximum_depth=0.49402499198913574,
  output_filename="glo12v4_nares_currents.nc", # GLO12v4 dataset
  force_download=True
)

print("Download complete. Saved as 'glo12v4_nares_currents.nc'.")
