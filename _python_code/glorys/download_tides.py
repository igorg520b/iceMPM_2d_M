import copernicusmarine

# Nares Strait Region
LAT_MIN, LAT_MAX = 76.00, 83.00
LON_MIN, LON_MAX = -83.4, -54.45

print("Starting download of tidal currents for Nares Strait...")
print(f"Lat: {LAT_MIN} to {LAT_MAX}")
print(f"Lon: {LON_MIN} to {LON_MAX}")

copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
    variables=[
        "utide",  # eastward tidal velocity
        "vtide"   # northward tidal velocity
    ],
    minimum_longitude=LON_MIN,
    maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN,
    maximum_latitude=LAT_MAX,
    start_datetime="2024-06-08T00:00:00",
    end_datetime="2024-06-21T00:00:00",
    output_filename="glo12v4_nares_tidal_currents.nc",
    force_download=True
)

print("Download complete. Saved as 'glo12v4_nares_tidal_currents.nc'.")