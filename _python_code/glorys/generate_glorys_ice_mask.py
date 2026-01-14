
import xarray as xr
import numpy as np
from PIL import Image
import os
import pyproj
from scipy.interpolate import RegularGridInterpolator

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NC_FILE = os.path.join(SCRIPT_DIR, 'glorys.nc')
OUTPUT_IMAGE = os.path.join(SCRIPT_DIR, 'ice_mask_08.png')

# User-defined parameters
WIDTH = 8464
HEIGHT = 5233
SCALE_FACTOR = 10.0

# Projection Definition
ORTHO_CRS_PROJ4 = "+proj=ortho +lat_0=79.87 +lon_0=-64.37 +x_0=0 +y_0=0 +R=6371000 +units=m +no_defs +type=crs"
ORIG_COEFFS = [11.796660738708361, 0.0, -499251.71124660445, 0.0, -11.796660738708361, 336065.7037738611]

# Ice Thickness Thresholds
THICK_MIN = 0.8  # Maps to 0
THICK_MAX = 3.0  # Maps to 255

def generate_ice_mask():
    if not os.path.exists(NC_FILE):
        print(f"Error: {NC_FILE} not found.")
        return

    print(f"Loading {NC_FILE}...")
    try:
        ds = xr.open_dataset(NC_FILE)
    except Exception:
        ds = xr.open_dataset(NC_FILE, decode_times=False)

    # Identify Variable
    ice_vars = ['sithick', 'sit', 'ist', 'ice_thick']
    ice_name = next((v for v in ice_vars if v in ds.variables), None)
    
    if not ice_name:
        print("Error: Could not find ice thickness variable.")
        return

    # Select Time 0
    print("Selecting first time frame...")
    if 'time' in ds:
         # Assume index 0 is June 8 based on user info
         ds_slice = ds.isel(time=0)
    elif 'time_counter' in ds:
         ds_slice = ds.isel(time_counter=0)
    else:
         ds_slice = ds
    
    # Extract Data as Numpy Arrays
    # We need strictly 1D lat/lon vectors for RegularGridInterpolator
    # GLORYS data can sometimes be curvilinear (nav_lat/nav_lon as 2D arrays).
    # If so, we need to flatten and use griddata (slow) or assume a regular grid if possible.
    # Usually 'latitude' and 'longitude' are 1D in these NetCDFs.
    
    lat_name = next((d for d in ds.coords if 'lat' in d and ds.coords[d].ndim==1), None)
    lon_name = next((d for d in ds.coords if 'lon' in d and ds.coords[d].ndim==1), None)
    
    # If standard 1D coords exist, use them
    if lat_name and lon_name:
        print(f"Using 1D coordinates: {lat_name}, {lon_name}")
        src_lats = ds[lat_name].values
        src_lons = ds[lon_name].values
        src_data = ds_slice[ice_name].values
    else:
        # Fallback for 2D coords (e.g. nav_lat/nav_lon)
        # This is more complex. Check if we can find them.
        print("Warning: Could not find 1D lat/lon coordinates. Checking for 2D...")
        # For simplicity in this script, we'll error out or assume 'latitude' and 'longitude' exist
        # based on previous task output "FrozeMapping... latitude: 2041, longitude: 4320".
        # This implies 1D dimensions.
        if 'latitude' in ds.coords and 'longitude' in ds.coords:
            src_lats = ds.coords['latitude'].values
            src_lons = ds.coords['longitude'].values
            src_data = ds_slice[ice_name].values
        else:
            print("Error: detailed coordinate check failed.")
            return

    # Prepare Interpolator
    # Note: RegularGridInterpolator expects (x, y, z...) points. 
    # Check data shape order. Usually (lat, lon) -> (y, x).
    # src_data shape matches (len(lats), len(lons))
    
    # Handling NaN (fill with 0 for "no ice")
    src_data = np.nan_to_num(src_data, nan=0.0)
    
    print("Creating interpolator...")
    # method='linear' is standard. 'nearest' is faster but blocky.
    # bounds_error=False, fill_value=0.0
    interp = RegularGridInterpolator((src_lats, src_lons), src_data, bounds_error=False, fill_value=0.0)

    # Create Target Grid
    print(f"Generating target grid ({WIDTH}x{HEIGHT})...")
    
    # Define Affine Transform
    # x_proj = a*px + c
    # y_proj = e*py + f
    # Adjust coefficients for scale
    a = ORIG_COEFFS[0] * SCALE_FACTOR
    c = ORIG_COEFFS[2]
    e = ORIG_COEFFS[4] * SCALE_FACTOR
    f = ORIG_COEFFS[5]
    
    # Create meshgrid of pixel coordinates
    # We want center of pixels? Usually integer coordinates refer to corners or centers depending on convention.
    # Let's use standard convention px=0..W-1
    px = np.arange(WIDTH)
    py = np.arange(HEIGHT)
    PX, PY = np.meshgrid(px, py) # Shape (H, W)
    
    # Calculate Projection Coordinates
    X_proj = a * PX + c
    Y_proj = e * PY + f
    
    # Setup Projections
    print("Transforming coordinates to WGS84...")
    crs_ortho = pyproj.CRS.from_proj4(ORTHO_CRS_PROJ4)
    crs_wgs84 = pyproj.CRS.from_string("EPSG:4326")
    transformer = pyproj.Transformer.from_crs(crs_ortho, crs_wgs84, always_xy=True)
    
    # Unproject (X, Y) -> (Lon, Lat)
    target_lons, target_lats = transformer.transform(X_proj, Y_proj)
    
    # Interpolate
    print("Interpolating data...")
    # Points must be (lat, lon) pairs
    # shape (N, 2)
    flat_lats = target_lats.flatten()
    flat_lons = target_lons.flatten()
    
    # RegularGridInterpolator needs points as array of shape (N, 2) corresponding to (dim1, dim2)
    query_points = np.stack((flat_lats, flat_lons), axis=-1)
    
    interpolated_flat = interp(query_points)
    interpolated_data = interpolated_flat.reshape(HEIGHT, WIDTH)
    
    # Apply Thresholds & Mapping
    print("Applying color mapping...")
    # 0.8 -> 0
    # 3.0 -> 255
    # Linear in between
    
    # Normalize to 0-1 range first
    # val_norm = (val - 0.8) / (3.0 - 0.8)
    # Clamp to 0-1
    
    norm_data = (interpolated_data - THICK_MIN) / (THICK_MAX - THICK_MIN)
    norm_data = np.clip(norm_data, 0.0, 1.0)
    
    # Convert to 0-255 uint8
    final_img_data = (norm_data * 255).astype(np.uint8)
    
    # Handle "Absence of ice" strictly
    # If the original interpolated value was effectively 0 (or < THICK_MIN), clipped to 0.
    # So 0 thickness -> (0-0.8)/2.2 < 0 -> clipped to 0 -> Black. Correct.
    # But wait, "Absence of ice or data to be shown in black".
    # If data was NaN, we filled with 0. 0 < 0.8, so maps to 0 (Black).
    # If data was real but < 0.8, maps to 0 (Black).
    # Matches requirements.

    # Save Image
    print(f"Saving to {OUTPUT_IMAGE}...")
    img = Image.fromarray(final_img_data, mode='L') # 'L' for 8-bit grayscale
    img.save(OUTPUT_IMAGE)
    print("Done.")

if __name__ == "__main__":
    generate_ice_mask()
