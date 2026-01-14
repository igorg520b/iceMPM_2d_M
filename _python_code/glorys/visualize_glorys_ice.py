
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from matplotlib.widgets import Slider
import os
from PIL import Image

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NC_FILE = os.path.join(SCRIPT_DIR, 'glorys.nc')
# Image is in parent directory
IMAGE_FILE = os.path.join(SCRIPT_DIR, '..', 'land_2k.png')

# Projection Parameters
LAT_0 = 79.87
LON_0 = -64.37
R = 6371000

# Original Coefficients (for full-res image)
ORIG_COEFFS = [11.796660738708361, 0.0, -499251.71124660445, 0.0, -11.796660738708361, 336065.7037738611]
SCALE_FACTOR = 40.0

VAR_ICE = 'sithick'
ALT_VAR_ICE = ['sit', 'ist', 'ice_thick']

def find_variable(ds, choices):
    for v in choices:
        if v in ds.variables:
            return v
    return None

def visualize_ice():
    if not os.path.exists(NC_FILE):
        print(f"Error: File {NC_FILE} not found.")
        return
    if not os.path.exists(IMAGE_FILE):
        print(f"Error: File {IMAGE_FILE} not found.")
        return

    # 1. Setup Projection and Image Extent
    img = Image.open(IMAGE_FILE)
    img_width, img_height = img.size
    print(f"Loaded image {IMAGE_FILE} ({img_width}x{img_height})")

    a = ORIG_COEFFS[0] * SCALE_FACTOR
    c = ORIG_COEFFS[2]
    e = ORIG_COEFFS[4] * SCALE_FACTOR
    f = ORIG_COEFFS[5]

    x_min = c
    x_max = c + (img_width * a)
    y_max = f
    y_min = f + (img_height * e)
    
    img_extent = [x_min, x_max, y_min, y_max]
    print(f"Calculated image extent (meters): {img_extent}")

    # 2. Load Data
    print(f"Loading {NC_FILE}...")
    try:
        ds = xr.open_dataset(NC_FILE)
    except Exception:
        ds = xr.open_dataset(NC_FILE, decode_times=False)

    ice_name = find_variable(ds, [VAR_ICE] + ALT_VAR_ICE)
    if not ice_name:
        print("Error: Could not find ice thickness variable.")
        print(ds.variables)
        return

    # Time handling
    if 'time' in ds: times = ds['time']
    elif 'time_counter' in ds: times = ds['time_counter']
    else: times = np.arange(ds.dims.get('time', 1))

    # 3. Plotting
    globe = ccrs.Globe(ellipse=None, semimajor_axis=R, semiminor_axis=R)
    projection = ccrs.Orthographic(central_latitude=LAT_0, central_longitude=LON_0, globe=globe)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1, projection=projection)
    
    ax.set_extent(img_extent, crs=projection)
    ax.imshow(img, origin='upper', extent=img_extent, transform=projection)

    # Initial Plot
    time_idx = 0
    t_val = times[time_idx].values if hasattr(times[time_idx], 'values') else times[time_idx]
    time_str = str(t_val)[:16]

    # Plot Ice
    data_ice = ds[ice_name].isel({times.dims[0]: time_idx})
    ice_plot = data_ice.plot(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap='Blues_r', add_colorbar=True,
        vmin=0, vmax=float(ds[ice_name].max(skipna=True)),
        alpha=0.6
    )

    plt.title(f"GLORYS Ice Thickness - {time_str}")

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Time', 0, len(times) - 1, valinit=0, valfmt='%0.0f')

    def update(val):
        idx = int(slider.val)
        t_now = times[idx].values if hasattr(times[idx], 'values') else times[idx]
        plt.title(f"GLORYS Ice Thickness - {str(t_now)[:16]}")
        
        # Ice update
        new_ice = ds[ice_name].isel({times.dims[0]: idx}).values
        ice_plot.set_array(new_ice.flatten())
            
        fig.canvas.draw_idle()

    slider.on_changed(update)
    print("Visualization ready. Showing plot...")
    plt.show()

if __name__ == "__main__":
    visualize_ice()
