
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from matplotlib.widgets import Slider
import os
from PIL import Image

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NC_FILE = os.path.join(SCRIPT_DIR, 'glo12v4_nares_currents.nc')
# Image is in parent directory
IMAGE_FILE = os.path.join(SCRIPT_DIR, '..', 'land_2k.png')

# Projection Parameters
LAT_0 = 79.87
LON_0 = -64.37
R = 6371000

# Original Coefficients (for full-res image)
# [a, b, c, d, e, f, ...]
# a = x_scale, e = y_scale (negative)
ORIG_COEFFS = [11.796660738708361, 0.0, -499251.71124660445, 0.0, -11.796660738708361, 336065.7037738611]

# Scaling Factor (Image is 40x smaller)
SCALE_FACTOR = 40.0

# Standard GLORYS variable names
VAR_U = 'uo'
VAR_V = 'vo'
ALT_VAR_U = ['u', 'ugeo', 'vozocrtx'] 
ALT_VAR_V = ['v', 'vgeo', 'vomecrty']

def find_variable(ds, choices):
    for v in choices:
        if v in ds.variables:
            return v
    return None

def visualize_currents():
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

    # Apply scaling to coefficients
    # x_coord = (a * 40) * x_pixel + c
    # y_coord = (e * 40) * y_pixel + f
    a = ORIG_COEFFS[0] * SCALE_FACTOR
    c = ORIG_COEFFS[2]
    e = ORIG_COEFFS[4] * SCALE_FACTOR
    f = ORIG_COEFFS[5]

    x_min = c
    x_max = c + (img_width * a) # a is positive width/px
    y_max = f
    y_min = f + (img_height * e) # e is negative height/px (top-down image)
    
    # Check if a/e signs match standard Affine expectation (usually a>0, e<0 for map-projected images from top-left)
    # Given coords: e = -11... which is correct for y decreasing as pixel y increases
    
    img_extent = [x_min, x_max, y_min, y_max]
    print(f"Calculated image extent (meters): {img_extent}")

    # 2. Load Data
    print(f"Loading {NC_FILE}...")
    try:
        ds = xr.open_dataset(NC_FILE)
    except Exception:
        ds = xr.open_dataset(NC_FILE, decode_times=False)

    u_name = find_variable(ds, [VAR_U] + ALT_VAR_U)
    v_name = find_variable(ds, [VAR_V] + ALT_VAR_V)

    if not (u_name and v_name):
        print("Error: Could not find current variables.")
        print(ds.variables)
        return

    # Handle depth (surface)
    dims = ds[u_name].dims
    if 'depth' in dims or 'deptht' in dims or 'lev' in dims:
         depth_dim = next((d for d in dims if 'depth' in d or 'lev' in d), None)
         u_da = ds[u_name].isel({depth_dim: 0})
         v_da = ds[v_name].isel({depth_dim: 0})
    else:
         u_da = ds[u_name]
         v_da = ds[v_name]

    ds['speed'] = np.sqrt(u_da**2 + v_da**2)

    # Time handling
    if 'time' in ds: times = ds['time']
    elif 'time_counter' in ds: times = ds['time_counter']
    else: times = np.arange(ds.dims.get('time', 1))

    # 3. Plotting
    globe = ccrs.Globe(ellipse=None, semimajor_axis=R, semiminor_axis=R)
    projection = ccrs.Orthographic(central_latitude=LAT_0, central_longitude=LON_0, globe=globe)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1, projection=projection)
    
    # Set extent to match image
    ax.set_extent(img_extent, crs=projection)

    # Show Background Image
    # origin='upper' is standard for images. extent matches projection coords provided.
    ax.imshow(img, origin='upper', extent=img_extent, transform=projection)

    # Initial Plot
    time_idx = 0
    t_val = times[time_idx].values if hasattr(times[time_idx], 'values') else times[time_idx]
    time_str = str(t_val)[:16]

    # Plot Speed
    speed_plot = ds.speed.isel({times.dims[0]: time_idx}).plot(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap='Blues_r', add_colorbar=True,
        vmin=0, vmax=float(ds.speed.max(skipna=True)) * 0.8,
        alpha=0.6 # Transparency
    )
    
    # Plot Quiver
    # Setup Quiver coords once (assuming static grid)
    def get_coord(ds, keys):
        for k in keys:
            if k in ds.coords or k in ds.variables: return ds[k]
        return None
    lons = get_coord(ds, ['nav_lon', 'longitude', 'lon'])
    lats = get_coord(ds, ['nav_lat', 'latitude', 'lat'])
    
    quiver = None
    if lons is not None and lats is not None:
        try:
            skip = (slice(None, None, 5), slice(None, None, 5)) # Aggressive skip for overlay
            
            u_vals = u_da.isel({times.dims[0]: time_idx}).values
            v_vals = v_da.isel({times.dims[0]: time_idx}).values
            
            X_q = lons[skip] if lons.ndim==2 else np.meshgrid(lons[skip[1]], lats[skip[0]])[0]
            Y_q = lats[skip] if lats.ndim==2 else np.meshgrid(lons[skip[1]], lats[skip[0]])[1]
            # Handle potential shape mismatch if using skip directly on 2d arrays 
            # Safe slice logic for data
            U_q = u_vals[skip]
            V_q = v_vals[skip]

            quiver = ax.quiver(X_q, Y_q, U_q, V_q, transform=ccrs.PlateCarree(),
                               scale=None, color='black', alpha=0.8, width=0.002)
        except Exception as e:
            print(f"Quiver init failed: {e}")

    plt.title(f"GLO12v4 Currents - {time_str}")

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Time', 0, len(times) - 1, valinit=0, valfmt='%0.0f')

    def update(val):
        idx = int(slider.val)
        t_now = times[idx].values if hasattr(times[idx], 'values') else times[idx]
        plt.title(f"GLO12v4 Currents - {str(t_now)[:16]}")
        
        # Speed
        new_speed = ds.speed.isel({times.dims[0]: idx}).values
        speed_plot.set_array(new_speed.flatten())
        
        # Quiver
        if quiver:
            new_u = u_da.isel({times.dims[0]: idx}).values[skip]
            new_v = v_da.isel({times.dims[0]: idx}).values[skip]
            quiver.set_UVC(new_u, new_v)
            
        fig.canvas.draw_idle()

    slider.on_changed(update)
    print("Visualization ready. Showing plot...")
    plt.show()

if __name__ == "__main__":
    visualize_currents()
