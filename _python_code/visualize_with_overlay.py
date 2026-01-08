import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from matplotlib.widgets import Slider
import os
from PIL import Image

# --- Configuration ---
# Use paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRIB_FILE = os.path.join(SCRIPT_DIR, 'data.grib')
IMAGE_FILE = os.path.join(SCRIPT_DIR, '08crr.png')

# Orthographic projection parameters from the provided script
LAT_0 = 79.87
LON_0 = -64.37
R = 6371000

# Original transform coefficients (for the full-resolution image)
# [a, b, c, d, e, f, ...] -> [a, 0, c, 0, e, f, ...]
# a=11.796660738708361, c=-499251.71124660445, e=-11.796660738708361, f=336065.7037738611
TRANSFORM_COEFFS = [11.796660738708361, 0.0, -499251.71124660445, 0.0, -11.796660738708361, 336065.7037738611]

# Reducer factor mentioned by the user
RESIZE_FACTOR = 20

def visualize_overlay():
    if not os.path.exists(GRIB_FILE):
        print(f"Error: File {GRIB_FILE} not found.")
        return
    if not os.path.exists(IMAGE_FILE):
        print(f"Error: File {IMAGE_FILE} not found.")
        return

    # Load image and get dimensions
    img = Image.open(IMAGE_FILE)
    img_width, img_height = img.size
    print(f"Loaded image {IMAGE_FILE} ({img_width}x{img_height})")

    # Calculate extent in meters based on the full-resolution coefficients
    # and the original dimensions (which are img_size * RESIZE_FACTOR)
    a, _, c, _, e, f = TRANSFORM_COEFFS
    
    # x_min = c
    # x_max = c + (img_width * RESIZE_FACTOR) * a
    # y_max = f
    # y_min = f + (img_height * RESIZE_FACTOR) * e   (e is negative)
    
    x_min = c
    x_max = c + (img_width * RESIZE_FACTOR) * a
    y_min = f + (img_height * RESIZE_FACTOR) * e
    y_max = f
    
    img_extent = [x_min, x_max, y_min, y_max]
    print(f"Calculated image extent (meters): {img_extent}")

    # Load ERA5 data
    print(f"Loading {GRIB_FILE}...")
    ds = xr.open_dataset(GRIB_FILE, engine='cfgrib')
    ds['speed'] = np.sqrt(ds.u**2 + ds.v**2)

    # Setup the figure and axis with the specific Orthographic projection
    globe = ccrs.Globe(ellipse=None, semimajor_axis=R, semiminor_axis=R)
    projection = ccrs.Orthographic(central_latitude=LAT_0, central_longitude=LON_0, globe=globe)
    
    fig = plt.figure(figsize=(12, 10))
    ax = plt.subplot(1, 1, 1, projection=projection)

    # Set the extent to the image area
    ax.set_extent(img_extent, crs=projection)

    # Display the background image
    # Note: imshow in cartopy with a projection as 'crs' expects the extent in those projected coordinates
    ax.imshow(img, origin='upper', extent=img_extent, transform=projection)

    # Plot ERA5 Wind Speed as an overlay
    # Using alpha to see the background image
    time_idx = 0
    time_str = np.datetime_as_string(ds.time[time_idx].values, unit='h')
    
    speed_plot = ds.speed.isel(time=time_idx).plot(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap='YlOrRd', add_colorbar=False,
        vmin=0, vmax=float(ds.speed.max()),
        alpha=0.4  # Transparency to see the image beneath
    )
    
    # Add colorbar
    cbar = plt.colorbar(speed_plot, ax=ax, orientation='horizontal', pad=0.08, aspect=50)
    cbar.set_label('Wind Speed (m/s)')

    # Add wind vectors (white arrows)
    # Filter points for better visibility
    skip = (slice(None, None, 2), slice(None, None, 8))
    u_data = ds.u.isel(time=time_idx).values[skip]
    v_data = ds.v.isel(time=time_idx).values[skip]
    lons, lats = np.meshgrid(ds.longitude.values[skip[1]], ds.latitude.values[skip[0]])
    
    quiver = ax.quiver(lons, lats, u_data, v_data, transform=ccrs.PlateCarree(),
                       scale=150, color='white', alpha=0.8, width=0.003)

    title = ax.set_title(f"ERA5 Wind Speed Overlay - {time_str}")

    # Add slider
    ax_slider = plt.axes([0.15, 0.02, 0.7, 0.03])
    slider = Slider(ax_slider, 'Time Step', 0, len(ds.time) - 1, valinit=0, valfmt='%0.0f')

    def update(val):
        idx = int(slider.val)
        time_val = np.datetime_as_string(ds.time[idx].values, unit='h')
        
        # Update speed plot data
        speed_data = ds.speed.isel(time=idx).values
        speed_plot.set_array(speed_data.flatten())
        
        # Update quiver data
        u_new = ds.u.isel(time=idx).values[skip]
        v_new = ds.v.isel(time=idx).values[skip]
        quiver.set_UVC(u_new, v_new)
        
        # Update title
        title.set_text(f"ERA5 Wind Speed Overlay - {time_val}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    print("Visualization ready. Showing plot...")
    plt.show()

if __name__ == "__main__":
    visualize_overlay()
