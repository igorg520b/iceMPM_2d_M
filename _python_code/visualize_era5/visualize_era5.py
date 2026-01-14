import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.widgets import Slider
import os

# --- Configuration ---
GRIB_FILE = 'data.grib'
LAT_MIN, LAT_MAX = 77.34, 82.40
LON_MIN, LON_MAX = -85.32, -43.42

def visualize_era5():
    if not os.path.exists(GRIB_FILE):
        print(f"Error: File {GRIB_FILE} not found.")
        return

    # Load the data
    print(f"Loading {GRIB_FILE}...")
    ds = xr.open_dataset(GRIB_FILE, engine='cfgrib')
    
    # Calculate wind speed
    ds['speed'] = np.sqrt(ds.u**2 + ds.v**2)
    ds.speed.attrs['units'] = 'm/s'
    ds.speed.attrs['long_name'] = 'Wind Speed'

    # Setup the figure and axis with Cartopy
    fig = plt.figure(figsize=(12, 8))
    # NorthPolarStereo provides the "view from top" (Arctic-centric) perspective
    central_lon = (LON_MIN + LON_MAX) / 2
    projection = ccrs.NorthPolarStereo(central_longitude=central_lon)
    ax = plt.subplot(1, 1, 1, projection=projection)

    # Set the extent to the area of interest
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Initial plot (first time step)
    time_idx = 0
    time_str = np.datetime_as_string(ds.time[time_idx].values, unit='h')
    
    speed_plot = ds.speed.isel(time=time_idx).plot(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap='viridis', add_colorbar=False,
        vmin=0, vmax=float(ds.speed.max())
    )
    
    # Add colorbar
    cbar = plt.colorbar(speed_plot, ax=ax, orientation='vertical', pad=0.05, aspect=30)
    cbar.set_label('Wind Speed (m/s)')

    # Add wind vectors (optional, can be cluttered)
    # Reducing density for better visibility
    skip = (slice(None, None, 2), slice(None, None, 8))
    u_data = ds.u.isel(time=time_idx).values[skip]
    v_data = ds.v.isel(time=time_idx).values[skip]
    lons, lats = np.meshgrid(ds.longitude.values[skip[1]], ds.latitude.values[skip[0]])
    
    quiver = ax.quiver(lons, lats, u_data, v_data, transform=ccrs.PlateCarree(),
                       scale=100, color='white', alpha=0.6)

    title = ax.set_title(f"ERA5 Wind Speed - {time_str}")

    # Add slider
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
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
        title.set_text(f"ERA5 Wind Speed - {time_val}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    print("Visualization ready. Showing plot...")
    plt.show()

if __name__ == "__main__":
    visualize_era5()
