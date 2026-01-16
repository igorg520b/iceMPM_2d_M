
import numpy as np
import matplotlib.pyplot as plt
import pyproj
import os

# ==========================================
# User-defined parameters
# ==========================================
WIDTH = 8464
HEIGHT = 5233
SCALE_FACTOR = 10.0  # Not directly used in calc, maybe reference?

# Projection Definition
ORTHO_CRS_PROJ4 = "+proj=ortho +lat_0=79.87 +lon_0=-64.37 +x_0=0 +y_0=0 +R=6371000 +units=m +no_defs +type=crs"

# Coefficients for Affine Transform: Pixel -> Projected Meters
# x_geo = coeffs[0]*px + coeffs[1]*py + coeffs[2]
# y_geo = coeffs[3]*px + coeffs[4]*py + coeffs[5]
ORIG_COEFFS = [11.796660738708361, 0.0, -499251.71124660445, 0.0, -11.796660738708361, 336065.7037738611]

LAT_MIN, LAT_MAX = 77.34, 82.40
LON_MIN, LON_MAX = -85.32, -43.42

# Downsampling for performance (full res 44MP is too slow for quick viz)
STEP = 20 

def main():
    print(f"Initializing Grid ({WIDTH}x{HEIGHT}) with Step {STEP}...")
    
    # Generate pixel grid (0.5 for center of pixel?)
    # User asked for "pixel (10,10)", let's assume centers or corners. 
    # Using integer indices i, j.
    x_idxs = np.arange(0, WIDTH, STEP)
    y_idxs = np.arange(0, HEIGHT, STEP)
    xx, yy = np.meshgrid(x_idxs, y_idxs)
    
    print("Computing Projected Coordinates (Meters)...")
    # Affine Transform to Meters
    # Coefficients are for the ORIGINAL FULL-RES IMAGE
    # We must scale our low-res coordinates to the original image space first.
    
    # Scale coordinates
    xx_orig = xx * SCALE_FACTOR
    yy_orig = yy * SCALE_FACTOR
    
    a, b, c = ORIG_COEFFS[0], ORIG_COEFFS[1], ORIG_COEFFS[2]
    d, e, f = ORIG_COEFFS[3], ORIG_COEFFS[4], ORIG_COEFFS[5]
    
    xm = a * xx_orig + b * yy_orig + c
    ym = d * xx_orig + e * yy_orig + f
    
    # Calculate neighbors for width/height calculation
    # We want the size of *this downscaled pixel*, so neighbor is +1 in low-res space.
    # In original space, this corresponds to a step of +SCALE_FACTOR.
    
    # Neighbor 1: (x+1, y) in low-res -> (x_orig + SCALE, y_orig)
    xm_dx = a * (xx_orig + SCALE_FACTOR) + b * yy_orig + c
    ym_dx = d * (xx_orig + SCALE_FACTOR) + e * yy_orig + f
    
    # Neighbor 2: (x, y+1) in low-res -> (x_orig, y_orig + SCALE)
    xm_dy = a * xx_orig + b * (yy_orig + SCALE_FACTOR) + c
    ym_dy = d * xx_orig + e * (yy_orig + SCALE_FACTOR) + f
    
    print("Inverse Projection to Lat/Lon...")
    # Setup Projection
    proj = pyproj.Proj(ORTHO_CRS_PROJ4)
    # Setup Geod for distance (Sphere with R=6371000)
    geod = pyproj.Geod(a=6371000, b=6371000)
    
    # Inverse Project Center
    lon, lat = proj(xm, ym, inverse=True)
    
    # Inverse Project Neighbors
    lon_dx, lat_dx = proj(xm_dx, ym_dx, inverse=True)
    lon_dy, lat_dy = proj(xm_dy, ym_dy, inverse=True)
    
    # Mask invalid points (inf generally means off-globe for ortho)
    # pyproj ortho might return inf or 1e30
    valid_mask = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(lon_dx) & np.isfinite(lon_dy)
    
    # Also clip to Region of Interest if desired, but user wants "Pixel of image"
    # The image might cover a larger area than Lat/Lon min/max? 
    # User provided Lat/Lon bounds, maybe we should crop visualization to that?
    # For now, visualizing the whole Image Frame data.
    
    print("Computing Distances on Surface...")
    # Compute Width (dx) and Height (dy) in meters
    # inv returns (az12, az21, dist)
    _, _, dist_w = geod.inv(lon, lat, lon_dx, lat_dx)
    _, _, dist_h = geod.inv(lon, lat, lon_dy, lat_dy)
    
    # Apply Mask
    dist_w[~valid_mask] = np.nan
    dist_h[~valid_mask] = np.nan
    
    print("Computing Metrics...")
    # Metrics
    avg_size = (dist_w + dist_h) / 2.0
    aspect_ratio = dist_w / dist_h
    area = dist_w * dist_h
    
    # ==========================================
    # Plotting
    # ==========================================
    print("Plotting...")
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    
    # Shared extent for imshow
    extent = [0, WIDTH, HEIGHT, 0] # Standard Image coords (0,0 top-left)
    
    # 1. Average Size
    im0 = axs[0].imshow(avg_size, extent=extent, cmap='viridis')
    axs[0].set_title("Average Pixel Size (m)")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    
    # 2. Aspect Ratio (Width / Height)
    # Use diverging colormap centered at 1.0
    im1 = axs[1].imshow(aspect_ratio, extent=extent, cmap='coolwarm', vmin=0.9, vmax=1.1)
    axs[1].set_title("Aspect Ratio (Width/Height)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    
    # 3. Pixel Area
    im2 = axs[2].imshow(area, extent=extent, cmap='plasma')
    axs[2].set_title("Pixel Area (m^2)")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    # 4. Lat/Lon Overlay (Simple Grid Visualization)
    # Just plot lat/lon as value
    im3 = axs[3].imshow(lat, extent=extent, cmap='gray')
    axs[3].set_title("Latitude")
    plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)


    # Attempt Coastline Overlay (without Cartopy dependency if possible, else skip)
    # Try importing cartopy only for feature fetching, but projecting manually is hard.
    # Simple workaround: Don't implement overlay for this quick script unless requested strictly.
    # User said "If possible".
    
    # Save or Show
    plt.tight_layout()
    output_file = "projection_distortion_analysis.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved visualization to {output_file}")
    plt.show()

if __name__ == "__main__":
    main()
