
# Auto-generated script for 'final_orthographic_image.png' in this directory.
import pyproj
from rasterio.transform import Affine

ORTHO_CRS_PROJ4 = "+proj=ortho +lat_0=79.87 +lon_0=-64.37 +x_0=0 +y_0=0 +R=6371000 +units=m +no_defs +type=crs"
TRANSFORM_COEFFS = [11.796660738708361, 0.0, -499251.71124660445, 0.0, -11.796660738708361, 336065.7037738611, 0.0, 0.0, 1.0]
ortho_crs = pyproj.CRS.from_proj4(ORTHO_CRS_PROJ4)
transform = Affine(*TRANSFORM_COEFFS)
to_wgs84 = pyproj.Transformer.from_crs(ortho_crs, "EPSG:4326", always_xy=True)
from_wgs84 = pyproj.Transformer.from_crs("EPSG:4326", ortho_crs, always_xy=True)

def pixel_to_lonlat(px, py):
    x, y = transform * (px + 0.5, py + 0.5)
    return to_wgs84.transform(x, y)
def lonlat_to_pixel(lon, lat):
    x, y = from_wgs84.transform(lon, lat)
    px, py = ~transform * (x, y)
    return int(px), int(py)
