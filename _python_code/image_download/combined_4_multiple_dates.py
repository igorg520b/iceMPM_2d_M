import sys
import os
import math
import json
from pathlib import Path

# --- Third-party library imports ---
# Ensure you have these installed:
# pip install sentinelhub-py imageio Pillow numpy pyproj rasterio

from sentinelhub import (
    SHConfig, MimeType, CRS, BBox, SentinelHubRequest, DataCollection,
    BBoxSplitter, bbox_to_dimensions
)
from imageio import imwrite
from PIL import Image
import numpy as np
import pyproj
import rasterio
from rasterio.transform import from_origin, Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling

# ==============================================================================
# --- 1. SCRIPT CONFIGURATION ---
# ==============================================================================

# --- Geographic Parameters (remain constant for all runs) ---

# Confederation Bridge
#LAT_MIN, LAT_MAX = 46.075, 46.377
#LON_MIN, LON_MAX = -64.12, -63.33

# Monbetsu
#LAT_MIN, LAT_MAX = 44.11, 46.747
#LON_MIN, LON_MAX = 141.7, 144.31
#TIME_INTERVALS = [
#    ("2022-01-27", "2022-01-27")
#    ("2024-02-06", "2024-02-06")
#]



# Nares Strait
LAT_MIN, LAT_MAX = 77.34, 82.40
LON_MIN, LON_MAX = -85.32, -43.42

    # --- Time Parameters: A LIST of time intervals to process ---
# Each tuple is a (start_date, end_date) for a separate run.
TIME_INTERVALS = [
    ("2024-06-08", "2024-06-08"),
    ("2024-06-20", "2024-06-20"),
    ("2024-06-10", "2024-06-10")
]

BASE_OUTPUT_FOLDER = "nares"


# Abegweit
#LAT_MIN, LAT_MAX = 45.57, 47.50
#LON_MIN, LON_MAX = -65.43, -60.51

#TIME_INTERVALS = [
#    ("2025-02-06", "2025-02-06"),
#    ("2025-02-11", "2025-02-11")
#]

#BASE_OUTPUT_FOLDER = "abegweit"

# --- Processing Parameters ---
RESOLUTION = 15  # meters

# --- Technical Parameters ---
TILE_SIZE_PX = 1024

# --- Post-Processing Options ---
SAVE_TILE_METADATA = True
CLEANUP_TILES = True

# --- Copernicus Data Space Ecosystem Credentials ---
#CLIENT_ID = "sh-48f9ee1f-2f0f-4c3b-8079-f883aa74d7d4"
#CLIENT_SECRET = "kCslDTrveF0k2EmWOCvTVGioujfJoILr"

CLIENT_ID = "sh-3f83e693-61a7-4ded-aed2-0a69a65e4b85"
CLIENT_SECRET = "nCJbv7HkgYzFJspfUnblQbmah7ZSjVDF"



# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================

def download_tiles(config, output_folder, time_interval):
    """Downloads tiles for a specific time interval."""
    print("--- STEP 1: Starting Tile Download ---")
    os.makedirs(output_folder, exist_ok=True)
    full_bbox = BBox(bbox=[LON_MIN, LAT_MIN, LON_MAX, LAT_MAX], crs=CRS.WGS84)
    full_size = bbox_to_dimensions(full_bbox, resolution=RESOLUTION)
    print(f"Full area dimensions at {RESOLUTION}m resolution: {full_size} pixels")

    n_tiles_x = math.ceil(full_size[0] / TILE_SIZE_PX)
    n_tiles_y = math.ceil(full_size[1] / TILE_SIZE_PX)
    bbox_splitter = BBoxSplitter([full_bbox], crs=CRS.WGS84, split_shape=(n_tiles_x, n_tiles_y))
    tile_bboxes, tile_info = bbox_splitter.get_bbox_list(), bbox_splitter.get_info_list()
    print(f"Splitting into a {n_tiles_x} x {n_tiles_y} grid. Total tiles: {len(tile_bboxes)}")

    evalscript = """
    //VERSION=3
    function setup() { return { input: ["B02", "B03", "B04", "dataMask"], output: { bands: 4 }}; }
    function evaluatePixel(s) { var gain = 0.8; return [s.B04 * gain*0.95, s.B03 * gain, s.B02 * gain, s.dataMask]; }
    """

    metadata_list = []
    for i, (tile_bbox, info) in enumerate(zip(tile_bboxes, tile_info)):
        row, col = info['index_y'], info['index_x']
        tile_pixel_size = bbox_to_dimensions(tile_bbox, resolution=RESOLUTION)
        print(f"-> Downloading tile [{row}, {col}] ({i + 1}/{len(tile_bboxes)})...")

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A.define_from(
                        "s2l2a", service_url="https://sh.dataspace.copernicus.eu"
                    ),
                    time_interval=time_interval,  # Use the passed-in interval
                    other_args={"dataFilter": {"mosaickingOrder": "leastCC"}}
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=tile_bbox, size=tile_pixel_size, config=config
        )

        try:
            image_tile = request.get_data()[0]
            filename = f"tile_{row}_{col}.png"
            filepath = os.path.join(output_folder, filename)
            imwrite(filepath, image_tile)

            tile_metadata = {
                'filename': filename, 'row': row, 'col': col,
                'bbox_wgs84': list(tile_bbox), 'pixel_size': tile_pixel_size
            }
            metadata_list.append(tile_metadata)
        except Exception as e:
            print(f"   ❌ ERROR downloading tile [{row}, {col}]: {e}")

    print("--- Tile Download Complete ---\n")
    return metadata_list


def save_tile_metadata(metadata, output_folder):
    """Saves the per-tile metadata to metadata.json."""
    print("--- STEP 2: Saving Per-Tile Metadata ---")
    metadata_path = os.path.join(output_folder, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"   ✅ Per-tile metadata saved to {metadata_path}\n")


def stitch_tiles(metadata, base_dir):
    """Stitches individual tiles into a single large image."""
    print("--- STEP 3: Stitching Tiles ---")
    lon_min = min(m["bbox_wgs84"][0] for m in metadata)
    lat_min = min(m["bbox_wgs84"][1] for m in metadata)
    lon_max = max(m["bbox_wgs84"][2] for m in metadata)
    lat_max = max(m["bbox_wgs84"][3] for m in metadata)
    extents = {'lon_min': lon_min, 'lat_min': lat_min, 'lon_max': lon_max, 'lat_max': lat_max}

    resolutions = [((m["bbox_wgs84"][2] - m["bbox_wgs84"][0]) / m["pixel_size"][0],
                    (m["bbox_wgs84"][3] - m["bbox_wgs84"][1]) / m["pixel_size"][1]) for m in metadata]
    ref_res_lon = min(abs(r[0]) for r in resolutions)
    ref_res_lat = min(abs(r[1]) for r in resolutions)

    final_width = int(round((lon_max - lon_min) / ref_res_lon))
    final_height = int(round((lat_max - lat_min) / ref_res_lat))
    print(f"Stitched image size: {final_width} x {final_height} pixels")

    stitched_image = Image.new("RGB", (final_width, final_height))

    def lon_to_x(lon):
        return int(round((lon - lon_min) / ref_res_lon))

    def lat_to_y(lat):
        return int(round((lat_max - lat) / ref_res_lat))

    for m in metadata:
        lon0, lat0, lon1, lat1 = m["bbox_wgs84"]
        x0, y0 = lon_to_x(lon0), lat_to_y(lat1)
        x1, y1 = lon_to_x(lon1), lat_to_y(lat0)
        tile_img = Image.open(os.path.join(base_dir, m['filename'])).convert("RGB")
        target_size = (x1 - x0, y1 - y0)
        if tile_img.size != target_size:
            tile_img = tile_img.resize(target_size, Image.Resampling.BILINEAR)
        stitched_image.paste(tile_img, (x0, y0))

    print("--- Tile Stitching Complete ---\n")
    return stitched_image, extents


def reproject_and_analyze(stitched_image, extents, base_dir):
    """Reprojects the image to Orthographic, saves it, and calculates pixel scale."""
    print("--- STEP 4: Reprojecting to Orthographic View & Analyzing ---")
    stitched_array = np.array(stitched_image)
    height, width, _ = stitched_array.shape
    lon0, lat0 = (extents['lon_min'] + extents['lon_max']) / 2, (extents['lat_min'] + extents['lat_max']) / 2
    src_crs, dst_crs = "EPSG:4326", pyproj.CRS.from_proj4(f"+proj=ortho +lat_0={lat0} +lon_0={lon0} +R=6371000")
    temp_tif_path = Path(base_dir) / "temp_stitched_platecarree.tif"

    try:
        res_lon, res_lat = (extents['lon_max'] - extents['lon_min']) / width, (
                    extents['lat_max'] - extents['lat_min']) / height
        src_transform = from_origin(extents['lon_min'], extents['lat_max'], res_lon, res_lat)
        with rasterio.open(
                temp_tif_path, "w", driver="GTiff", height=height, width=width,
                count=3, dtype=stitched_array.dtype, crs=src_crs, transform=src_transform
        ) as dst:
            for i in range(3): dst.write(stitched_array[:, :, i], i + 1)

        with rasterio.open(temp_tif_path) as src:
            ortho_transform, ortho_width, ortho_height = calculate_default_transform(src.crs, dst_crs, src.width,
                                                                                     src.height, *src.bounds)
            ortho_array = np.zeros((3, ortho_height, ortho_width), dtype=src.meta["dtype"])
            for i in range(3):
                reproject(source=rasterio.band(src, i + 1), destination=ortho_array[i],
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=ortho_transform, dst_crs=dst_crs, resampling=Resampling.bilinear)

        ortho_array = np.transpose(ortho_array, (1, 2, 0))
        out_png_path = Path(base_dir) / "final_orthographic_image.png"
        Image.fromarray(ortho_array).save(out_png_path)
        print(f"   ✅ Final orthographic image saved to {out_png_path}")

        geod = pyproj.Geod(ellps="WGS84")
        px_cx, px_cy = ortho_width // 2, ortho_height // 2
        x0_c, y0_c = rasterio.transform.xy(ortho_transform, px_cy, px_cx, offset="center")
        x1_c, y1_c = rasterio.transform.xy(ortho_transform, px_cy, px_cx + 1, offset="center")
        to_wgs84 = pyproj.Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True)
        lon0_c, lat0_c = to_wgs84.transform(x0_c, y0_c)
        lon1_c, lat1_c = to_wgs84.transform(x1_c, y1_c)
        dist_m, _, _ = geod.inv(lon0_c, lat0_c, lon1_c, lat1_c)
        print(f"   ✅ Approx pixel size at center: {dist_m:.2f} m/pixel")
        print("--- Reprojection & Analysis Complete ---\n")
        return dst_crs, ortho_transform, dist_m, ortho_width
    finally:
        if os.path.exists(temp_tif_path): os.remove(temp_tif_path)


def save_final_metadata(extents, final_image_width, output_folder):
    """
    Saves the final metadata after recalculating the pixel scale based on the
    ground distance of the lat_min parallel divided by the full image width.
    """
    print("--- STEP 5: Saving Final Image Metadata (with Custom Calculation) ---")

    # 1. Define the geodetic calculator for accurate distance measurement.
    geod = pyproj.Geod(ellps="WGS84")

    # 2. Get the endpoints of the reference parallel (the longer arc).
    lon_min, lon_max = extents['lon_min'], extents['lon_max']
    reference_lat = extents['lat_min']

    # 3. Calculate the true ground distance in meters along this parallel.
    _, _, distance_meters = geod.inv(lon_min, reference_lat, lon_max, reference_lat)

    # 4. Calculate the resolution using your formula.
    custom_resolution = distance_meters / final_image_width

    # 5. Update the dictionary with this new, authoritative value.
    extents['resolution_meters_per_pixel'] = round(custom_resolution, 2)
    # Remove the old key to avoid confusion
    if 'approx_meters_per_pixel_at_center' in extents:
        del extents['approx_meters_per_pixel_at_center']

    print(f"   Calculation based on your formula:")
    print(f"     - Ground distance along {reference_lat}° parallel: {distance_meters:,.2f} meters")
    print(f"     - Full image width: {final_image_width:,} pixels")
    print(f"   ✅ Final calculated pixel size: {custom_resolution:.2f} meters/pixel")

    # 6. Save the final JSON file.
    extents_path = os.path.join(output_folder, 'extents.json')
    with open(extents_path, 'w') as f:
        json.dump(extents, f, indent=4)
    print(f"   ✅ Final image metadata saved to {extents_path}\n")


def create_coordinate_converter_script(ortho_crs, ortho_transform, output_folder):
    """Generates a helper script for coordinate conversions."""
    print("--- STEP 6: Generating Coordinate Converter Script ---")
    script_content = f"""
# Auto-generated script for 'final_orthographic_image.png' in this directory.
import pyproj
from rasterio.transform import Affine

ORTHO_CRS_PROJ4 = "{ortho_crs.to_proj4()}"
TRANSFORM_COEFFS = {list(ortho_transform)}
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
"""
    script_path = os.path.join(output_folder, 'coordinate_converter.py')
    with open(script_path, 'w') as f: f.write(script_content)
    print(f"   ✅ Coordinate converter script saved to {script_path}\n")


def cleanup_tile_files(metadata, output_folder):
    """Deletes the individual downloaded tile images."""
    print("--- STEP 7: Cleaning Up Intermediate Files ---")
    deleted_count = 0
    for tile_info in metadata:
        filepath = os.path.join(output_folder, tile_info['filename'])
        if os.path.exists(filepath):
            os.remove(filepath)
            deleted_count += 1
    print(f"   ✅ Deleted {deleted_count} individual tile files.\n")


# ==============================================================================
# --- 3. PROCESSING PIPELINE FOR A SINGLE INTERVAL ---
# ==============================================================================

def process_time_interval(config, time_interval, base_output_folder):
    """
    Executes the full download and processing pipeline for a single time interval.
    """
    # Create a unique subdirectory for this run based on the start date
    start_date = time_interval[0]
    current_output_folder = os.path.join(base_output_folder, start_date)
    print(f"Output directory for this run: {current_output_folder}")

    # --- Execute the pipeline steps ---
    tile_metadata = download_tiles(config, current_output_folder, time_interval)
    if not tile_metadata:
        print(f"❌ No tiles were downloaded for {time_interval}. Skipping this interval.")
        return

    if SAVE_TILE_METADATA:
        save_tile_metadata(tile_metadata, current_output_folder)

    stitched_image, geo_extents = stitch_tiles(tile_metadata, current_output_folder)

    ortho_crs, ortho_transform, _, final_width = reproject_and_analyze(
        stitched_image, geo_extents, current_output_folder
    )

    save_final_metadata(geo_extents, final_width, current_output_folder)

    create_coordinate_converter_script(ortho_crs, ortho_transform, current_output_folder)

    if CLEANUP_TILES:
        cleanup_tile_files(tile_metadata, current_output_folder)


# ==============================================================================
# --- 4. MAIN EXECUTION SCRIPT ---
# ==============================================================================

if __name__ == "__main__":
    try:
        sh_config = SHConfig(
            sh_client_id=CLIENT_ID, sh_client_secret=CLIENT_SECRET,
            sh_base_url="https://sh.dataspace.copernicus.eu",
            sh_token_url="https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        )
        print("✅ Sentinel Hub configuration is set correctly.")
    except Exception as e:
        print(f"❌ Error setting up Sentinel Hub configuration: {e}");
        sys.exit(1)

    # --- Loop over all specified time intervals and process each one ---
    for interval in TIME_INTERVALS:
        print(f"\n{'=' * 80}\nProcessing time interval: {interval}\n{'=' * 80}")
        try:
            process_time_interval(sh_config, interval, BASE_OUTPUT_FOLDER)
        except Exception as e:
            print(f"🔥🔥🔥 An unexpected error occurred while processing interval {interval}: {e}")
            print("Moving to the next interval.")
            continue

    print(f"\n🎉🎉🎉 Full processing pipeline complete for all time intervals! 🎉🎉🎉")
    print(f"All outputs are located in the base directory: '{BASE_OUTPUT_FOLDER}'")