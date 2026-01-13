#include "windandcurrentinterpolator.h"
#include <H5Cpp.h>
#include <spdlog/spdlog.h>
#include <cmath>
#include <filesystem>
#include <fmt/format.h>
#include <algorithm> // for upper_bound
#include <utility>
#include <omp.h>

WindAndCurrentInterpolator::WindAndCurrentInterpolator(SimParams& params) : prms(params)
{
}

WindAndCurrentInterpolator::~WindAndCurrentInterpolator() = default;


void WindAndCurrentInterpolator::SetHDF5Path(const std::string& filePath)
{
    // Check file exists before attempting to open
    if (!std::filesystem::exists(filePath)) {
        throw std::runtime_error(fmt::format("Flow field file not found: {}", filePath));
    }

    hdf5_path = filePath;

    // Lazily open file and load HDF5 metadata (currents)
    LoadHDF5Metadata();
    
    // Lazily open file and load HDF5 metadata (currents)
    LoadHDF5Metadata();
}

void WindAndCurrentInterpolator::SetEra5Path(const std::string& filePath)
{
    // Check file exists before attempting to open
    if (!std::filesystem::exists(filePath)) {
        throw std::runtime_error(fmt::format("ERA5 file not found: {}", filePath));
    }
    
    era5_path = filePath;

    if (prms.UseWindData) {
        LoadEra5Metadata();
    }
}


void WindAndCurrentInterpolator::LoadHDF5Metadata()
{
    if (hdf5_path.empty()) {
        return;
    }

    file_flow = std::make_unique<H5::H5File>(hdf5_path, H5F_ACC_RDONLY);

    // 1. Read Flow Type from the root group attribute
    H5::Group root = file_flow->openGroup("/");
    
    // Check if flow_type exists
    if(root.attrExists("flow_type")) {
        H5::Attribute attr_type = root.openAttribute("flow_type");
        H5::StrType str_type = attr_type.getStrType();
        size_t len = str_type.getSize();
        std::vector<char> buf(len + 1, '\0');
        attr_type.read(str_type, buf.data());
        flow_type_id = std::string(buf.data());
    } else {
        flow_type_id = "default";
    }

    // Standard wave/current flow
    H5::DataSet ds_vx = file_flow->openDataSet("water_current_vx");
    H5::DataSpace space = ds_vx.getSpace();
    hsize_t dims[3];
    space.getSimpleExtentDims(dims, NULL);
    num_frames = static_cast<int>(dims[0]);
    gx = static_cast<int>(dims[1]);
    gy = static_cast<int>(dims[2]);

    ds_vx.openAttribute("time_interval").read(H5::PredType::NATIVE_DOUBLE, &time_interval);
    ds_vx.openAttribute("loop_mode").read(H5::PredType::NATIVE_INT, &loop_mode);
}

void WindAndCurrentInterpolator::LoadEra5Metadata()
{
    if (era5_path.empty()) return;
    
    spdlog::info("Loading ERA5 Metadata from {}", era5_path);
    file_wind = std::make_unique<H5::H5File>(era5_path, H5F_ACC_RDONLY);
    
    // Load Time
    {
        H5::DataSet ds_time = file_wind->openDataSet("valid_time");
        H5::DataSpace space = ds_time.getSpace();
        hsize_t dims[1];
        space.getSimpleExtentDims(dims, NULL);
        era5_num_frames = static_cast<int>(dims[0]);
        era5_times.resize(era5_num_frames);
        ds_time.read(era5_times.data(), H5::PredType::NATIVE_LLONG); // Linux timestamp
        
        if (era5_times.empty()) throw std::runtime_error("ERA5 time dataset empty");
        era5_start_time = era5_times[0];
        spdlog::info("ERA5 Start Time: {}", era5_start_time);
    }
    
    // Load Latitude
    {
        H5::DataSet ds_lat = file_wind->openDataSet("latitude");
        H5::DataSpace space = ds_lat.getSpace();
        hsize_t dims[1];
        space.getSimpleExtentDims(dims, NULL);
        era5_lats.resize(dims[0]);
        ds_lat.read(era5_lats.data(), H5::PredType::NATIVE_DOUBLE);
    }

    // Load Longitude
    {
        H5::DataSet ds_lon = file_wind->openDataSet("longitude");
        H5::DataSpace space = ds_lon.getSpace();
        hsize_t dims[1];
        space.getSimpleExtentDims(dims, NULL);
        era5_lons.resize(dims[0]);
        ds_lon.read(era5_lons.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    spdlog::info("ERA5 Grid: {}x{} (Lat/Lon)", era5_lats.size(), era5_lons.size());
}


void WindAndCurrentInterpolator::LoadOceanFrame(int frameIdx, int bufferSlot)
{
    if (!file_flow) return;

    // Clamp frame index based on loop mode
    int actualIdx = frameIdx;
    if (loop_mode == 0) {
        // Periodic: wrap around
        actualIdx = frameIdx % num_frames;
        if (actualIdx < 0) actualIdx += num_frames;
    } else {
        // Hold last frame
        if (actualIdx >= num_frames) actualIdx = num_frames - 1;
        if (actualIdx < 0) actualIdx = 0;
    }

    int gridSize = gx * gy;
    ocean_vx_frame_buffer[bufferSlot].resize(gridSize);
    ocean_vy_frame_buffer[bufferSlot].resize(gridSize);

    try {
        H5::DataSet ds_vx = file_flow->openDataSet("water_current_vx");
        H5::DataSet ds_vy = file_flow->openDataSet("water_current_vy");

        hsize_t offset[3] = {static_cast<hsize_t>(actualIdx), 0, 0};
        hsize_t dims[3] = {1, static_cast<hsize_t>(gx), static_cast<hsize_t>(gy)};

        H5::DataSpace space_vx = ds_vx.getSpace();
        space_vx.selectHyperslab(H5S_SELECT_SET, dims, offset);
        H5::DataSpace mem_space(3, dims);

        // Read directly as float
        ds_vx.read(ocean_vx_frame_buffer[bufferSlot].data(), H5::PredType::NATIVE_FLOAT, mem_space, space_vx);
        ds_vy.read(ocean_vy_frame_buffer[bufferSlot].data(), H5::PredType::NATIVE_FLOAT, mem_space, space_vx);

    } catch (const H5::Exception& e) {
        spdlog::error("HDF5 Error loading flow frame: {}", e.getDetailMsg());
        throw std::runtime_error("Failed to load flow field frame from HDF5");
    }
}

// --------------------------------------------------------------------------------------
// Projection & Rotation Logic
// --------------------------------------------------------------------------------------

WindAndCurrentInterpolator::LatLon WindAndCurrentInterpolator::ProjectPixel(int x, int y) const
{
    // 1. Pixel to Geo Coordinates (meters)
    // Transform Coeffs: [a, b, c, d, e, f] -> x' = ax + by + c, y' = dx + ey + f
    // x, y are input pixel indices (local to the full image)
    
    double px = (double)x * prms.PROJ_RESIZE_FACTOR;
    double py = (double)y * prms.PROJ_RESIZE_FACTOR;
    
    double x_geo = prms.PROJ_TRANSFORM_COEFFS[0] * px + prms.PROJ_TRANSFORM_COEFFS[1] * py + prms.PROJ_TRANSFORM_COEFFS[2];
    double y_geo = prms.PROJ_TRANSFORM_COEFFS[3] * px + prms.PROJ_TRANSFORM_COEFFS[4] * py + prms.PROJ_TRANSFORM_COEFFS[5];
    
    // 2. Inverse Orthographic
    double rho = std::sqrt(x_geo*x_geo + y_geo*y_geo);
    double c = 0.0;
    double R = prms.PROJ_R;
    
    // Inverse projection is only valid for rho <= R (technically), but we handle gracefully
    if (rho > R) return {0,0,false}; // Off globe (horizon)
    
    c = std::asin(rho / R);
    
    double phi0 = prms.PROJ_LAT_0 * (M_PI/180.0);
    double lam0 = prms.PROJ_LON_0 * (M_PI/180.0);
    
    double phi = 0.0, lam = 0.0;
    
    if (rho == 0.0) {
        phi = phi0;
        lam = lam0;
    } else {
        double cos_c = std::cos(c);
        double sin_c = std::sin(c);
        
        phi = std::asin(cos_c * std::sin(phi0) + (y_geo * sin_c * std::cos(phi0)) / rho);
        
        double num = x_geo * sin_c;
        double den = rho * std::cos(phi0) * cos_c - y_geo * std::sin(phi0) * sin_c;
        lam = lam0 + std::atan2(num, den);
    }
    
    return {phi * (180.0/M_PI), lam * (180.0/M_PI), true};
}

WindAndCurrentInterpolator::RotMat WindAndCurrentInterpolator::ComputeRotation(double phi, double lam) const
{
    // Compute local North and East direction vectors projected onto Grid
    // We use numerical forward differentiation: moving slightly North/East on sphere -> how much x_geo/y_geo changes
    // This gives us the transformation from (u,v) to (vx_grid, vy_grid)
    
    // NOTE: Requires (phi, lam) in RADIANS for trig functions
    
    double delta = 1e-5; // small radian step
    
    auto Fwd = [&](double p, double l) -> std::pair<double,double> {
        double phi0 = prms.PROJ_LAT_0 * (M_PI/180.0);
        double lam0 = prms.PROJ_LON_0 * (M_PI/180.0);
        double R = prms.PROJ_R;
        
        // Sphere -> Orthographic Plane
        double x = R * std::cos(p) * std::sin(l - lam0);
        double y = R * (std::cos(phi0) * std::sin(p) - std::sin(phi0) * std::cos(p) * std::cos(l - lam0));
        return {x, y};
    };
    
    std::pair<double, double> P = Fwd(phi, lam);
    
    // North neighbor
    std::pair<double, double> Pn = Fwd(phi + delta, lam);
    double dx_n = Pn.first - P.first;
    double dy_n = Pn.second - P.second;
    double len_n = std::sqrt(dx_n*dx_n + dy_n*dy_n);
    if(len_n < 1e-9) len_n = 1.0;
    
    // East neighbor
    // Adjust delta for longitude to keep roughly same arc length? Not strictly needed for direction.
    std::pair<double, double> Pe = Fwd(phi, lam + delta);
    double dx_e = Pe.first - P.first;
    double dy_e = Pe.second - P.second;
    double len_e = std::sqrt(dx_e*dx_e + dy_e*dy_e);
    if(len_e < 1e-9) len_e = 1.0;
    
    return {dx_e/len_e, dy_e/len_e, dx_n/len_n, dy_n/len_n};
}


void WindAndCurrentInterpolator::LoadWindFrame(int frameIdx, int bufferSlot)
{
    if (!file_wind || !prms.UseWindData) return;
    
    // 1. Read Raw ERA5 Frame (Entire slice)
    // u(time, level, lat, lon) -> we want 2D slice at specific time, level 0
    std::vector<float> raw_u, raw_v; // use float to save temp memory, sufficient precision
    int n_lat = era5_lats.size();
    int n_lon = era5_lons.size();
    size_t n_elem = n_lat * n_lon;
    
    raw_u.resize(n_elem);
    raw_v.resize(n_elem);
    
    try {
        H5::DataSet ds_u = file_wind->openDataSet("u");
        H5::DataSet ds_v = file_wind->openDataSet("v");
        // Check dims to see if 3D or 4D. User said u(time, level, lat, lon).
        // Let's assume 4D. If level is squeezed, might be 3D.
        
        H5::DataSpace space = ds_u.getSpace();
        int ndims = space.getSimpleExtentNdims();
        std::vector<hsize_t> dims_file(ndims);
        space.getSimpleExtentDims(dims_file.data(), NULL);

        std::vector<hsize_t> start(ndims, 0);
        std::vector<hsize_t> count(ndims, 1);
        
        // Time index
        start[0] = frameIdx;
        
        // Lat/Lon are last two
        count[ndims-2] = n_lat;
        count[ndims-1] = n_lon;
        
        space.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());
        
        // Mem space needs to match the selection (2D flattened or specific shape)
        // We read into flat buffer, so mem space can be simple 1D or matched 2D
        hsize_t mem_dims[2] = {(hsize_t)n_lat, (hsize_t)n_lon}; 
        H5::DataSpace mem_space(2, mem_dims);
        
        ds_u.read(raw_u.data(), H5::PredType::NATIVE_FLOAT, mem_space, space);
        ds_v.read(raw_v.data(), H5::PredType::NATIVE_FLOAT, mem_space, space);
        
    } catch (const H5::Exception& e) {
        spdlog::error("Error reading ERA5 frame {}: {}", frameIdx, e.getDetailMsg());
        return;
    }
    
    // 2. Interpolate to Simulation Grid
    int gridSize = gx * gy;
    wind_vx_frame_buffer[bufferSlot].resize(gridSize);
    wind_vy_frame_buffer[bufferSlot].resize(gridSize);
    
    // Ensure ERA5 coords are sorted as expected: Lat Descending, Lon Ascending
    bool lat_descending = era5_lats.size() > 1 && era5_lats[0] > era5_lats[1];
    
    // Parallel Loop
    #pragma omp parallel for
    for (int j = 0; j < gy; ++j) {
        for (int i = 0; i < gx; ++i) {
            int grid_idx = j + i*gy;
            
            // Global pixel coordinates
            // Assuming InitializationImageSize aligns with projection size logic
            // Assuming ModeledRegionOffset relates to this full image.
            int global_x = i + prms.ModeledRegionOffsetX;
            // Invert Y because PROJ coeffs are based on top-left origin image, 
            // while grid is bottom-left origin.
            int global_y_grid = j + prms.ModeledRegionOffsetY;
            int global_y = prms.InitializationImageSizeY - 1 - global_y_grid;
            
            // 1. Project
            LatLon ll = ProjectPixel(global_x, global_y);
            if (!ll.valid) {
                 wind_vx_frame_buffer[bufferSlot][grid_idx] = 0.0f;
                 wind_vy_frame_buffer[bufferSlot][grid_idx] = 0.0f;
                 continue;
            }
            
            // 2. Find Float Index in ERA5
            // Latitude
            double r_idx = 0;
            if (lat_descending) {
                 // Map lat to index. Lats are decreasing: 90 ... -90
                 // idx = (lat - lat0) / dlat. dlat is negative.
                 double dlat = era5_lats[1] - era5_lats[0]; // negative
                 r_idx = (ll.lat_deg - era5_lats[0]) / dlat;
            } else {
                 double dlat = era5_lats[1] - era5_lats[0];
                 r_idx = (ll.lat_deg - era5_lats[0]) / dlat;
            }
            
            // Longitude
            // Need to handle conventions: ERA5 0..360 vs -180..180
            // We assume ERA5 is 0..360 if it goes > 180.
            // Our ll.lon_deg is -180..180.
            double target_lon = ll.lon_deg;
            // Scan era5_lons to see range?
            bool era5_is_0_360 = (era5_lons.back() > 180.0);
            if (era5_is_0_360 && target_lon < 0) target_lon += 360.0;
            
            double dlon = era5_lons[1] - era5_lons[0];
            double c_idx = (target_lon - era5_lons[0]) / dlon;
            
            // 3. Bilinear Interpolate
            // Clamp to edges (or handle wrap for lon). For now clamp is safer.
            // But Lon wrapping is cleaner.
            
            auto getColor = [&](const std::vector<float>& src, double r, double c) -> double {
                int r0 = (int)std::floor(r);
                int c0 = (int)std::floor(c);
                int r1 = r0 + 1;
                int c1 = c0 + 1;
                
                double dr = r - r0;
                double dc = c - c0;
                
                // Clamp Row
                if (r0 < 0) r0 = 0; if (r1 >= n_lat) r1 = n_lat-1;
                if (r0 >= n_lat) r0 = n_lat-1; // shouldn't happen if clamped
                
                // Wrap Col
                // This assumes full coverage 0..360. If regional, clamp.
                // Assuming global coverage for now as is typical for ERA5.
                c0 = (c0 % n_lon + n_lon) % n_lon;
                c1 = (c1 % n_lon + n_lon) % n_lon;
                
                float v00 = src[c0 + r0*n_lon];
                float v01 = src[c1 + r0*n_lon];
                float v10 = src[c0 + r1*n_lon];
                float v11 = src[c1 + r1*n_lon];
                
                double top = v00 * (1.0 - dc) + v01 * dc;
                double bot = v10 * (1.0 - dc) + v11 * dc;
                return top * (1.0 - dr) + bot * dr;
            };
            
            double u_val = getColor(raw_u, r_idx, c_idx);
            double v_val = getColor(raw_v, r_idx, c_idx);
            
            // 4. Rotate
            // Need phi/lam in radians for rotation calc
            RotMat rot = ComputeRotation(ll.lat_deg * (M_PI/180.0), ll.lon_deg * (M_PI/180.0));
            
            double vx_grid = u_val * rot.ex + v_val * rot.nx;
            double vy_grid = u_val * rot.ey + v_val * rot.ny;
            
            wind_vx_frame_buffer[bufferSlot][grid_idx] = (float)vx_grid;
            wind_vy_frame_buffer[bufferSlot][grid_idx] = (float)vy_grid;
        }
    }
}


// Helper for Smart Slot Management (Ring Buffer of 3 slots for 2 needed frames)
void WindAndCurrentInterpolator::UpdateRingBufferSlots(int needed_f0, int needed_f1, int* slot_frames, int* active_slots, const std::function<void(int, int)>& load_func)
{
    // Goal: We need physical slots for 'needed_f0' and 'needed_f1'.
    // We have 3 physical slots (NUM_SLOTS).
    // We preserve existing data if possible (Reuse).
    // If we must load, we pick an optimal slot to overwrite (Evict).

    // --- 1. Resolve Slot for First Frame (needed_f0) ---
    int s0 = -1;
    
    // Search: Is needed_f0 already in memory?
    for(int k=0; k<NUM_SLOTS; ++k) { 
        if(slot_frames[k] == needed_f0) { 
            s0 = k; 
            break; // Found! Reuse it.
        } 
    }
    
    // Not found? Need to load it. Find a victim slot to overwrite.
    if (s0 == -1) {
        // Eviction Policy:
        // 1. Do NOT evict a slot that contains the *other* frame we need (needed_f1).
        // 2. Prefer empty slots (-1).
        // 3. Otherwise, pick any slot (e.g., k=0, or LRU if we tracked it, but simple round-robin or first-available is fine for 3-slot/2-active case).
        
        int best_cand = -1;
        
        for(int k=0; k<NUM_SLOTS; ++k) {
            if (slot_frames[k] == needed_f1) continue; // CRITICAL: Protect the sibling frame!
            
            if (slot_frames[k] == -1) { 
                best_cand = k; 
                break; // Found an empty slot, perfect.
            }
            // If not empty but not protected, it's a valid candidate for overwrite.
            best_cand = k; 
        }
        
        if (best_cand == -1) {
            // This theoretically shouldn't happen given NUM_SLOTS=3 and only 1 protected frame (needed_f1).
            // But as a fallback, just overwrite slot 0.
            best_cand = 0; 
        }
        s0 = best_cand;
        
        // Perform the Load
        load_func(needed_f0, s0);
        slot_frames[s0] = needed_f0; // Mark this slot as holding frame f0
    }

    // --- 2. Resolve Slot for Second Frame (needed_f1) ---
    int s1 = -1;
    
    // Search: Is needed_f1 already in memory?
    for(int k=0; k<NUM_SLOTS; ++k) { 
        if(slot_frames[k] == needed_f1) { 
            s1 = k; 
            break; // Found! Reuse it.
        } 
    }
    
    // Not found? Load it.
    if (s1 == -1) {
        // Eviction Policy:
        // 1. Do NOT evict the slot we just assigned to f0 (s0).
        // 2. Prefer empty slots.
        
        int best_cand = -1;
        for(int k=0; k<NUM_SLOTS; ++k) {
            if (k == s0) continue; // CRITICAL: Protect the frame we just locked!
            
            if (slot_frames[k] == -1) { 
                best_cand = k; 
                break; 
            }
            best_cand = k;
        }
        
        // Fallback (simple rotation) if loop didn't pick (unlikely)
        if (best_cand == -1) best_cand = (s0 + 1) % NUM_SLOTS;
        s1 = best_cand;

        load_func(needed_f1, s1);
        slot_frames[s1] = needed_f1;
    }
    
    // Output the mapping: Logical 0 -> s0, Logical 1 -> s1
    active_slots[0] = s0;
    active_slots[1] = s1;
}


std::pair<bool, bool> WindAndCurrentInterpolator::SetTime(double t)
{
    // --- 0. Wait for any pending async preloads ---
    if (ocean_preload_future.valid()) ocean_preload_future.wait();
    if (wind_preload_future.valid()) wind_preload_future.wait();
    
    // --- 1. OCEAN CURRENT (Standard) ---
    bool ocean_frames_changed = false;
    if (num_frames > 0) {
        // Calculate logical frames based on Time 't'
        double frame_idx_f = (time_interval > 0.0) ? (t / time_interval) : 0.0;

        // Apply Loop Mode (Periodic vs Hold)
        if (loop_mode == 0) { // Periodic
            frame_idx_f = std::fmod(frame_idx_f, static_cast<double>(num_frames));
            if (frame_idx_f < 0.0) frame_idx_f += num_frames;
        } else { // Hold Last
            if (frame_idx_f > num_frames - 1) frame_idx_f = num_frames - 1;
            if (frame_idx_f < 0.0) frame_idx_f = 0.0;
        }

        int first_idx = static_cast<int>(std::floor(frame_idx_f));
        int second_idx = first_idx;
        double alpha = frame_idx_f - first_idx;

        if (loop_mode == 0) second_idx = (first_idx + 1) % num_frames;
        else second_idx = std::min(first_idx + 1, num_frames - 1);

        // Check if our interpolation window has shifted
        bool time_window_changed = (first_idx != current_ocean_first_idx || second_idx != current_ocean_second_idx);
        bool sequential_flow = false;
        
        // Check for sequential flow: Did we move from [n, n+1] to [n+1, n+2]?
        // Basically, if the new first_idx is exactly the old second_idx (or old first_idx + 1)
        if (current_ocean_first_idx != -1) {
             int expected_next = (current_ocean_first_idx + 1);
             if (loop_mode == 0) expected_next %= num_frames;
             else expected_next = std::min(expected_next, num_frames - 1);
             
             if (first_idx == expected_next) sequential_flow = true;
        }
        
        if (time_window_changed) {
            ocean_frames_changed = true;
            current_ocean_first_idx = first_idx;
            current_ocean_second_idx = second_idx;
            
            // Delegate buffer management to helper (Synchronous for current frames)
            UpdateRingBufferSlots(first_idx, second_idx, ocean_slot_frames, current_ocean_active_slots, 
                [&](int f, int s){ LoadOceanFrame(f, s); });
            
            // ASYNC PRELOAD LOGIC
            // If we are moving sequentially, try to preload the NEXT frame (n+2)
            if (sequential_flow) {
                int next_preload_idx = -1;
                
                if (loop_mode == 0) { // Periodic
                    next_preload_idx = (second_idx + 1) % num_frames;
                } else { // Hold Last
                    if (second_idx < num_frames - 1) next_preload_idx = second_idx + 1;
                }
                
                // Only preload if valid frame and not already loaded
                if (next_preload_idx != -1) {
                    // Check if already in slot
                    bool already_loaded = false;
                    for(int k=0; k<NUM_SLOTS; ++k) if(ocean_slot_frames[k] == next_preload_idx) already_loaded = true;
                    
                    if (!already_loaded) {
                        // Find the spare slot (not used by first_idx or second_idx)
                        int spare_slot = -1;
                        for(int k=0; k<NUM_SLOTS; ++k) {
                            if (k != current_ocean_active_slots[0] && k != current_ocean_active_slots[1]) {
                                spare_slot = k;
                                break;
                            }
                        }
                        
                        // Launch Async Load
                        if (spare_slot != -1) {
                            // Mark slot as occupied by this frame (so future SetTime calls see it)
                            ocean_slot_frames[spare_slot] = next_preload_idx;
                            
                            // Capture by value carefully
                            ocean_preload_future = std::async(std::launch::async, [this, next_preload_idx, spare_slot]() {
                                LoadOceanFrame(next_preload_idx, spare_slot);
                            });
                        }
                    }
                }
            }
        }
        current_ocean_alpha = alpha;
    }
    
    // --- 2. WIND (ERA5) ---
    bool wind_frames_changed = false;
    if (prms.UseWindData && era5_num_frames > 0) {
         long long target_ts = era5_start_time + (long long)t;
         auto it = std::upper_bound(era5_times.begin(), era5_times.end(), target_ts);
         int idx = (it == era5_times.begin()) ? 0 : std::distance(era5_times.begin(), it) - 1;
         
         if (idx >= era5_num_frames - 1) idx = era5_num_frames - 2;
         if (idx < 0) idx = 0;
         
         int w_first = idx;
         int w_second = idx + 1;
         
         long long t0 = era5_times[w_first];
         long long t1 = era5_times[w_second];
         double dt = (double)(t1 - t0);
         double w_alpha = (dt > 1e-3) ? (double)(target_ts - t0) / dt : 0.0;
         if (w_alpha < 0) w_alpha = 0; if (w_alpha > 1) w_alpha = 1;

         bool w_sequential = false;
         if (current_wind_first_idx != -1) {
             if (w_first == current_wind_first_idx + 1) w_sequential = true;
         }
         
         if (w_first != current_wind_first_idx || w_second != current_wind_second_idx) {
             wind_frames_changed = true;
             current_wind_first_idx = w_first;
             current_wind_second_idx = w_second;
             
             spdlog::info("Loading Wind Frames {} and {}; using Ring Buffer logic", w_first, w_second);
             
             UpdateRingBufferSlots(w_first, w_second, wind_slot_frames, current_wind_active_slots,
                [&](int f, int s){ LoadWindFrame(f, s); });
                
             // Async Preload for Wind
             if (w_sequential) {
                 int next_preload_idx = -1;
                 // ERA5 is typically non-periodic in this context (linear time series)
                 if (w_second < era5_num_frames - 1) next_preload_idx = w_second + 1;
                 
                 if (next_preload_idx != -1) {
                     bool already_loaded = false;
                     for(int k=0; k<NUM_SLOTS; ++k) if(wind_slot_frames[k] == next_preload_idx) already_loaded = true;
                     
                     if (!already_loaded) {
                         int spare_slot = -1;
                         for(int k=0; k<NUM_SLOTS; ++k) {
                             if (k != current_wind_active_slots[0] && k != current_wind_active_slots[1]) {
                                 spare_slot = k;
                                 break;
                             }
                         }
                         
                         if (spare_slot != -1) {
                             wind_slot_frames[spare_slot] = next_preload_idx;
                             wind_preload_future = std::async(std::launch::async, [this, next_preload_idx, spare_slot]() {
                                 LoadWindFrame(next_preload_idx, spare_slot);
                             });
                         }
                     }
                 }
             }
         }
         current_wind_alpha = w_alpha;
    }

    return {ocean_frames_changed, wind_frames_changed};
}


std::pair<double, double> WindAndCurrentInterpolator::GetOceanValue(int i, int j) const
{
    if (hdf5_path.empty() || num_frames == 0) return {0.0, 0.0};
    if (i < 0 || i >= gx || j < 0 || j >= gy) return {0.0, 0.0};

    size_t idx = j + static_cast<size_t>(i) * gy;

    // Use current active slots
    int s0 = current_ocean_active_slots[0];
    int s1 = current_ocean_active_slots[1];

    if (num_frames == 1) {
        return {(double)ocean_vx_frame_buffer[s0][idx], (double)ocean_vy_frame_buffer[s0][idx]};
    }

    double vx_first = ocean_vx_frame_buffer[s0][idx];
    double vx_second = ocean_vx_frame_buffer[s1][idx];
    double vy_first = ocean_vy_frame_buffer[s0][idx];
    double vy_second = ocean_vy_frame_buffer[s1][idx];

    double vx = (1.0 - current_ocean_alpha) * vx_first + current_ocean_alpha * vx_second;
    double vy = (1.0 - current_ocean_alpha) * vy_first + current_ocean_alpha * vy_second;
    
    return {vx, vy};
}

std::pair<double, double> WindAndCurrentInterpolator::GetWindValue(int i, int j) const
{
    if (!prms.UseWindData) return {0.0, 0.0};
    // Ensure we have loaded data (check active slots - although SetTime ensures they are set)
    // Note: Checking empty() on slot 0 is not enough as active slot might vary.
    // Just check the active slots.
    int s0 = current_wind_active_slots[0];
    int s1 = current_wind_active_slots[1];
    
    if (wind_vx_frame_buffer[s0].empty()) return {0.0, 0.0};

    size_t idx = (size_t)i * gy + j;
    if (idx >= wind_vx_frame_buffer[s0].size()) return {0.0, 0.0};

    double vx0 = wind_vx_frame_buffer[s0][idx];
    double vx1 = wind_vx_frame_buffer[s1][idx];
    double vy0 = wind_vy_frame_buffer[s0][idx];
    double vy1 = wind_vy_frame_buffer[s1][idx];

    double vx = vx0 * (1.0 - current_wind_alpha) + vx1 * current_wind_alpha;
    double vy = vy0 * (1.0 - current_wind_alpha) + vy1 * current_wind_alpha;

    return {vx, vy};
}

const float* WindAndCurrentInterpolator::GetOceanDataPointer(int logicalFrame, int component) const
{
    // logicalFrame: 0 or 1
    if (logicalFrame < 0 || logicalFrame > 1) return nullptr;
    int slot = current_ocean_active_slots[logicalFrame];
    
    if (component == 0) return ocean_vx_frame_buffer[slot].data();
    else return ocean_vy_frame_buffer[slot].data();
}

const float* WindAndCurrentInterpolator::GetWindDataPointer(int logicalFrame, int component) const
{
    if (logicalFrame < 0 || logicalFrame > 1) return nullptr;
    int slot = current_wind_active_slots[logicalFrame];
    
    if (component == 0) return wind_vx_frame_buffer[slot].data();
    else return wind_vy_frame_buffer[slot].data();
}

std::pair<double, double> WindAndCurrentInterpolator::GetLatLon(int i, int j) const
{
    // Global pixel coordinates
    int global_x = i + prms.ModeledRegionOffsetX;
    
    // Invert Y because PROJ coeffs are based on top-left origin image, 
    // while grid is bottom-left origin.
    int global_y_grid = j + prms.ModeledRegionOffsetY;
    int global_y = prms.InitializationImageSizeY - 1 - global_y_grid;

    LatLon ll = ProjectPixel(global_x, global_y);
    
    if (!ll.valid) {
        return {0.0, 0.0};
    }
    return {ll.lat_deg, ll.lon_deg};
}

