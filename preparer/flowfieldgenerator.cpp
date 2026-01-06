// flowfieldgenerator.cpp

#include "flowfieldgenerator.h"
#include "parameterparser.h"
#include "fluentflowimporter.h"

#include <H5Cpp.h>
#include <spdlog/spdlog.h>
#include <cmath>
#include <vector>
#include <fmt/format.h>
#include <cstdlib>
#include <omp.h>
#include <limits>


void FlowFieldGenerator::GenerateConstantFlow(double flowBearing, double flowSpeed, bool compressFlow)
{
    spdlog::info("FlowFieldGenerator::GenerateConstantFlow: bearing={}, speed={}", flowBearing, flowSpeed);

    // Convert bearing angle to radians (0° = north = +y direction)
    double bearing_rad = flowBearing * M_PI / 180.0;
    double vx = flowSpeed * std::sin(bearing_rad);  // east component
    double vy = flowSpeed * std::cos(bearing_rad);  // north component

    spdlog::info("Constant flow: vx={}, vy={}", vx, vy);

    // Create single frame
    std::vector<double> vx_data(gx * gy, vx);
    std::vector<double> vy_data(gx * gy, vy);

    std::vector<std::vector<double>> vx_frames = {vx_data};
    std::vector<std::vector<double>> vy_frames = {vy_data};
    std::vector<std::vector<double>> eta_frames = {std::vector<double>(gx * gy, 0.0)};

    double time_interval = 0.0;  // Static flow
    int loop_mode = 1;           // Hold last frame (irrelevant for single frame)

    WriteFlowFieldToHDF5("constant", 1, time_interval, loop_mode, compressFlow,
                        vx_frames, vy_frames, eta_frames);

    spdlog::info("GenerateConstantFlow completed");
}


void FlowFieldGenerator::GenerateWaveFlow(double flowBearing, double waveAmplitude, double waveLength, double phaseSpeed,
                                         int nFrames, bool compressFlow)
{
    spdlog::info("FlowFieldGenerator::GenerateWaveFlow: bearing={}, amplitude={}, waveLength={}, phaseSpeed={}, nFrames={}",
                 flowBearing, waveAmplitude, waveLength, phaseSpeed, nFrames);

    // Wave parameters
    double k = 2.0 * M_PI / waveLength;  // wavenumber
    double omega = k * phaseSpeed;        // angular frequency
    double amplitude_omega_sq = waveAmplitude * std::pow(omega, 2.0);  // A*ω²

    // Convert bearing to wave direction
    double bearing_rad = flowBearing * M_PI / 180.0;
    double wave_dir_x = std::sin(bearing_rad);   // wave propagates east
    double wave_dir_y = std::cos(bearing_rad);   // wave propagates north

    spdlog::info("Wave parameters: k={}, omega={}, A*omega²={}", k, omega, amplitude_omega_sq);
    spdlog::info("Wave direction: ({}, {})", wave_dir_x, wave_dir_y);

    // Calculate time interval
    double T = 2.0 * M_PI / omega;
    double time_interval = T / nFrames;

    spdlog::info("Wave period: {}, time_interval: {}, num_frames: {}", T, time_interval, nFrames);

    int loop_mode = 0;  // Periodic for waves
    CreateFlowFieldHDF5("wave", nFrames, time_interval, loop_mode, compressFlow);

    // Generate frames one at a time to manage RAM
    for (int frame = 0; frame < nFrames; frame++) {
        double t = frame * time_interval;

        for (int i = 0; i < gx; i++) {
            for (int j = 0; j < gy; j++) {
                // Physical coordinates (cell centers)
                double x = (i + ox) * cellsize;
                double y = (j + oy) * cellsize;

                // Project onto wave direction: phase = k*(x*wave_dir_x + y*wave_dir_y) - omega*t
                double phase = k * (x * wave_dir_x + y * wave_dir_y) - omega * t;

                // Velocity magnitude: u' = A*ω²*sin(phase)
                double vel_mag = amplitude_omega_sq * std::sin(phase);

                // Decompose velocity into x and y components
                size_t idx = j + (size_t)i * gy;
                vx_frame[idx] = vel_mag * wave_dir_x;
                vy_frame[idx] = vel_mag * wave_dir_y;

                // Calculate eta: eta = A * sin(phase)
                // Note: velocity is A*omega^2*sin(phase), so eta = vel_mag / omega^2
                // Or simply recompute sin(phase)
                eta_frame[idx] = waveAmplitude * std::sin(phase);

                // Compute derivatives: d(eta)/dx and d(eta)/dy
                // eta(x,y,t) = A * sin(k*(x*wave_dir_x + y*wave_dir_y) - omega*t)
                // d(eta)/dx = A * k * wave_dir_x * cos(phase)
                // d(eta)/dy = A * k * wave_dir_y * cos(phase)
                double cos_phase = std::cos(phase);
                d_eta_dx_frame[idx] = waveAmplitude * k * wave_dir_x * cos_phase;
                d_eta_dy_frame[idx] = waveAmplitude * k * wave_dir_y * cos_phase;
            }
        }

        WriteFrameToHDF5(frame);
    }

    spdlog::info("GenerateWaveFlow completed: saved {} frames", nFrames);
}





void FlowFieldGenerator::GenerateFlow(const ParameterParser& params,
                                     int gx, int gy, double cellsize, int ox, int oy,
                                     int imageWidth, int imageHeight,
                                     const std::string& projectDirectory)
{
    // Initialize members
    this->gx = gx;
    this->gy = gy;
    this->cellsize = cellsize;
    this->ox = ox;
    this->oy = oy;
    this->imageWidth = imageWidth;
    this->imageHeight = imageHeight;
    this->projectDirectory = projectDirectory;

    // Resize member buffers
    vx_frame.resize(gx * gy);
    vy_frame.resize(gx * gy);
    eta_frame.resize(gx * gy);
    d_eta_dx_frame.resize(gx * gy);
    d_eta_dy_frame.resize(gx * gy);
    spdlog::info("FlowFieldGenerator::GenerateFlow: FlowType='{}'", params.FlowType);

    if (params.FlowType == "constant") {
        GenerateConstantFlow(params.FlowBearing, params.FlowSpeed, params.CompressFlow);
    }
    else if (params.FlowType == "wave") {
        GenerateWaveFlow(params.FlowBearing, params.WaveAmplitude, params.WaveLength, params.PhaseSpeed,
                        params.NFrames, params.CompressFlow);
    }
    else if (params.FlowType == "river") {
        GenerateRiverFlow(params.FlowSpeed, params.WaveAmplitude, params.CompressFlow);
    }
    else if (params.FlowType == "FLUENT-static") {
        // ... handled inside GenerateFluentFlow which currently calls WriteFlowFieldToHDF5
        GenerateFluentFlow(params);
    }
    else if (params.FlowType == "standing_wave") {
        // Periodic standing wave with linear particle motion at surface
        GenerateStandingWave(params.FlowBearing, params.WaveAmplitude, params.WaveLength, params.WavePeriod, params.NFrames, params.CompressFlow);
    }
    else if (params.FlowType.empty()) {
        spdlog::info("No flow field requested (FlowType is empty)");
    }
    else {
        throw std::runtime_error("Unknown FlowType: " + params.FlowType);
    }

    spdlog::info("FlowFieldGenerator::GenerateFlow completed");
}


void FlowFieldGenerator::GenerateRiverFlow(double flowSpeed, double waveAmplitude, bool compressFlow)
{
    spdlog::info("FlowFieldGenerator::GenerateRiverFlow: flowSpeed={}, waveAmplitude={}", 
                 flowSpeed, waveAmplitude);

    int num_frames = 1;
    double time_interval = 0.0;
    int loop_mode = 1;

    // Create the HDF5 file structure first
    CreateFlowFieldHDF5("river", num_frames, time_interval, loop_mode, compressFlow);

    double L = waveAmplitude; // transition length
    double x_center = (gx * cellsize) * 0.5;

    // Fill the frame buffers (frame 0)
    for (int i = 0; i < gx; i++) {
        double x = (i + 0.5) * cellsize;
        double eta_val = 0.0;
        double deta_dx = 0.0;
        
        if (x < x_center) {
            eta_val = 0.0;
            deta_dx = 0.0;
        } else if (x > x_center + L) {
            eta_val = -waveAmplitude;
            deta_dx = 0.0;
        } else {
            // Smooth step transition: f(s) = 3s^2 - 2s^3, f'(s) = 6s - 6s^2
            double s = (x - x_center) / L;
            double f_s = 3 * s * s - 2 * s * s * s;
            double df_ds = 6 * s - 6 * s * s;
            
            eta_val = -waveAmplitude * f_s;
            deta_dx = -waveAmplitude / L * df_ds;
        }

        for (int j = 0; j < gy; j++) {
            int idx = j + i * gy;
            vx_frame[idx] = flowSpeed;
            vy_frame[idx] = 0.0;
            eta_frame[idx] = eta_val;
            d_eta_dx_frame[idx] = deta_dx;
            d_eta_dy_frame[idx] = 0.0;
        }
    }

    // Write the single frame
    WriteFrameToHDF5(0);

    spdlog::info("GenerateRiverFlow completed");
}

void FlowFieldGenerator::GenerateFluentFlow(const ParameterParser& params)
{
    spdlog::info("FlowFieldGenerator::GenerateFluentFlow starting");
    spdlog::info("  ConfigDirectory: {}", params.ConfigFileDirectory);
    spdlog::info("  CAS file: {}", params.InputFluentCAS);
    spdlog::info("  DAT file: {}", params.InputFluentDAT);
    spdlog::info("  SVG file: {}", params.SVG);
    spdlog::info("  RectanglePathID: {}", params.RectanglePathID);
    spdlog::info("  FluentPathID: {}", params.FluentPathID);

    // Note: The FLUENT raster must match the initialization image dimensions exactly
    // so that it aligns pixel-for-pixel with the land/ice/color masks
    // The image dimensions are the ground truth from the loaded images (PrepareGridAndPoints)

    spdlog::info("InitializationImageSize: {}x{}", imageWidth, imageHeight);

    // Import FLUENT flow field - rasterize to match initialization image dimensions
    FluentFlowImporter importer;
    importer.Import(params.ConfigFileDirectory,
                   params.InputFluentCAS,
                   params.InputFluentDAT,
                   params.SVG,
                   params.RectanglePathID,
                   params.FluentPathID,
                   imageWidth, imageHeight,  // Must match the initialization image dimensions
                   params.VelocityMultiplier);  // Apply velocity multiplier

    // Get the actual rasterized dimensions and velocity data
    int width = importer.image_width;
    int height = importer.image_height;
    std::vector<double> vx_full = importer.vx_data;
    std::vector<double> vy_full = importer.vy_data;

    spdlog::info("FLUENT import completed: {}x{} grid", width, height);

    // Verify that FLUENT raster matches initialization image dimensions
    if (width != imageWidth || height != imageHeight) {
        throw std::runtime_error(fmt::format("FLUENT raster dimensions ({}x{}) do not match initialization image ({}x{})",
                                            width, height, imageWidth, imageHeight));
    }

    // Extract modeled region from full raster
    // Full raster is column-major (i + width*j) and matches image dimensions
    // Output needs to be row-major (j + i*gy) per WriteFlowFieldToHDF5 convention

    spdlog::info("Extracting region: grid={}x{}, offset=({}, {})", gx, gy, ox, oy);
    spdlog::info("  Full raster size: {}x{}", width, height);
    spdlog::info("  Modeled region size: {}x{}", gx, gy);

    std::vector<double> vx_data(gx * gy, 0.0);
    std::vector<double> vy_data(gx * gy, 0.0);

    for (int i = 0; i < gx; ++i) {
        for (int j = 0; j < gy; ++j) {
            // Source: full raster grid (image coordinates) with column-major indexing
            // The modeled region has offset (ox, oy) from the image origin
            int src_i = i + ox;
            int src_j = j + oy;

            // Verify within bounds
            if (src_i < 0 || src_i >= width || src_j < 0 || src_j >= height) {
                spdlog::warn("Region index ({}, {}) out of raster bounds ({}x{})",
                           src_i, src_j, width, height);
                vx_data[j + (size_t)i * gy] = 0.0;
                vy_data[j + (size_t)i * gy] = 0.0;
                continue;
            }

            int src_idx = src_i + width * src_j;  // Column-major indexing in raster
            int dst_idx = j + (size_t)i * gy;    // Row-major indexing in output

            vx_data[dst_idx] = vx_full[src_idx];
            vy_data[dst_idx] = vy_full[src_idx];
        }
    }

    spdlog::info("Region extraction completed");

    // Package into frame containers (single frame for static FLUENT flow)
    std::vector<std::vector<double>> vx_frames = {vx_data};
    std::vector<std::vector<double>> vy_frames = {vy_data};
    std::vector<std::vector<double>> eta_frames = {std::vector<double>(gx * gy, 0.0)};

    // Write to HDF5 with static flow parameters
    double time_interval = 0.0;  // Static flow (single frame, no interpolation)
    int loop_mode = 1;           // Hold last frame (irrelevant for single frame)

    WriteFlowFieldToHDF5(params.FlowType, 1, time_interval, loop_mode, params.CompressFlow,
                        vx_frames, vy_frames, eta_frames);

    spdlog::info("GenerateFluentFlow completed");
}


void FlowFieldGenerator::CreateFlowFieldHDF5(const std::string& flowType, int num_frames, double time_interval,
                                            int loop_mode,
                                            bool compressFlow)
{
    spdlog::info("CreateFlowFieldHDF5: flowType={}, gx={}, gy={}, num_frames={}, compress={}", flowType, gx, gy, num_frames, compressFlow);

    std::string flowFilePath = projectDirectory + "/grid_flow.h5";
    H5::H5File file(flowFilePath, H5F_ACC_TRUNC);

    // Create datasets for vx and vy (3D: [num_frames, gx, gy])
    // Data is stored in column-major order (row-minor): gx is the major axis
    hsize_t dims[3] = {static_cast<hsize_t>(num_frames), static_cast<hsize_t>(gx), static_cast<hsize_t>(gy)};
    H5::DataSpace dataspace(3, dims);

    H5::DSetCreatPropList props;
    if (compressFlow) {
        hsize_t chunks[3] = {1, std::min<hsize_t>(gx, 64), std::min<hsize_t>(gy, 64)};
        props.setChunk(3, chunks);
        props.setDeflate(6);
    }

    H5::DataSet ds_vx = file.createDataSet("water_current_vx", H5::PredType::NATIVE_DOUBLE, dataspace, props);
    H5::DataSet ds_vy = file.createDataSet("water_current_vy", H5::PredType::NATIVE_DOUBLE, dataspace, props);
    H5::DataSet ds_eta = file.createDataSet("water_current_eta", H5::PredType::NATIVE_DOUBLE, dataspace, props);
    H5::DataSet ds_d_eta_dx = file.createDataSet("water_current_d_eta_dx", H5::PredType::NATIVE_DOUBLE, dataspace, props);
    H5::DataSet ds_d_eta_dy = file.createDataSet("water_current_d_eta_dy", H5::PredType::NATIVE_DOUBLE, dataspace, props);

    H5::DataSpace att_space(H5S_SCALAR);

    // Set flow_type attribute on the root group
    H5::Group root = file.openGroup("/");
    H5::StrType str_type(H5::PredType::C_S1, flowType.size() + 1);
    root.createAttribute("flow_type", str_type, att_space).write(str_type, flowType.c_str());

    ds_vx.createAttribute("time_interval", H5::PredType::NATIVE_DOUBLE, att_space)
        .write(H5::PredType::NATIVE_DOUBLE, &time_interval);
    ds_vx.createAttribute("loop_mode", H5::PredType::NATIVE_INT, att_space)
        .write(H5::PredType::NATIVE_INT, &loop_mode);
    ds_vx.createAttribute("num_frames", H5::PredType::NATIVE_INT, att_space)
        .write(H5::PredType::NATIVE_INT, &num_frames);

    file.close();

    spdlog::info("CreateFlowFieldHDF5 completed: file created at {}", flowFilePath);
}

void FlowFieldGenerator::WriteFrameToHDF5(int frame_index)
{
    std::string flowFilePath = projectDirectory + "/grid_flow.h5";
    H5::H5File file(flowFilePath, H5F_ACC_RDWR);

    // Open existing datasets
    H5::DataSet ds_vx = file.openDataSet("water_current_vx");
    H5::DataSet ds_vy = file.openDataSet("water_current_vy");
    H5::DataSet ds_eta = file.openDataSet("water_current_eta");
    H5::DataSet ds_d_eta_dx = file.openDataSet("water_current_d_eta_dx");
    H5::DataSet ds_d_eta_dy = file.openDataSet("water_current_d_eta_dy");

    // Get dataspace and select hyperslab for this frame
    H5::DataSpace dataspace = ds_vx.getSpace();
    hsize_t frame_offset[3] = {static_cast<hsize_t>(frame_index), 0, 0};
    hsize_t frame_dims[3] = {1, static_cast<hsize_t>(gx), static_cast<hsize_t>(gy)};
    dataspace.selectHyperslab(H5S_SELECT_SET, frame_dims, frame_offset);

    // Create memory dataspace for single frame
    H5::DataSpace frame_space(3, frame_dims);

    // Write vx, vy, eta, and derivative data for this frame
    ds_vx.write(vx_frame.data(), H5::PredType::NATIVE_DOUBLE, frame_space, dataspace);
    ds_vy.write(vy_frame.data(), H5::PredType::NATIVE_DOUBLE, frame_space, dataspace);
    ds_eta.write(eta_frame.data(), H5::PredType::NATIVE_DOUBLE, frame_space, dataspace);
    ds_d_eta_dx.write(d_eta_dx_frame.data(), H5::PredType::NATIVE_DOUBLE, frame_space, dataspace);
    ds_d_eta_dy.write(d_eta_dy_frame.data(), H5::PredType::NATIVE_DOUBLE, frame_space, dataspace);

    file.close();
}

void FlowFieldGenerator::WriteFlowFieldToHDF5(const std::string& flowType, int num_frames, double time_interval,
                                             int loop_mode,
                                             bool compressFlow,
                                             const std::vector<std::vector<double>>& vx_frames,
                                             const std::vector<std::vector<double>>& vy_frames,
                                             const std::vector<std::vector<double>>& eta_frames)
{
    // Legacy wrapper: Create file structure then write all frames
    CreateFlowFieldHDF5(flowType, num_frames, time_interval, loop_mode, compressFlow);

    for (int frame = 0; frame < num_frames; frame++) {
        // Copy data to member buffers
        vx_frame = vx_frames[frame];
        vy_frame = vy_frames[frame];
        eta_frame = eta_frames[frame];
        WriteFrameToHDF5(frame);
    }

    spdlog::info("WriteFlowFieldToHDF5 completed: saved {} frames to {}/grid_flow.h5", num_frames, projectDirectory);
}





void FlowFieldGenerator::GenerateStandingWave(double flowBearing, double waveAmplitude, double waveLength, double wavePeriod, int nFrames, bool compressFlow)
{
    spdlog::info("FlowFieldGenerator::GenerateStandingWave: bearing={}, amplitude={}, wavelength={}, period={}, nFrames={}",
                 flowBearing, waveAmplitude, waveLength, wavePeriod, nFrames);

    // Standing wave parameters with amplitude ramp over first 5 periods
    // Water elevation: eta(x,t) = A(t) * sin(k·r) * sin(omega*t)
    // where A(t) = A_max * min(1.0, t / (5*T)), T = wave period
    // r is the projection onto the wave direction
    // Horizontal velocity: u(x,t) = (A(t)*omega) * sin(k·r) * cos(omega*t)

    const double k = 2.0 * M_PI / waveLength;  // wavenumber
    const double omega = 2.0 * M_PI / wavePeriod;  // angular frequency

    // Convert bearing to wave direction (same as GenerateWaveFlow)
    double bearing_rad = flowBearing * M_PI / 180.0;
    double wave_dir_x = std::sin(bearing_rad);   // wave propagates east
    double wave_dir_y = std::cos(bearing_rad);   // wave propagates north

    // Ramp period: amplitude ramps up during first 5 periods
    const double ramp_time = 5.0 * wavePeriod;

    // Frame spacing
    double time_interval = wavePeriod / nFrames;

    spdlog::info("Standing wave parameters: k={}, omega={}, ramp_time={}, time_interval={}", k, omega, ramp_time, time_interval);
    spdlog::info("Wave direction: ({}, {})", wave_dir_x, wave_dir_y);

    // Periodic: loop_mode = 0
    CreateFlowFieldHDF5("standing_wave", nFrames, time_interval, 0, compressFlow);

    for (int frame = 0; frame < nFrames; frame++) {
        double t = frame * time_interval;

        // Precompute sin(omega*t) and cos(omega*t) for efficiency
        double sin_wt = std::sin(omega * t);
        double cos_wt = std::cos(omega * t);

        for (int i = 0; i < gx; i++) {
            for (int j = 0; j < gy; j++) {
                // Physical coordinates
                double x = (i + ox) * cellsize;
                double y = (j + oy) * cellsize;

                // Project onto wave direction: r = x*wave_dir_x + y*wave_dir_y
                double r = x * wave_dir_x + y * wave_dir_y;

                // Precompute sin(k*r)
                double sin_kr = std::sin(k * r);

                // Water elevation: eta(x,y,t) = A * sin(k*r) * sin(omega*t)
                double eta = waveAmplitude * sin_kr * sin_wt;

                // Velocity magnitude along wave direction: u(x,y,t) = A * omega * sin(k*r) * cos(omega*t)
                double u_mag = waveAmplitude * omega * sin_kr * cos_wt;

                // Decompose velocity into x and y components
                size_t idx = j + (size_t)i * gy;
                vx_frame[idx] = u_mag * wave_dir_x;   // x-component of particle velocity
                vy_frame[idx] = u_mag * wave_dir_y;   // y-component of particle velocity
                eta_frame[idx] = eta;

                // Compute derivatives: d(eta)/dx and d(eta)/dy
                // eta(x,y,t) = A * sin(k*r) * sin(omega*t), where r = x*wave_dir_x + y*wave_dir_y
                // d(eta)/dx = A * k * wave_dir_x * cos(k*r) * sin(omega*t)
                // d(eta)/dy = A * k * wave_dir_y * cos(k*r) * sin(omega*t)
                double cos_kr = std::cos(k * r);
                double spatial_factor = waveAmplitude * k * cos_kr * sin_wt;
                d_eta_dx_frame[idx] = spatial_factor * wave_dir_x;
                d_eta_dy_frame[idx] = spatial_factor * wave_dir_y;
            }
        }

        WriteFrameToHDF5(frame);
    }

    spdlog::info("GenerateStandingWave completed: created {} frames with period={}, bearing={}, eta(r,t) = A * sin(2π*r/{}) * sin(2π*t/{}) where r is projection onto wave direction, A(t) ramps from 0 to {} over 5 periods",
                 nFrames, wavePeriod, flowBearing, waveLength, wavePeriod, waveAmplitude);
}





