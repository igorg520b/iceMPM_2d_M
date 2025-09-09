// framedata.cpp

#include "framedata.h"
#include <fmt/format.h>
#include <fmt/std.h>

#include <vtkVersionMacros.h>
#include <QDebug>

#include "vtk_representation.h"

namespace fs = std::filesystem;


FrameData::FrameData(GeneralGridData &ggd_) : ggd(ggd_) {}


bool FrameData::LoadFrame(int frameNumber)
{
    if (frameNumber < 0 || frameNumber >= ggd.countFrames) {
        LOGR("Error: Requested frame number {} is out of range (0-{}).", frameNumber, ggd.countFrames - 1);
        return false;
    }

    const std::string baseName = fmt::format(fmt::runtime("f{:05d}.h5"), frameNumber);
    const fs::path fullPath = fs::path(ggd.frameDirectory) / baseName;

    if (!fs::exists(fullPath)) {
        LOGR("Error: Frame file does not exist: {}", fullPath.string());
        return false;
    }

    LOGR("Loading frame {} from {}", frameNumber, fullPath.string());

    try {
        const H5::H5File file(fullPath.string(), H5F_ACC_RDONLY);

        // 1. Read Frame Attributes from the "rgb" dataset.
        const H5::DataSet attr_dset = file.openDataSet("rgb");
        attr_dset.openAttribute("SimulationStep").read(H5::PredType::NATIVE_INT, &this->simulationStep);
        attr_dset.openAttribute("SimulationTime").read(H5::PredType::NATIVE_DOUBLE, &this->simulationTime);
        this->currentFrameNumber = frameNumber;

        // 2. Prepare Memory Buffers
        const auto gx = ggd.prms.GridXTotal;
        const auto gy = ggd.prms.GridYTotal;
        const size_t gridSize = (size_t)gx * gy;
        this->rgb.resize(gridSize * 3);
        this->host_grid_buffer.assign(gridSize * SimParams::nGridArrays, 0.0);

        // 3. Load the pre-rendered RGB data into this object's `rgb` buffer.
        try {
            // Re-use the dataset handle from attribute reading.
            attr_dset.read(this->rgb.data(), H5::PredType::NATIVE_UINT8);
        } catch (const H5::Exception& e) {
            LOGR("HDF5 Error reading 'rgb' dataset: {}", e.getCDetailMsg());
            // This is a critical error for visualization, so we might want to fail.
            return false;
        }

        // 4. "Undo" the preparation: Populate the r, g, b slices of the host_grid_buffer.
        // This de-interleaves, normalizes, and transposes the RGB data into the
        // planar format expected by the VisualRepresentation class.
        const size_t r_offset = gridSize * SimParams::grid_idx_vis_r;
        const size_t g_offset = gridSize * SimParams::grid_idx_vis_g;
        const size_t b_offset = gridSize * SimParams::grid_idx_vis_b;

        for (int i = 0; i < gx; ++i) { // x-coordinate
            for (int j = 0; j < gy; ++j) { // y-coordinate
                // The HDF5 data has dims (gx, gy, 3), which implies row-major (x varies fastest)
                const size_t src_idx = ((size_t)j * gx + i) * 3;
                // Our internal grid buffers are column-major (y varies fastest)
                const size_t dst_idx = (size_t)j + (size_t)i * gy;

                this->host_grid_buffer[r_offset + dst_idx] = static_cast<t_GridReal>(this->rgb[src_idx + 0]) / 255.0;
                this->host_grid_buffer[g_offset + dst_idx] = static_cast<t_GridReal>(this->rgb[src_idx + 1]) / 255.0;
                this->host_grid_buffer[b_offset + dst_idx] = static_cast<t_GridReal>(this->rgb[src_idx + 2]) / 255.0;
            }
        }

        // 5. Load all other raw scalar grid datasets.
        readGridDataset(file, "grid_idx_px", this->host_grid_buffer, gridSize * SimParams::grid_idx_px, gridSize);
        readGridDataset(file, "grid_idx_py", this->host_grid_buffer, gridSize * SimParams::grid_idx_py, gridSize);
        readGridDataset(file, "grid_idx_mass", this->host_grid_buffer, gridSize * SimParams::grid_idx_mass, gridSize);
        readGridDataset(file, "grid_idx_vis_pts_density", this->host_grid_buffer, gridSize * SimParams::grid_idx_vis_pts_density, gridSize);
        readGridDataset(file, "grid_idx_vis_Jpinv", this->host_grid_buffer, gridSize * SimParams::grid_idx_vis_Jpinv, gridSize);
        readGridDataset(file, "grid_idx_vis_P", this->host_grid_buffer, gridSize * SimParams::grid_idx_vis_P, gridSize);
        readGridDataset(file, "grid_idx_vis_Q", this->host_grid_buffer, gridSize * SimParams::grid_idx_vis_Q, gridSize);
        readGridDataset(file, "grid_idx_vis_strain_vonMises", this->host_grid_buffer, gridSize * SimParams::grid_idx_vis_strain_vonMises, gridSize);
        // Note: I'm intentionally omitting 'strain_EqvGreenLagrange' as it was commented out in your SaveFrame function.
        // Adding it back is as simple as uncommenting the next line.
        // readGridDataset(file, "grid_idx_vis_strain_EqvGreenLagrange", this->host_grid_buffer, gridSize * SimParams::grid_idx_vis_strain_EqvGreenLagrange, gridSize);

    } catch (const H5::Exception& e) {
        LOGR("Critical HDF5 Error during file open for frame {}: {}", frameNumber, e.getCDetailMsg());
        return false;
    }

    LOGR("Successfully loaded frame {}.", frameNumber);
    return true;
}




void FrameData::readGridDataset(const H5::H5File& file, const std::string& dataset_name,
                     std::vector<t_GridReal>& dest_buffer, size_t offset, size_t plane_size)
{
    try {
        if (!H5Lexists(file.getId(), dataset_name.c_str(), H5P_DEFAULT)) {
            LOGR("Dataset '{}' not found in HDF5 file, skipping.", dataset_name);
            return;
        }

        H5::DataSet dataset = file.openDataSet(dataset_name);
        H5::DataType file_dtype = H5::PredType::NATIVE_FLOAT;
        H5::DataType mem_dtype = H5::PredType::NATIVE_DOUBLE;

        hsize_t plane_dims[1] = {plane_size};
        H5::DataSpace memspace(1, plane_dims);
        dataset.read(dest_buffer.data() + offset, file_dtype, memspace, dataset.getSpace());

    } catch (const H5::Exception& e) {
        LOGR("HDF5 Error reading dataset '{}': {}", dataset_name, e.getCDetailMsg());
    }
}


void FrameData::LinkRepresentation(icy::VisualRepresentation &representation)
{
    // This function provides a clean, single point of connection between the data
    // and the view. It configures the representation object to use the data
    // loaded and owned by this FrameData instance.

    // 1. Link to data owned by this FrameData instance.
    //    - host_grid_buffer is populated by LoadFrame().
    representation.host_grid_buffer = this->host_grid_buffer.data();

    // 2. Link to static, simulation-wide data that is shared via the ggd reference.
    //    - prms and original_image_colors_rgb are constant for all frames.
    representation.prms = &this->ggd.prms;
    representation.original_image_colors_rgb = &this->ggd.original_image_colors_rgb;
    representation.grid_status_buffer = ggd.grid_status_buffer.data();

    // 3. Explicitly nullify pointers to data sources that are not available in frame files.
    //    This is a critical safety step to prevent the VisualRepresentation from
    //    attempting to render stale or invalid data from a previous context.
    //    The renderer's internal logic must check for these null pointers.
    representation.hssoa = nullptr;                 // Point data is not saved in frame files.
    representation.wac_interpolator = nullptr;      // Wind/current data is not saved in frame files.
    representation.point_partitions = nullptr;      // Point partition data is not saved in frame files.


//    if (!prms || !host_grid_buffer || !grid_status_buffer || !original_image_colors_rgb) {
//        LOGV("VisualRepresentation::SynchronizeTopology - Aborting: Essential data pointers are not set.");
//        return;
 //   }
}
