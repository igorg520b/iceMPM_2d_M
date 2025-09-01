#include "windandcurrentinterpolator.h"
#include <spdlog/spdlog.h>
#include <iostream>


WindAndCurrentInterpolator::WindAndCurrentInterpolator(SimParams& params) : prms(params)
{}


void WindAndCurrentInterpolator::OpenCustomHDF5(std::string fileName)
{
    file = H5::H5File(fileName, H5F_ACC_RDONLY);
    ReadStaticFlowData();
    LOGV("WindAndCurrentInterpolator::OpenCustomHDF5 done");
}


void WindAndCurrentInterpolator::ReadStaticFlowData()
{
    LOGV("WindAndCurrentInterpolator::ReadStaticFlowData");
    const int &width = prms.InitializationImageSizeX;
    const int &height = prms.InitializationImageSizeY;
    const int &ox = prms.ModeledRegionOffsetX;
    const int &oy = prms.ModeledRegionOffsetY;
    const int &gx = prms.GridXTotal;
    const int &gy = prms.GridYTotal;

    H5::DataSet ds = file.openDataSet("velocities");

    int rank = ds.getSpace().getSimpleExtentNdims(); // Get number of dimensions
    if (rank != 3) {
        throw std::runtime_error("Incorrect dataset rank");
    }

    hsize_t file_dims[3];
    ds.getSpace().getSimpleExtentDims(file_dims, NULL); // Read dimensions into file_dims

    // Map HDF5 dims to conceptual width/height based on SaveAsHDF5
    const hsize_t hdf5_height = file_dims[0];
    const hsize_t hdf5_width  = file_dims[1];
    const hsize_t hdf5_comps  = file_dims[2];

    // 3. Verify dimensions
    if (hdf5_height != height || hdf5_width  != width  || hdf5_comps  != 2)
    {
        throw std::runtime_error("ReadStaticFlowData: Dimension mismatch for dataset '");
    }

    std::vector<Eigen::Vector2f> velocity_field(width*height); // Temporary vector
    ds.read(velocity_field.data(), H5::PredType::NATIVE_FLOAT);

    // transfer to current_flow_data
    current_flow_data.resize(2*gx*gy);

    for(int i=0;i<gx;i++)
        for(int j=0;j<gy;j++)
        {
            size_t idx = (i+ox) + (j+oy)*width;
            size_t idx2 = j + i*gy;
            current_flow_data[idx2] = (t_GridReal)velocity_field[idx].x();
            current_flow_data[gx*gy + idx2] = (t_GridReal)velocity_field[idx].y();
        }
}


bool WindAndCurrentInterpolator::SetTime(double t)
{
    // todo: unfinished
    return false;
}
