#include "flowdataprocessor.h"

#include <vtkUnstructuredGrid.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkImageData.h>
#include <vtkProbeFilter.h>
#include <vtkSmartPointer.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkProperty.h>
#include <vtkCellDataToPointData.h>
#include <vtkCellData.h>

#include <H5Cpp.h>
#include <iostream>
#include <filesystem>

#include <spdlog/spdlog.h>


FlowDataProcessor::FlowDataProcessor()
{
    actor->GetProperty()->SetEdgeVisibility(true);
    actor->GetProperty()->SetEdgeColor(0.2,0.2,0.2);
    actor->GetProperty()->LightingOff();
    actor->GetProperty()->ShadingOff();
    actor->GetProperty()->SetInterpolationToFlat();
    actor->PickableOff();
    actor->GetProperty()->SetColor(0.1,0.1,0.1);
//    actor->GetProperty()->SetRepresentationToWireframe();


    // set up custom lut

}

void FlowDataProcessor::LoadFluentResult(const std::string fileNameDAT, const std::string fileNameCAS, int width, int height)
{
    spdlog::info("FlowDataProcessor::LoadFluentResult");
    this->width = width;
    this->height = height;

    reader->SetDataFileName(fileNameDAT.c_str());
    reader->SetFileName(fileNameCAS.c_str());  // Set the mesh file (.cas.h5)
    reader->Update(); // Update the reader
}

void FlowDataProcessor::PopulateWithConstValue(int width, int height, double vx, double vy)
{
    this->width = width;
    this->height = height;
    velocity_field.assign(width*height, Eigen::Vector2f(vx, vy));
}



void FlowDataProcessor::ApplyTransform(const Eigen::Vector2f extents_fluent[2])
{
    std::cout << "FlowDataProcessor::ApplyTransform (auto-scaling to extents)" << std::endl;

    vtkUnstructuredGrid* grid = vtkUnstructuredGrid::SafeDownCast(reader->GetOutput()->GetBlock(0));
    if (!grid) {
        std::cerr << "ERROR: Original grid is null. Cannot get data from reader output." << std::endl;
        throw std::runtime_error("Original grid is null. Cannot get data from reader output.");
    }

    // 1. Get the bounds of the original grid
    double original_bounds[6];
    grid->GetBounds(original_bounds);

    Eigen::Vector2f source_min(original_bounds[0], original_bounds[2]);
    Eigen::Vector2f source_max(original_bounds[1], original_bounds[3]);
    Eigen::Vector2f source_dims = source_max - source_min;

    if (source_dims.x() <= 0 || source_dims.y() <= 0) {
        std::cerr << "ERROR: Original grid has zero or negative dimensions." << std::endl;
        throw std::runtime_error("Invalid source grid dimensions.");
    }

    std::cout << "--- Original Grid Bounds ---\n"
              << "  X range: " << source_min.x() << " to " << source_max.x() << "\n"
              << "  Y range: " << source_min.y() << " to " << source_max.y() << std::endl;

    // 2. Define the target bounds from the input parameter
    const Eigen::Vector2f& target_min = extents_fluent[0];
    const Eigen::Vector2f& target_max = extents_fluent[1];
    Eigen::Vector2f target_dims = target_max - target_min;

    if (target_dims.x() <= 0 || target_dims.y() <= 0) {
        std::cerr << "ERROR: Target extents have zero or negative dimensions." << std::endl;
        throw std::runtime_error("Invalid target extents dimensions.");
    }

    std::cout << "--- Target Bounds (from extents_fluent) ---\n"
              << "  X range: " << target_min.x() << " to " << target_max.x() << "\n"
              << "  Y range: " << target_min.y() << " to " << target_max.y() << std::endl;

    // 3. Calculate the uniform scale and check aspect ratio
    double scale = target_dims.x() / source_dims.x();

    float source_aspect = source_dims.x() / source_dims.y();
    float target_aspect = target_dims.x() / target_dims.y();
    if (std::abs(source_aspect - target_aspect) > 1e-3) {
        std::cerr << "WARN: Source grid aspect ratio (" << source_aspect
                  << ") differs from target extents aspect ratio (" << target_aspect
                  << "). This may cause distortion." << std::endl;
    }

    std::cout << "INFO: Calculated uniform scale: " << scale << std::endl;

    // 4. Set up the transformation
    // The correct order to map one box to another is:
    // 1. Translate the source to the origin.
    // 2. Scale it.
    // 3. Translate it to the target's final position.
    // VTK applies these in reverse order of declaration.
    transform->Identity();
    transform->Translate(target_min.x(), target_min.y(), 0.0);
    transform->Scale(scale, scale, 1.0); // Use 1.0 for Z scale in 2D
    transform->Translate(-source_min.x(), -source_min.y(), 0.0);
    transform->Update();

    // 5. Apply the transform
    transformFilter->SetTransform(transform);
    transformFilter->SetInputData(grid);
    transformFilter->Update();

    // 6. Verify the results
    vtkUnstructuredGrid* grid2 = transformFilter->GetUnstructuredGridOutput();
    if (!grid2 || !grid2->GetCellData()->GetArray("SV_V")) {
        std::cerr << "ERROR: Transformed grid or 'SV_V' array not found" << std::endl;
        throw std::runtime_error("Transformed grid or 'SV_V' array not found");
    }

    double transformed_bounds[6];
    grid2->GetBounds(transformed_bounds);
    std::cout << "--- Transformed Grid (grid2) Bounds ---\n"
              << "  X range: " << transformed_bounds[0] << " to " << transformed_bounds[1] << "\n"
              << "  Y range: " << transformed_bounds[2] << " to " << transformed_bounds[3] << std::endl;

    // 7. Continue with the rest of the original function's logic
    grid2->GetCellData()->SetActiveScalars("SV_V");
    mapper->SetInputData(grid2);

    mapper->SetScalarModeToUseCellData();
    mapper->UseLookupTableScalarRangeOn();
    mapper->SetColorModeToMapScalars();
    mapper->ScalarVisibilityOn();
    lut->SetTableRange(-2.2, 2.2);
    lut->Build();

    mapper->SetLookupTable(lut);

    mapper->Modified();
    actor->SetMapper(mapper);
}


void FlowDataProcessor::Rasterize()
{
    spdlog::info("FlowDataProcessor::Rasterize()");

    vtkNew<vtkCellDataToPointData> filter_cd2pd;
    vtkNew<vtkImageData> imageData;
    vtkNew<vtkProbeFilter> probeFilter;

    filter_cd2pd->SetInputData(transformFilter->GetUnstructuredGridOutput());
    filter_cd2pd->Update();


    vtkUnstructuredGrid* ug = filter_cd2pd->GetUnstructuredGridOutput();
    if (!ug->GetPointData()->HasArray("SV_V") || !ug->GetPointData()->HasArray("SV_U")) {
        spdlog::error("SV_V or SV_U missing in source data");
        throw std::runtime_error("Source data missing arrays");
    }

    // Set imageData bounds to match unstructured grid
    double bounds[6];
    ug->GetBounds(bounds);
    std::cout << "UG bounds: [" << bounds[0] << ", " << bounds[1] << "] x ["
              << bounds[2] << ", " << bounds[3] << "] x ["
              << bounds[4] << ", " << bounds[5] << "]" << std::endl;

    spdlog::info("imageData dimensions {} x {}", width, height);
    // set the size of vtkImageData
    imageData->SetDimensions(width, height, 1);
    imageData->SetSpacing(1.0, 1.0, 1.0); // Adjust spacing as needed

    // transfer to imageData via probeFilter
    probeFilter->SetInputData(imageData); // Set the 2D structured grid as input
    probeFilter->SetSourceData(filter_cd2pd->GetOutput()); // Set the unstructured grid as source
    probeFilter->PassPointArraysOn();
    probeFilter->Update();

    vtkImageData* probedData = vtkImageData::SafeDownCast(probeFilter->GetOutput());
    if (!probedData) {
        spdlog::error("Probe filter output invalid");
        throw std::runtime_error("Probe filter failed");
    }

    vtkDoubleArray* dataArray = vtkDoubleArray::SafeDownCast(probedData->GetPointData()->GetScalars("SV_V"));

    if (!dataArray) {
        spdlog::error("Failed to extract SV_V from probedData");
        throw std::runtime_error("Failed to extract SV_V from probedData");
    }

    velocity_field.resize(width*height, Eigen::Vector2f::Zero());

    int count = 0;
    for(int i=0;i<width;i++)
        for(int j=0;j<height;j++)
        {
            double val = dataArray->GetValue(i + width*j);
            velocity_field[i + width*j].y() = val;
            if(val == 0) count++;

        }

    if(count == width*height) throw std::runtime_error("all zeros");

    dataArray = vtkDoubleArray::SafeDownCast(probedData->GetPointData()->GetScalars("SV_U"));
    for(int i=0;i<width;i++)
        for(int j=0;j<height;j++)
        {
            double val = dataArray->GetValue(i + width*j);
            velocity_field[i + width*j].x() = val;
        }
}


void FlowDataProcessor::SaveAsHDF5(std::string ProjectDirectory)
{
    std::vector<float> norms(width*height,0);
    for(int k=0;k<width*height;k++) norms[k] = velocity_field[k].norm();

    H5::H5File file(ProjectDirectory + std::string("/flow_velocities.h5"), H5F_ACC_TRUNC);

    // Define data spaces (2D: width x height now)
    hsize_t dims[2] = {static_cast<hsize_t>(height), static_cast<hsize_t>(width)};
    H5::DataSpace dataspace(2, dims);

    // Set chunking and compression properties
    H5::DSetCreatPropList props;
    hsize_t chunk_dims[2] = {std::min<hsize_t>(height, 64), std::min<hsize_t>(width, 64)}; // Swapped: width x height
    props.setChunk(2, chunk_dims);
    props.setDeflate(6);

    // Save norms
    H5::DataSet dataset_indices = file.createDataSet("norms", H5::PredType::NATIVE_FLOAT,
                                                     dataspace, props);
    dataset_indices.write(norms.data(), H5::PredType::NATIVE_FLOAT);


    // Save velocity field (3D: width x height x 2)
    hsize_t dims_normals[3] = {static_cast<hsize_t>(height), static_cast<hsize_t>(width), 2};
    H5::DataSpace dataspace_normals(3, dims_normals);
    hsize_t chunk_dims_normals[3] = {std::min<hsize_t>(height, 64), std::min<hsize_t>(width, 64), 2};
    props.setChunk(3, chunk_dims_normals);
    H5::DataSet dataset_normals = file.createDataSet("velocities", H5::PredType::NATIVE_FLOAT,
                                                     dataspace_normals, props);
    dataset_normals.write(velocity_field.data(), H5::PredType::NATIVE_FLOAT);
}


void FlowDataProcessor::ApplyDiffusion(std::vector<int> &path_indices, int iterations, float alpha)
{
    // --- Basic Checks ---
    if (width <= 0 || height <= 0) {
        return; // Cannot process empty or invalid grid
    }
    size_t expected_size = static_cast<size_t>(width) * height;
    if (velocity_field.size() != expected_size) {
        throw std::runtime_error("Velocity field size does not match width*height.");
    }
    if (path_indices.size() != expected_size) {
        throw std::runtime_error("Path indices mask size does not match width*height.");
    }
    if (iterations <= 0 || alpha == 0.0f) {
        return; // No work to do
    }

    // --- Temporary Storage ---
    std::vector<Eigen::Vector2f> temp_velocity_field(expected_size);

    // --- Initial Masking ---
    // Ensure boundaries are zero before starting
    for (size_t idx = 0; idx < expected_size; ++idx) {
        if (path_indices[idx] != 1000) {
            velocity_field[idx].setZero();
        }
    }

    // --- Diffusion Loop ---
    for (int step = 0; step < iterations; ++step) {
        const std::vector<Eigen::Vector2f>& current_field = (step % 2 == 0) ? velocity_field : temp_velocity_field;
        std::vector<Eigen::Vector2f>& next_field = (step % 2 == 0) ? temp_velocity_field : velocity_field;

        // --- Grid Traversal ---
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Current point index
                int current_idx = x + y * width;

                // --- Check if current cell is ACTIVE ---
                if (path_indices[current_idx] != 1000) {
                    // If not active, it's a boundary cell, velocity must remain zero.
                    next_field[current_idx].setZero();
                    continue; // Move to the next pixel
                }

                // --- Current cell IS ACTIVE: Calculate Diffusion ---

                // --- Get Neighbor Indices (simple bounds check) ---
                // Note: The *value* used for neighbors depends on whether they are active
                int x_prev_idx = (x > 0)       ? (x - 1) + y * width : current_idx;
                int x_next_idx = (x < width-1) ? (x + 1) + y * width : current_idx;
                int y_prev_idx = (y > 0)       ? x + (y - 1) * width : current_idx; // "up"
                int y_next_idx = (y < height-1)? x + (y + 1) * width : current_idx; // "down"

                // --- Get Neighbor Values (respecting boundary mask) ---
                // If a neighbor is not active (path_indices != 1000), treat its velocity as zero (Dirichlet boundary)
                const Eigen::Vector2f v_left = (path_indices[x_prev_idx] == 1000) ? current_field[x_prev_idx] : Eigen::Vector2f::Zero();
                const Eigen::Vector2f v_right= (path_indices[x_next_idx] == 1000) ? current_field[x_next_idx] : Eigen::Vector2f::Zero();
                const Eigen::Vector2f v_up   = (path_indices[y_prev_idx] == 1000) ? current_field[y_prev_idx] : Eigen::Vector2f::Zero();
                const Eigen::Vector2f v_down = (path_indices[y_next_idx] == 1000) ? current_field[y_next_idx] : Eigen::Vector2f::Zero();

                // --- Calculate Laplacian using potentially zeroed neighbors ---
                Eigen::Vector2f laplacian_v = v_left + v_right + v_up + v_down - 4.0f * current_field[current_idx];

                // --- Update Rule (Forward Euler) for ACTIVE cell ---
                next_field[current_idx] = current_field[current_idx] + alpha * laplacian_v;

            } // end for x
        } // end for y
        // --- End of Grid Traversal for one step ---

        // Buffers are swapped implicitly for the next iteration.
        // Boundary cells in `next_field` were explicitly set to zero if `path_indices != 1000`.

    } // --- End of Diffusion Loop ---

    // --- Final Swap (if needed) ---
    if (iterations % 2 != 0) {
        velocity_field.swap(temp_velocity_field);
    }

    // --- Final Masking (Safety check) ---
    // Ensure all boundary cells are definitively zero after all iterations.
    // This might be redundant if the logic inside the loop is perfect, but good practice.
    for (size_t idx = 0; idx < expected_size; ++idx) {
        if (path_indices[idx] != 1000) {
            velocity_field[idx].setZero();
        }
    }
}
