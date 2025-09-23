#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <regex>
#include <chrono>
#include <algorithm>
#include <H5Cpp.h>
#include <omp.h>

namespace fs = std::filesystem;

const std::vector<std::string> GRID_DATASET_NAMES = {
    "grid_idx_px", "grid_idx_py", "grid_idx_mass", "grid_idx_vis_pts_density",
    "grid_idx_vis_Jpinv", "grid_idx_vis_P", "grid_idx_vis_Q",
    "grid_idx_vis_strain_vonMises"
};

// --- Function Prototypes ---
std::vector<fs::path> scanForFrameFiles(const fs::path& directory);
void processFrameFile(const fs::path& filePath);


// --- Main Entry Point (No Changes Needed) ---
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <directory_with_frames>" << std::endl;
        return 1;
    }
    const fs::path directory(argv[1]);
    if (!fs::is_directory(directory)) {
        std::cerr << "Error: Provided path is not a directory: " << directory << std::endl;
        return 1;
    }

    std::cout << "Scanning directory: " << directory.string() << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<fs::path> files_to_process = scanForFrameFiles(directory);

    if (files_to_process.empty()) {
        std::cout << "No frame files matching the pattern 'f*.h5' were found." << std::endl;
        return 0;
    }
    std::cout << "Found " << files_to_process.size() << " frame files. Starting sequential compression..." << std::endl;

    const fs::path newDir = directory.parent_path() / "frames_compressed";
    fs::create_directories(newDir);
    std::cout << "Output will be saved in: " << newDir.string() << std::endl;

    // The loop is sequential as requested in the previous turn.
    for (size_t i = 0; i < files_to_process.size(); ++i) {
        processFrameFile(files_to_process[i]);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "\n--------------------------------------------------" << std::endl;
    std::cout << "Compression complete." << std::endl;
    std::cout << "Processed " << files_to_process.size() << " files in " << duration.count() << " seconds." << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    return 0;
}


// --- scanForFrameFiles (No Changes Needed) ---
std::vector<fs::path> scanForFrameFiles(const fs::path& directory) {
    std::vector<fs::path> frameFiles;
    const std::regex filePattern(R"(^f\d+\.h5$)");

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && std::regex_match(entry.path().filename().string(), filePattern)) {
            frameFiles.push_back(entry.path());
        }
    }
    std::sort(frameFiles.begin(), frameFiles.end());
    return frameFiles;
}


// --- processFrameFile (Modified with the skip logic) ---
void processFrameFile(const fs::path& filePath) {
    const fs::path newDir = filePath.parent_path().parent_path() / "frames_compressed";
    const fs::path newFilePath = newDir / filePath.filename();

    // --- KEY ADDITION: Check if the destination file already exists ---
    if (fs::exists(newFilePath)) {
        // Use a std::cout block for thread-safe printing if you re-enable OpenMP
        // For sequential code, it's fine as is.
        std::cout << "  Skipping (already exists): " << filePath.filename().string() << "\n";
        return; // Exit the function immediately.
    }

    H5::H5File originalFile(filePath.string(), H5F_ACC_RDONLY);
    H5::H5File newFile(newFilePath.string(), H5F_ACC_TRUNC);

    // --- Process the RGB Dataset ---
    {
        H5::DataSet originalRgbDset = originalFile.openDataSet("rgb");
        H5::DataSpace filespace = originalRgbDset.getSpace();

        std::vector<uint8_t> rgb_buffer(filespace.getSelectNpoints());
        originalRgbDset.read(rgb_buffer.data(), H5::PredType::NATIVE_UINT8);

        H5::DSetCreatPropList rgbProps;
        hsize_t chunk_dims[3] = {128, 128, 3};
        rgbProps.setChunk(3, chunk_dims);
        rgbProps.setDeflate(8);

        H5::DataSet newRgbDset = newFile.createDataSet("rgb", H5::PredType::NATIVE_UINT8, filespace, rgbProps);
        newRgbDset.write(rgb_buffer.data(), H5::PredType::NATIVE_UINT8);

        H5::DataSpace att_space(H5S_SCALAR);
        int simStep;
        double simTime;
        originalRgbDset.openAttribute("SimulationStep").read(H5::PredType::NATIVE_INT, &simStep);
        originalRgbDset.openAttribute("SimulationTime").read(H5::PredType::NATIVE_DOUBLE, &simTime);
        newRgbDset.createAttribute("SimulationStep", H5::PredType::NATIVE_INT, att_space).write(H5::PredType::NATIVE_INT, &simStep);
        newRgbDset.createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE, att_space).write(H5::PredType::NATIVE_DOUBLE, &simTime);
    }

    // --- Process All Other Grid Datasets ---
    {
        H5::DSetCreatPropList props;
        hsize_t chunk_dims[2] = {256, 256};
        props.setChunk(2, chunk_dims);
        props.setDeflate(4);

        std::vector<float> buffer;
        for (const auto& name : GRID_DATASET_NAMES) {
            if (H5Lexists(originalFile.getId(), name.c_str(), H5P_DEFAULT)) {
                H5::DataSet originalDset = originalFile.openDataSet(name);
                buffer.resize(originalDset.getSpace().getSelectNpoints());
                originalDset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);
                H5::DataSet newDset = newFile.createDataSet(name, H5::PredType::NATIVE_FLOAT, originalDset.getSpace(), props);
                newDset.write(buffer.data(), H5::PredType::NATIVE_FLOAT);
            }
        }
    }

    std::cout << "  Successfully compressed: " << filePath.filename().string() << "\n";
}
