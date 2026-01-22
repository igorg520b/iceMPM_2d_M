#ifndef COMPRESSOR_CORE_H
#define COMPRESSOR_CORE_H

#include <string>
#include <vector>
#include <hdf5.h>

extern const std::vector<std::string> GRID_DATASET_NAMES; // Keeping for compatibility or just removing? 
// Actually I removed GRID_DATASET_NAMES usage in core.cpp, so I should remove it here too.
extern const std::vector<std::string> FLOAT_DATASETS;
extern const std::vector<std::string> UINT8_DATASETS;
extern const std::vector<std::string> SUBCATEGORIES;

/**
 * Processes a single frame file: reads from inputFile, compresses designated datasets,
 * and writes to outputFile.
 *
 * @param inputFile Path to the original HDF5 file.
 * @param outputFile Path where the compressed HDF5 file will be written.
 * @param overwrite If true, successful compression triggers deletion of inputFile 
 *                  and renaming of outputFile to inputFile.
 * @return True if processing (and optional overwrite) was successful, false otherwise.
 */
bool process_frame_file(const std::string& inputFile, const std::string& outputFile, bool overwrite);

#endif // COMPRESSOR_CORE_H
