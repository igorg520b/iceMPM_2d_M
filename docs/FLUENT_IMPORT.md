# FLUENT Import: Flow Field Generation from CFD Data

Implementation of steady-state flow field import from FLUENT CFD simulations via geometry specification in SVG format.

---

## Overview

The FLUENT import system enables users to use velocity fields computed by ANSYS FLUENT CFD solver as ocean current data in plateMPM simulations. The workflow:

1. **Prepare geometry**: Define image bounds and FLUENT domain in SVG file
2. **Export FLUENT data**: CAS (mesh) + DAT (velocity) files from FLUENT
3. **Configure in JSON**: Specify file paths and geometry references
4. **plate_preparer**: Automatically imports and rasterizes FLUENT grid → grid_flow.h5
5. **Simulation**: Uses same HDF5 format as constant/wave flows

---

## Configuration

### JSON Parameters

Add the following fields to your JSON configuration file when using FLUENT flow:

```json
{
  "FlowType": "FLUENT-static",
  "InputFluentCAS": "path/to/mesh.cas",
  "InputFluentDAT": "path/to/velocity.dat",
  "SVG": "path/to/geometry.svg",
  "RectanglePathID": "image_bounds",
  "FluentPathID": "fluent_domain"
}
```

**Fields:**
- `FlowType`: Must be `"FLUENT-static"` (indicates steady, not time-varying)
- `InputFluentCAS`: Path to FLUENT case file (mesh definition, in project directory)
- `InputFluentDAT`: Path to FLUENT data file (velocity data, in project directory)
- `SVG`: Path to SVG file with geometry definitions (in project directory)
- `RectanglePathID`: XML ID of SVG path/rect defining full image bounds (e.g., `"image_bounds"`)
- `FluentPathID`: XML ID of SVG path/rect defining FLUENT grid region (e.g., `"fluent_domain"`)

**Example SVG geometry:**
```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 5000 3000">
  <!-- Full image bounds in physical coordinates -->
  <rect id="image_bounds" x="100" y="200" width="4900" height="2800"/>

  <!-- FLUENT domain bounds (smaller region within image) -->
  <rect id="fluent_domain" x="500" y="800" width="2000" height="1200"/>
</svg>
```

---

## Implementation Architecture

### 1. ParameterParser Enhancement

**File:** `preparer/parameterparser.h`

Added FLUENT-specific fields:

```cpp
struct ParameterParser {
    // Existing fields...

    // FLUENT-specific parameters (optional, only used when FlowType == "FLUENT-static")
    std::string InputFluentDAT = "";   // HDF5 file: velocity data
    std::string InputFluentCAS = "";   // HDF5 file: mesh definition
    std::string SVG = "";              // SVG file: geometry + path definitions
    std::string RectanglePathID = "";  // SVG path ID: image bounds
    std::string FluentPathID = "";     // SVG path ID: FLUENT grid bounds
};
```

**File:** `preparer/parameterparser.cpp`

Parser extracts FLUENT fields from JSON:

```cpp
if(doc.HasMember("InputFluentDAT")) InputFluentDAT = doc["InputFluentDAT"].GetString();
if(doc.HasMember("InputFluentCAS")) InputFluentCAS = doc["InputFluentCAS"].GetString();
if(doc.HasMember("SVG")) SVG = doc["SVG"].GetString();
if(doc.HasMember("RectanglePathID")) RectanglePathID = doc["RectanglePathID"].GetString();
if(doc.HasMember("FluentPathID")) FluentPathID = doc["FluentPathID"].GetString();
```

### 2. FluentFlowImporter Class

**File:** `preparer/fluentflowimporter.h`

Core class that orchestrates the 4-step import workflow:

```cpp
class FluentFlowImporter {
public:
    FluentFlowImporter();
    ~FluentFlowImporter();

    // Main workflow: orchestrate SVG + FLUENT loading and rasterization
    void Import(const std::string& configDirectory,
                const std::string& casFile,
                const std::string& datFile,
                const std::string& svgFile,
                const std::string& rectanglePathID,
                const std::string& fluentPathID,
                int imageWidth,
                int imageHeight);

    // Output data
    int image_width = 0;
    int image_height = 0;
    std::vector<double> vx_data;
    std::vector<double> vy_data;

    // For debugging: transformed grid visualization
    vtkSmartPointer<vtkUnstructuredGrid> transformed_grid;

private:
    // Step 1: Extract SVG bounding boxes
    void LoadSVGGeometry(const std::string& svgFile,
                         const std::string& rectanglePathID,
                         const std::string& fluentPathID);

    // Step 2: Load FLUENT mesh and velocity data
    void LoadFluentGrid(const std::string& casFile,
                        const std::string& datFile);

    // Step 3: Transform to image coordinate space
    void TransformFluentGrid();

    // Step 4: Rasterize to regular pixel grid
    void RasterizeToImageGrid(int imageWidth, int imageHeight);

    // SVG-derived extents
    Eigen::Vector2f extents_rectangle[2];  // Rectangle bounds (image space)
    Eigen::Vector2f extents_fluent[2];     // FLUENT grid bounds (image space)

    // VTK objects
    vtkFLUENTCFFCustomReader* reader = nullptr;
    vtkTransform* transform = nullptr;
    vtkTransformFilter* transformFilter = nullptr;
    vtkImageData* imageData = nullptr;
    vtkProbeFilter* probeFilter = nullptr;
};
```

**File:** `preparer/fluentflowimporter.cpp`

#### Step 1: LoadSVGGeometry()

Extracts bounding boxes from SVG file using nanosvg library:

```cpp
void FluentFlowImporter::LoadSVGGeometry(const std::string& svgFile,
                                        const std::string& rectanglePathID,
                                        const std::string& fluentPathID) {
    NSVGimage* svgImage = nsvgParseFromFile(svgFile.c_str(), "px", 96);
    if (!svgImage) {
        throw std::runtime_error("Failed to parse SVG file: " + svgFile);
    }

    // Iterate through SVG shapes to find rectangle and fluent paths
    for (NSVGshape* shape = svgImage->shapes; shape != nullptr; shape = shape->next) {
        if (!shape->id) continue;

        std::string shapeId(shape->id);

        if (shapeId == rectanglePathID) {
            // Rectangle defines full image bounds in SVG space
            // bounds[0]=xmin, bounds[1]=ymin, bounds[2]=xmax, bounds[3]=ymax
            Eigen::Vector2f svg_rect_min(shape->bounds[0], shape->bounds[1]);
            Eigen::Vector2f svg_rect_max(shape->bounds[2], shape->bounds[3]);
            extents_rectangle[0] = svg_rect_min;
            extents_rectangle[1] = svg_rect_max;
        }
        else if (shapeId == fluentPathID) {
            // FLUENT path defines where FLUENT grid sits in SVG space
            Eigen::Vector2f svg_fluent_min(shape->bounds[0], shape->bounds[1]);
            Eigen::Vector2f svg_fluent_max(shape->bounds[2], shape->bounds[3]);
            extents_fluent[0] = svg_fluent_min;
            extents_fluent[1] = svg_fluent_max;
        }
    }

    nsvgDelete(svgImage);
}
```

**Input:** SVG file in scientific coordinates (origin bottom-left, Y points up)
**Output:** Two bounding boxes stored in `extents_rectangle` and `extents_fluent`
**Algorithm:** Iterates through NSVGshape objects, finds shapes matching IDs, extracts bounds field

#### Step 2: LoadFluentGrid()

Loads FLUENT mesh and velocity data using VTK reader:

```cpp
void FluentFlowImporter::LoadFluentGrid(const std::string& casFile,
                                        const std::string& datFile) {
    // Use custom VTK FLUENT reader (vtkFLUENTCFFCustomReader)
    reader->SetDataFileName(datFile.c_str());    // Velocity data
    reader->SetFileName(casFile.c_str());        // Mesh data
    reader->Update();
}
```

**Input:** FLUENT CAS/DAT files
**Output:** VTK unstructured grid with velocity arrays (SV_U, SV_V)
**Dependencies:** vtkFLUENTCFFCustomReader (custom reader in simulation/data_manager/fluent_importer/)

#### Step 3: TransformFluentGrid()

Transforms FLUENT grid coordinates from scientific space to image pixel space:

```cpp
void FluentFlowImporter::TransformFluentGrid() {
    // Get original grid bounds from FLUENT mesh
    vtkUnstructuredGrid* grid = vtkUnstructuredGrid::SafeDownCast(
        reader->GetOutput()->GetBlock(0));
    double original_bounds[6];
    grid->GetBounds(original_bounds);

    Eigen::Vector2f source_min(original_bounds[0], original_bounds[2]);
    Eigen::Vector2f source_max(original_bounds[1], original_bounds[3]);
    Eigen::Vector2f source_dims = source_max - source_min;

    // Map SVG extents to image pixel space
    Eigen::Vector2f svg_rect_min = extents_rectangle[0];
    Eigen::Vector2f svg_rect_max = extents_rectangle[1];
    Eigen::Vector2f svg_rect_dims = svg_rect_max - svg_rect_min;

    // Compute scale (pixels per unit in SVG space)
    float scale = svg_rect_dims.x() / source_dims.x();

    // Map FLUENT domain extents from SVG to image space
    Eigen::Vector2f target_min = (extents_fluent[0] - svg_rect_min) * scale;
    Eigen::Vector2f target_max = (extents_fluent[1] - svg_rect_min) * scale;

    // Y-flip: convert from SVG Y-axis (up) to image Y-axis (down)
    float old_y_min = target_min.y();
    float old_y_max = target_max.y();
    target_min.y() = image_height - old_y_max;
    target_max.y() = image_height - old_y_min;

    // Create VTK transformation (right-multiply order)
    transform->Identity();
    transform->Translate(target_min.x(), target_min.y(), 0.0);
    transform->Scale(scale, scale, 1.0);
    transform->Translate(-source_min.x(), -source_min.y(), 0.0);
    transform->Update();

    // Apply transform to grid
    transformFilter->SetTransform(transform);
    transformFilter->SetInputData(grid);
    transformFilter->Update();
}
```

**Algorithm:**
1. Extract FLUENT grid bounds in physical space
2. Compute scale factor from SVG rectangle to image pixels
3. Map FLUENT domain bounds to image coordinate space
4. Apply Y-flip (scientific bottom-left origin → image top-left origin)
5. Create VTK transform matrix
6. Apply to unstructured grid

**Coordinate System Conversion:**
```
SVG Space (scientific)           Image Space (pixels)
Origin: bottom-left              Origin: top-left
Y-axis: up                        Y-axis: down
```

#### Step 4: RasterizeToImageGrid()

Interpolates velocity values from transformed VTK grid to regular pixel grid:

```cpp
void FluentFlowImporter::RasterizeToImageGrid(int imageWidth, int imageHeight) {
    // Convert cell data to point data (required for probing)
    vtkNew<vtkCellDataToPointData> cellToPoint;
    cellToPoint->SetInputData(transformFilter->GetOutput());
    cellToPoint->Update();

    vtkUnstructuredGrid* ug = cellToPoint->GetUnstructuredGridOutput();
    if (!ug->GetPointData()->HasArray("SV_V") ||
        !ug->GetPointData()->HasArray("SV_U")) {
        throw std::runtime_error("FLUENT grid missing velocity arrays (SV_V, SV_U)");
    }

    // Create regular image grid (pixel locations)
    imageData->SetDimensions(imageWidth, imageHeight, 1);
    imageData->SetSpacing(1.0, 1.0, 1.0);

    // Probe VTK grid at image pixel locations
    probeFilter->SetInputData(imageData);
    probeFilter->SetSourceData(cellToPoint->GetOutput());
    probeFilter->PassPointArraysOn();
    probeFilter->Update();

    // Extract velocity components
    vtkImageData* probedData = vtkImageData::SafeDownCast(probeFilter->GetOutput());
    vtkDoubleArray* sv_v = vtkDoubleArray::SafeDownCast(
        probedData->GetPointData()->GetArray("SV_V"));
    vtkDoubleArray* sv_u = vtkDoubleArray::SafeDownCast(
        probedData->GetPointData()->GetArray("SV_U"));

    // Fill output arrays
    vx_data.resize(imageWidth * imageHeight);
    vy_data.resize(imageWidth * imageHeight);

    for (int i = 0; i < imageWidth; ++i) {
        for (int j = 0; j < imageHeight; ++j) {
            int idx = i + imageWidth * j;
            vx_data[idx] = sv_u->GetValue(idx);  // SV_U = vx
            vy_data[idx] = sv_v->GetValue(idx);  // SV_V = vy
        }
    }
}
```

**Algorithm:**
1. Convert cell data to point data (required for probe filter)
2. Create regular image grid with pixel dimensions
3. Use vtkProbeFilter to interpolate velocity at each pixel
4. Extract SV_U (x-velocity) and SV_V (y-velocity) arrays
5. Fill output vectors in row-major order: `idx = i + width*j`

---

### 3. FlowFieldGenerator Enhancement

**File:** `preparer/flowfieldgenerator.h`

Added unified dispatcher method:

```cpp
class FlowFieldGenerator {
public:
    // Unified flow generation dispatcher
    void GenerateFlow(const ParameterParser& params);

private:
    void GenerateConstantFlow(...);  // Existing
    void GenerateWaveFlow(...);      // Existing
    void GenerateFluentFlow(const ParameterParser& params);  // NEW
};
```

**File:** `preparer/flowfieldgenerator.cpp`

#### GenerateFlow() Dispatcher

```cpp
void FlowFieldGenerator::GenerateFlow(const ParameterParser& params) {
    if (params.FlowType == "constant") {
        GenerateConstantFlow(params.GridXTotal, params.GridYTotal, params.cellsize,
                            params.ModeledRegionOffsetX, params.ModeledRegionOffsetY,
                            params.FlowBearing, params.FlowSpeed,
                            params.ProjectDirectory, params.CompressFlow);
    }
    else if (params.FlowType == "wave") {
        GenerateWaveFlow(params.GridXTotal, params.GridYTotal, params.cellsize,
                        params.ModeledRegionOffsetX, params.ModeledRegionOffsetY,
                        params.FlowBearing, params.WaveAmplitude, params.WaveLength,
                        params.PhaseSpeed, params.NFrames,
                        params.ProjectDirectory, params.CompressFlow);
    }
    else if (params.FlowType == "FLUENT-static") {
        GenerateFluentFlow(params);
    }
}
```

#### GenerateFluentFlow() Implementation

```cpp
void FlowFieldGenerator::GenerateFluentFlow(const ParameterParser& params) {
    spdlog::info("Generating FLUENT flow field...");

    // Import FLUENT velocity field
    FluentFlowImporter importer;
    importer.Import(params.ConfigFileDirectory,
                   params.InputFluentCAS,
                   params.InputFluentDAT,
                   params.SVG,
                   params.RectanglePathID,
                   params.FluentPathID,
                   params.InitializationImageSizeX,
                   params.InitializationImageSizeY);

    // Extract velocity data (full image size)
    std::vector<double> vx_full = importer.vx_data;
    std::vector<double> vy_full = importer.vy_data;

    // Extract modeled region subset [GridXTotal × GridYTotal]
    int gx = params.GridXTotal;
    int gy = params.GridYTotal;
    int ox = params.ModeledRegionOffsetX;
    int oy = params.ModeledRegionOffsetY;
    int width = params.InitializationImageSizeX;

    std::vector<double> vx_data(gx * gy);
    std::vector<double> vy_data(gx * gy);

    // Extract modeled region from full image
    for (int i = 0; i < gx; ++i) {
        for (int j = 0; j < gy; ++j) {
            int src_idx = (i + ox) + width * (j + oy);  // full image index
            int dst_idx = i + gx * j;                    // modeled region index
            vx_data[dst_idx] = vx_full[src_idx];
            vy_data[dst_idx] = vy_full[src_idx];
        }
    }

    // Package into frame containers (single frame for steady flow)
    std::vector<std::vector<double>> vx_frames = {vx_data};
    std::vector<std::vector<double>> vy_frames = {vy_data};

    // Write to HDF5 using standard format
    double time_interval = 0.0;  // Static flow (single frame)
    int loop_mode = 1;           // Hold last frame (irrelevant for single frame)

    WriteFlowFieldToHDF5(gx, gy, 1, time_interval, loop_mode,
                        params.ProjectDirectory, params.CompressFlow,
                        vx_frames, vy_frames);

    spdlog::info("FLUENT flow field generated and written to grid_flow.h5");
}
```

**Workflow:**
1. Create FluentFlowImporter instance
2. Import FLUENT data and rasterize to full image size
3. Extract modeled region subset using OffsetX/Y
4. Package into single frame (steady flow)
5. Write to grid_flow.h5 using standard HDF5 format (same as constant/wave)

---

### 4. plate_preparer Integration

**File:** `preparer/mainwindow.cpp`

Simplified flow generation call:

```cpp
if (!params.FlowType.empty()) {
    FlowFieldGenerator flowGen;
    flowGen.GenerateFlow(params);  // Unified dispatcher handles all 3 types

    // Initialize wind/current interpolator
    hsd.waci.SetHDF5Path(projectDir + "/grid_flow.h5");
    hsd.waci.SetTime(0.0);
    spdlog::info("Preparer: Flow field generated successfully and initialized to t=0");
}
```

No need for if/else branches - dispatcher automatically routes to correct flow generator.

---

### 5. Build Configuration

**File:** `CMakeLists.txt`

Added FluentFlowImporter files to plate_preparer target:

```cmake
add_executable(plate_preparer
    preparer/preparer_main.cpp
    preparer/mainwindow.cpp
    preparer/mainwindow.h
    preparer/mainwindow.ui

    preparer/parameterparser.h
    preparer/parameterparser.cpp

    preparer/flowfieldgenerator.cpp
    preparer/flowfieldgenerator.h

    # FLUENT flow import
    preparer/fluentflowimporter.h
    preparer/fluentflowimporter.cpp

    # ... rest of existing files ...
)
```

Dependencies already in place:
- VTK libraries (required for vtkFLUENTCFFCustomReader)
- nanosvg (required for SVG geometry extraction)
- Eigen (required for vector math)

---

## Data Flow Diagram

```
JSON Configuration
    ↓
    ├─ FlowType: "FLUENT-static"
    ├─ InputFluentCAS, InputFluentDAT
    ├─ SVG, RectanglePathID, FluentPathID
    ↓
ParameterParser::ParseFile()
    ↓
FlowFieldGenerator::GenerateFlow()
    ↓
FluentFlowImporter::Import()
    ├─ Step 1: LoadSVGGeometry() → extents_rectangle, extents_fluent
    ├─ Step 2: LoadFluentGrid() → VTK unstructured grid
    ├─ Step 3: TransformFluentGrid() → SVG→image coordinate transform
    └─ Step 4: RasterizeToImageGrid() → vx_data[w*h], vy_data[w*h]
    ↓
Region extraction [GridXTotal × GridYTotal]
    ↓
WriteFlowFieldToHDF5()
    ↓
grid_flow.h5 (single frame, steady flow)
```

---

## Key Design Features

### Modular Architecture
- **FluentFlowImporter**: Self-contained, independent of flow generator
- **Dispatcher pattern**: GenerateFlow() routes to correct generator
- **Standard format**: FLUENT output written as single-frame HDF5 (compatible with simulation)

### Coordinate System Handling
- **Input**: FLUENT in physical/scientific coordinates (origin bottom-left, Y up)
- **SVG geometry**: Defines mapping from FLUENT space to image space
- **Output**: Pixel coordinates (origin top-left, Y down)
- **Y-flip**: Automatic conversion during transformation

### Rasterization Strategy
- **VTK-native**: Uses vtkProbeFilter for interpolation (handles gaps, extrapolation)
- **Bilinear interpolation**: Smooth velocity field at pixel boundaries
- **No manual interpolation**: Avoids implementation errors

### Region Extraction
- **Full→modeled region**: Extract subset after rasterization
- **Indexing**: Row-major, same as constant/wave flows
- **Offset handling**: Uses ModeledRegionOffsetX/Y to map image→grid coordinates

---

## Example Workflow

### 1. Prepare SVG Geometry

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 5000 3000" width="5000" height="3000">
  <!-- Full image bounds (what we're simulating) -->
  <rect id="image_bounds" x="0" y="0" width="5000" height="3000" fill="none" stroke="blue"/>

  <!-- FLUENT domain (where we have CFD data) -->
  <rect id="fluent_domain" x="500" y="600" width="3000" height="1800" fill="none" stroke="red"/>
</svg>
```

### 2. Configure JSON

```json
{
  "GridData": "input/grid.h5",
  "InputPNG": "input/satellite_image.png",
  "InitializationSnapshot": "input/s00000.h5",
  "GridXTotal": 1000,
  "GridYTotal": 600,
  "ModeledRegionOffsetX": 100,
  "ModeledRegionOffsetY": 120,
  "InitializationImageSizeX": 5000,
  "InitializationImageSizeY": 3000,

  "FlowType": "FLUENT-static",
  "InputFluentCAS": "input/fluent_mesh.cas",
  "InputFluentDAT": "input/fluent_velocity.dat",
  "SVG": "input/geometry.svg",
  "RectanglePathID": "image_bounds",
  "FluentPathID": "fluent_domain"
}
```

### 3. Run plate_preparer

```bash
./plate_preparer config.json
```

**Output:**
```
INFO: FlowFieldGenerator::GenerateFlow
INFO: Generating FLUENT flow field...
INFO: FluentFlowImporter::Import starting
INFO: FluentFlowImporter::LoadSVGGeometry from: input/geometry.svg
INFO: Rectangle bounds: (0, 0) to (5000, 3000)
INFO: FLUENT path bounds: (500, 600) to (3500, 2400)
INFO: FluentFlowImporter::LoadFluentGrid
INFO: FLUENT grid loaded successfully
INFO: FluentFlowImporter::TransformFluentGrid
INFO: FLUENT grid original bounds: X[...] Y[...]
INFO: Target extents (image space): (...) to (...)
INFO: FLUENT grid transformation complete
INFO: FluentFlowImporter::RasterizeToImageGrid
INFO: Rasterization complete: 5000 x 3000 grid
INFO: FLUENT flow field generated and written to grid_flow.h5
```

### 4. Use in Simulation

The generated `grid_flow.h5` is automatically loaded by the simulator using the standard WindAndCurrentInterpolator interface.

---

## Troubleshooting

### SVG Parsing Error

**Error:** `Failed to parse SVG file: input/geometry.svg`

**Check:**
- SVG file exists and is valid XML
- Path is correct relative to project directory
- SVG elements have `id` attributes matching RectanglePathID and FluentPathID

### Missing Velocity Arrays

**Error:** `FLUENT grid missing velocity arrays (SV_V, SV_U)`

**Check:**
- FLUENT DAT file contains velocity data (not just pressure/temperature)
- Custom VTK reader correctly extracts velocity arrays
- FLUENT case uses standard variable names

### Dimension Mismatch

**Error:** `FLUENT grid has invalid dimensions` or rasterized data is all zeros

**Check:**
- FLUENT grid bounds in SVG match actual CFD domain bounds
- InitializationImageSizeX/Y match SVG viewBox dimensions
- GridXTotal/GridYTotal ≤ InitializationImageSizeX/Y
- ModeledRegionOffsetX/Y place grid within image bounds

### Transform Looks Wrong

**Debug:**
- Check transformed_grid visualization if available
- Verify SVG bounds are in same coordinate system as FLUENT
- Confirm Y-flip is applied (FLUENT Y-up → image Y-down)

---

## Performance Notes

### Memory Usage
- **Intermediate**: Full image size velocity arrays [width × height × 2] (temporary)
- **Final**: Modeled region [gx × gy × 2] + single-frame HDF5

### Computation Time
- **SVG parsing**: Milliseconds
- **FLUENT loading**: Seconds (depends on grid size)
- **Rasterization**: Seconds (VTK interpolation)
- **Total**: Usually 10-30 seconds for typical CFD grids

### File Size
- **grid_flow.h5**: Typically 5-50 MB (single frame, may compress to 1-5 MB)
- **Uncompressed velocity data**: ~32 MB for 1000×600 grid (8 bytes × 2 × 1M points)

---

## Limitations & Future Improvements

### Current Limitations
1. **Steady flow only**: Single frame (time_interval = 0)
2. **Rectangular mapping**: Assumes rectangular FLUENT domain in SVG
3. **Rectangular bounds**: SVG paths must have bounding box (not arbitrary paths)
4. **One FLUENT per simulation**: No support for multiple FLUENT imports

### Possible Extensions
1. **Time-varying FLUENT**: Export multiple frames from FLUENT transient simulation
2. **Non-rectangular mapping**: Support arbitrary path geometries
3. **Field scaling/normalization**: Adjust velocity magnitude if needed
4. **Comparison tools**: Visualize FLUENT grid vs rasterized output

---

## Testing Checklist

- [ ] SVG file with rectangle shapes parses correctly
- [ ] FLUENT CAS/DAT files load and display grid in VTK
- [ ] Coordinate transformation maps FLUENT bounds to image bounds correctly
- [ ] Rasterized velocity field has expected dimensions [width × height]
- [ ] Extracted region [gx × gy] matches modeled domain
- [ ] grid_flow.h5 file is created with correct HDF5 structure
- [ ] Simulation reads and uses grid_flow.h5 velocity field
- [ ] Velocity field visualization shows expected direction/magnitude
- [ ] Time comparison with constant flow shows FLUENT field in expected region

---

This documentation covers the complete FLUENT import implementation. See CLASS_REFERENCE.md for API details on FluentFlowImporter and FlowFieldGenerator classes.
