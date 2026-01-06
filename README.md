# iceMPM_multi
Multi-GPU implementation of MPM for modeling of ice (2D version)

## Project Overview

This project implements a high-resolution, GPU-accelerated Material Point Method (MPM) to simulate the breakup of sea ice. The simulation captures complex ice dynamics such as fracture, ridging, and dispersion under ocean current forcing.

The implementation is written in CUDA C++ and designed to run efficiently on multi-GPU systems. It uses double-precision arithmetic and supports over 100 million material points on a large-scale 2D domain. The computational domain and initial ice cover are initialized from satellite imagery, with simulation output reflecting key physical processes like stress accumulation and fracture propagation.

The method uses a Moving Least Squares (MLS) formulation for MPM with an elliptical failure criterion for ice fracture and a Drucker-Prager plasticity model to simulate granular post-failure behavior. Ocean current drag is applied as the primary external force, and the solver runs explicit time integration with fixed timestep.

This codebase serves as a research tool to explore sea ice fragmentation and motion, with potential extensions to other geophysical and engineering problems involving large deformation and progressive material failure.

Some formulation details are available here: [PDF](/screenshots/POAC25_paper_71.pdf)

[High-resolution simulation result](https://youtu.be/OyP-zuegrtc)

![Screenshot of the simulation result](/screenshots/snapshot.png)


## 🧱 Dependencies & Build Requirements

This project requires a modern Linux environment with CUDA-capable GPUs. It has been tested on:

- **Ubuntu 24.04**
- **Ubuntu 25.04**

### ✅ Minimum Build Tools

- [CMake](https://cmake.org/) (version ≥ 3.20.2 recommended)
- `gcc` and `g++` (Ubuntu default versions for 24.04/25.04 are sufficient)
- NVIDIA CUDA Toolkit with:
  - `nvcc` compiler
  - CUDA runtime and drivers

> ⚠️ **CUDA Device Requirement**  
> A GPU with **Compute Capability 6.1 or higher** is required.  
> Recommended GPUs for high performance:
> - **RTX 4090 / 5090**
> - **NVIDIA A100 / H100**

---

### 📦 Required Ubuntu Packages

Install the following dependencies using APT:

```bash
sudo apt update && sudo apt install -y \
    cmake gcc g++ \
    libeigen3-dev \
    libspdlog-dev \
    libcxxopts-dev \
    rapidjson-dev \
    libhdf5-dev \
    libopenjp2-7-dev \
    libdxflib-dev
```

These are required for both the command-line and GUI versions. Command-line version is normally run on a remote server. GUI version consumes slightly more RAM, but has a more intuitive visualization.

### 🖼️ GUI Version (Optional)

CLI version is typically executed on a server, whereas GUI is the desktop version. In Cmake configuration, enable the options BUILD_GUI_VERSION, BUILD_POSTPROCESSOR, and BUILD_PREPARER (by default they are off). Install the following packages:

```
sudo apt install qtcreator libvtk9-dev libvtk9-qt-dev ffmpeg libnanosvg-dev
```

The GUI version can be built with or without QtCreator, but QtCreator is recommended for modifying and managing the GUI components.

Ffmpeg is used to convert the resulting images into animations.


## 🛠️ Build Instructions

With all required libraries installed (see [Dependencies](#-dependencies--build-requirements)), building the project should be straightforward using CMake.

### 🔧 Recommended Build Options (Desktop / GUI Use)

To enable full desktop functionality — including the simulation GUI, data preparation tools, and postprocessing/visualization — enable the following CMake options:

- `BUILD_GUI_VERSION=ON`
- `BUILD_POSTPROCESSOR=ON`
- `BUILD_PREPARER=ON`

These can be passed to CMake via command line:

```bash
cmake -B build -S . \
    -DBUILD_GUI_VERSION=ON \
    -DBUILD_POSTPROCESSOR=ON \
    -DBUILD_PREPARER=ON
cmake --build build -j
```

> 💡 Tip: GUI builds are best managed using QtCreator, which can open the project directly via CMakeLists.txt.

### ⚙️ CLI-Only Build (e.g., Server-Side)

For compute servers or headless environments, build the complete project (no special configuration needed):
```
cmake -B build -S .
cmake --build build -j
```

The CLI simulation executable is `cplate`. Run headless simulations with:
```bash
./build/cplate config.json
```

The CLI version generates relatively compact HDF5 output files which can be downloaded from the server and visualized later using the visualizer. This option is best suited for large simulations (>10M points).

### 🎯 CUDA Architecture Flags

In `CMakeLists.txt`, you may find lines like the following for the simulation targets:
```cmake
set_target_properties(gplate PROPERTIES CUDA_ARCHITECTURES "80;89")
set_target_properties(cplate PROPERTIES CUDA_ARCHITECTURES "80;89")
```

You can customize these lines to match your GPU's compute capability. The minimum supported version is 6.1, although this hasn't been explicitly tested.

### 🪟 Building on Windows (Experimental)
While not officially tested, the project should be buildable on Windows using MSVC and QtCreator. Assuming that the required libraries are present and accessible, the project should be able to compile and run in Windows.



## 🔄 Complete Workflow: Prepare → Simulate → Visualize → Compress

This project follows a four-stage pipeline to simulate and visualize sea ice dynamics:

```
1. PREPARE (plate_preparer)     → Generate grid.h5 and initial snapshot
2. SIMULATE (gplate/cplate)     → Run physics simulation, save frames
3. VISUALIZE (visualizer)       → View and analyze results, generate frames
4. COMPRESS (compressor)        → Compress rendered frames for archival
```

---

## Stage 1: Data Preparation (plate_preparer)

The preparation stage converts geographical data (PNG images) into HDF5 format for the simulation. This comprehensive guide covers the purpose, design, and workflow of plate_preparer.

### Purpose of plate_preparer

plate_preparer converts PNG image data into initial conditions for the MPM ice dynamics simulation:
- Reads a **landmask** image to identify modeled region (water/ice) vs. land
- Reads a **color image** for background visualization
- Reads an **ice mask** to identify ice-covered areas
- Reads a **crushed ice mask** to mark ice damage/thickness
- Generates a structured **computational grid** covering the modeled region
- Distributes **material points** (particles) throughout the ice domain using Poisson disk sampling
- Exports grid and point data to **HDF5 files** for simulation

### Coordinate System Design

A critical design decision in plate_preparer is the coordinate system conversion between image and simulation space.

**Image Coordinate System (PNG):**
- Origin (0,0) at **top-left** corner
- X increases to the right
- Y increases downward

**Simulation Coordinate System:**
- Origin (0,0) at **bottom-left** corner
- X increases to the right
- Y increases upward

**Solution: Y-Flip During Loading**

All images are loaded and immediately flipped vertically during `PrepareGridAndPoints()`. After flipping, all subsequent operations use consistent simulation coordinates. **No additional Y-flip is needed** elsewhere in the code.

```cpp
for (int y = 0; y < height; y++) {
    int y_flipped = height - y - 1;
    for (int x = 0; x < width; x++) {
        // Copy from y_flipped to y in flipped buffers
    }
}
```

**Grid Offset and Origin:**
- `ModeledRegionOffsetX, ModeledRegionOffsetY`: Position of modeled region (in flipped image coordinates)
- `GridXTotal, GridYTotal`: Dimensions of computational grid
- Grid cells are indexed from (ox, oy) to (ox + gx - 1, oy + gy - 1)
- Grid cell **centers** correspond to physical coordinates: (i*h, j*h) where h = cellsize

### Grid Design

#### Column-Major Indexing

Grid data uses **column-major** (Fortran-style) indexing for cache efficiency:

```cpp
// Grid cell (i, j) maps to array index:
size_t idx = j + (size_t)i * GridYTotal;
```

This layout is critical for GPU memory access patterns in the simulation stage.

#### Grid Structure

- Raster grid coordinate mapping with cell centers at grid coordinates
- Raster origin: (0, 0) - bottom-left of image in physical space
- Raster dimensions: width*h × height*h in physical coordinates
- Cell (i,j) center at physical coordinates: (i*h, j*h)

#### Landmask Buffer

The landmask stores the modeled region status for each grid cell:
- Value `255`: Modeled area (water/ice)
- Value `0-254`: Land (no material points, excluded from simulation)

### Material Points (Particles)

#### Point Generation: Poisson Disk Sampling

Material points are distributed using **Poisson disk sampling** to avoid clustering:
- **Target**: ~PointsPerCell points per grid cell (configurable)
- **Caching**: Points are cached in `_data/poisson_cache/` to avoid expensive recomputation (can take >1 hour for large grids)
- **Cache key**: `points_GridX×GridY_PointsPerCell.h5`

#### Point Filtering

Generated points are filtered by:
1. **Boundary**: Points within 2 cells of grid boundary are removed
2. **Land mask**: Points outside water/ice region are removed
3. **Ice mask**: Only points in ice-covered areas are retained


#### Point Data Storage (HostSideSOA)

Points are stored in **Structure-of-Arrays** (SOA) format for GPU efficiency:
- **Position**: (x, y) in physical coordinates
- **Velocity**: (vx, vy) - initially zero
- **Deformation gradient**: Fe (2×2 matrix, identity initially)
- **Stress**: Pressure (P) and deviatoric stress (Q)
- **Plasticity**: Jp_inv (inverse plastic compressibility)
- **Color**: RGB from original image (normalized 0-1)
- **Thickness**: From crushed mask (1.0 for intact ice, <1.0 for damaged)

### 1.1 Run plate_preparer GUI

```bash
./build/plate_preparer
```

Launch the plate_preparer application. This opens a graphical interface for setting up your simulation domain.

### 1.2 Open Configuration File

In the application menu, select **File → Open JSON** and choose your configuration file. Example configuration:

```json
{
  "ImageLandMask": "landmask.png",
  "ImageColor": "color.png",
  "ImageIceMask": "ice_mask.png",
  "ImageCrushedMask": "crushed_mask.png",
  "ProjectDirectory": "./output",
  "DimensionHorizontal": 1000.0,
  "PointsPerCell": 5
}
```

**Configuration Fields:**
- `ImageLandMask`: PNG image identifying land vs. water/ice areas
- `ImageColor`: RGB visualization background image
- `ImageIceMask`: Binary mask identifying ice-covered regions
- `ImageCrushedMask`: Optional mask for pre-damaged ice (scale 0-254 for damage levels)
- `ProjectDirectory`: Output folder where grid.h5 and initial snapshot are saved
- `DimensionHorizontal`: Physical width of domain in meters (or your unit)
- `PointsPerCell`: Material points per grid cell (typically 5)

### 1.3 Review Grid and Points

Use the **Visualization Mode** dropdown to inspect your setup:
- `none`: Shows background image with grid boundary overlay and debug grid
- `regions`: Shows modeled area (blue) vs. land (pastel colors)
- `grid_colors`: Shows grid structure with ice colors
- `grid_density`: Shows material point distribution density

**Debug Grid**: Black points appear at 10-cell intervals (0, 10h, 20h, ...) in `none` mode. This grid helps verify cell spacing and coordinate system consistency.

### 1.4 Output Files Generated

When you close plate_preparer, the following files are created in `ProjectDirectory/`:

**grid.h5**
- Computational grid structure with dimensions, cell size, offsets
- Landmask for modeled region (255 = water/ice, 0 = land)
- Visualization colors for grid cells

**s00000.h5**
- Initial material point snapshot
- Point positions, velocities, and properties (22 separate arrays)
- Point colors from original satellite image
- Ready to be loaded by simulation

---

## Stage 2a: Generate Flow Data (grid_flow.h5)

The simulation requires a flow field (ocean currents) as the external forcing. There are **three options** for providing this data:

### Option A: Generate from FLUENT Output

If you have FLUENT CFD mesh and solution files:

1. Open plate_preparer with your configuration
2. Use **Tools → Import FLUENT Grid** to load CFD mesh
3. Maps flow velocity from FLUENT mesh onto your simulation grid
4. Creates `grid_flow.h5` with velocity field

This is the recommended approach for physically-based ocean current data.

### Option B: Create a Uniform Flow Field

For testing or simple scenarios, create a constant velocity field:

```cpp
// Example: Create grid_flow.h5 with uniform flow (1.0 m/s east)
// Use plate_preparer's flow generation tools or write custom HDF5 file
```

### Option C: Use Analytical/Prescribed Flow

For research purposes, provide flow as a mathematical function (e.g., shear flow, rotating vortex):

```cpp
// Example: Rotating flow field
// u(x, y) = -ω * y
// v(x, y) =  ω * x
```

### Flow File Format

Regardless of source, `grid_flow.h5` must contain:

**Dataset: velocity_field**
- 3D array {GridX, GridY, 2} float32
- Stores (vx, vy) velocity components at each grid cell

**Attributes:**
- GridXTotal, GridYTotal: Grid dimensions
- TimeStep: Data collection time interval (if applicable)

---

## Stage 2b: Run Simulation (gplate or cplate)

### 2.1 GUI Version (Recommended for Interactive Use)

```bash
./build/gplate config.json
```

Launch the GUI simulation with a configuration file. The configuration file should be located in the output directory (created by plate_preparer) alongside `grid.h5`, `grid_flow.h5`, and `s00000.h5`. The window shows:
- **Left panel**: Visualization area with grid overlay
- **Toolbar**: Visualization dropdown, value ranges, transparency control
- **Controls**: Play/pause button, frame slider, frame range controls

### 2.2 Visualization Options During Simulation

Select from the **Visualization** dropdown to monitor simulation state:

- `grid_Jpinv`: Relative surface density (1/Jₚ = det(Fₚ)⁻¹)
  - Blue: Dispersed material (lower density)
  - Red: Accumulated material (ridges)

- `grid_P`: In-plane pressure (ice sheet stress)
  - Set range to 5.0 for [0, 10^5.0 Pa]

- `grid_Q`: Deviatoric stress magnitude

- `grid_mass`: Material mass at grid cells

- `grid_vnorm`: Velocity magnitude (|v|)

- `partitions`: GPU partition assignment (for multi-GPU systems)

### 2.3 Controls

- **Play/Pause**: Start/stop simulation
- **Visualization Value Range**: Adjust color scale (first spinbox)
- **GPU Slow-down**: Manually slow GPU computation for UI responsiveness (second spinbox)
- **Frame Slider**: Jump to specific timestep
- **Transparency**: Control visualization opacity

### 2.4 CLI Version (For Remote/Batch Execution)

For headless servers without display:

```bash
./build/cplate config.json
```

The CLI version:
- Runs without GUI, consuming less memory
- Writes output files in the same format
- Can be resumed from snapshots (see below)
- Suitable for large simulations (>10M points)

### 2.5 Resume Interrupted Simulation

If a simulation crashes or is interrupted, resume from the last snapshot:

```bash
./build/cplate config.json --resume 'output/snapshots/s00120.h5'
```

The simulation resumes from the snapshot with full particle state restored. Resume works reliably for the CLI version; GUI version support is experimental.

### 2.6 Simulation Output

Results are saved to `ProjectDirectory/output/`:

**frames/**
- HDF5 files: `f00000.h5, f00001.h5, ...`
- Frame format: Grid-only data (no particles)
- Contains: Visualization data for post-processing
- Size: Relatively compact due to compression

**snapshots/**
- HDF5 files: `s00000.h5, s00120.h5, ...`
- Full state backup: All particle data
- Used for resuming simulations
- Much larger than frame files

---

## Stage 3: Post-Processing and Visualization

The visualizer (post-processor) visualizes simulation results and generates high-quality animation frames.

### 3.1 Launch Visualizer

```bash
./build/visualizer
```

The visualizer opens with an empty visualization. Use the menus to load your simulation results.

### 3.2 Load Project

**File → Open Project...**

Select the JSON configuration file from your simulation. This loads:
- Grid structure and dimensions
- Parameter metadata
- Output directory configuration

### 3.3 Load Frames

**File → Open Frames...**

Navigate to `ProjectDirectory/output/frames/` and open the frames directory. The post-processor:
- Scans for frame files (f00000.h5, f00001.h5, ...)
- Counts total frames
- Loads the last frame for preview
- Enables the frame slider

### 3.4 Navigate and Visualize

**Frame Slider**
- Drag slider to jump between frames
- If grid width > 4000 pixels: Release slider to load (tracking disabled for performance)
- If grid width ≤ 4000 pixels: Real-time loading as you drag (tracking enabled)

**Status Bar** (bottom)
- Shows current frame number
- Displays simulation time in HH:MM:SS format

**Visualization Options** (dropdown)
- Same options as simulation: grid_Jpinv, grid_P, grid_Q, etc.
- Select visualization mode before rendering

**Value Range** (spinbox)
- Adjust color scale range for current visualization
- Default: -2 (covers 0 to 10^-2)

**Tools Menu**
- `&Reset Camera`: Ctrl+Shift+R - Reset view to fit grid
- `&Render Frame`: Ctrl+F - Save high-quality image of current frame
- `&Render All`: F5 - Batch render all frames to JPG images

### 3.5 Render Single Frame

**Tools → Render Frame** (or Ctrl+F)

Saves the current frame as a high-quality JPEG:
- Resolution: 1920×1080 pixels
- Filename: `{visualization_name}_{frame_number}.jpg`
- Location: Depends on where frame slider is positioned
- Status bar confirms: "Rendered frame N as visualization_name"

### 3.6 Render All Frames for Animation

**Tools → Render All** (or F5)

Batch processes all frames for animation creation:
1. Specify frame range using the **Frame From/To** spinboxes
2. Click **Render All**
3. Progress dialog shows rendering status
4. For each visualization type, generates JPG images:
   ```
   output/raster/grid_Jpinv/00000.jpg
   output/raster/grid_Jpinv/00001.jpg
   ...
   output/raster/grid_P/00000.jpg
   output/raster/grid_P/00001.jpg
   ...
   ```
5. Automatically generates `output/raster/genvideo.sh` bash script

### 3.7 Generate Animation Videos

After rendering all frames, convert images to MP4 videos:

```bash
cd output/raster
chmod +x genvideo.sh
./genvideo.sh
```

This runs ffmpeg for each visualization type:
- Input: JPG sequence (00000.jpg, 00001.jpg, ...)
- Output: MP4 video (grid_Jpinv.mp4, grid_P.mp4, grid_Q.mp4, ...)
- Codec: H.264 (libx264)
- Resolution: 1920×1080
- Frame rate: 30 fps
- Quality: CRF 21 (high quality)

Result: `grid_Jpinv.mp4, grid_P.mp4, grid_Q.mp4, grid_vnorm.mp4, ...`

---

## Stage 4: Frame Compression (Optional)

For long-term storage or remote transfer, compress the frame files:

### 4.1 Compress Frame Range

The compressor reads uncompressed frame HDF5 files and writes compressed versions:

```bash
./compressor /path/to/project frames_start frames_end
```

**Example:**
```bash
cd build
./compressor /home/user/project/output/frames 0 500
```

This compresses frames f00000.h5 through f00500.h5 and saves to:
```
output/frames_compressed/f00000.h5
output/frames_compressed/f00001.h5
...
output/frames_compressed/f00500.h5
```

### 4.2 Compression Details

The compressor:
- **RGB dataset**: HDF5 deflate compression level 8 (strong, slower)
- **Grid datasets**: HDF5 deflate compression level 4 (balanced)
- **Chunking**: RGB 128×128×3, Grid 256×256 (optimized for I/O patterns)
- **Typical ratio**: 5-10× compression (depends on data variation)

### 4.3 Resume Compression

If compression is interrupted, resume where it left off:
- Already compressed files are **automatically skipped**
- Only missing files are processed
- Safe to run multiple times on same directory

---

## 📁 Example Setups

Complete example setups are included in the repository:

**Confederation Bridge Test Case** (with FLUENT-imported ocean currents)
- Location: `preparer/input/test_fluent_confederation_bridge/`
- Region: Canadian Maritimes (Confederation Bridge)
- Includes: FLUENT mesh, velocity data, satellite imagery, SVG geometry
- Demonstrates: Complete workflow from data preparation through visualization

**Nares Strait Test Case** (with FLUENT-imported ocean currents)
- Location: `preparer/input/nares_strait/`
- Region: Nares Strait between Greenland and Canada
- Includes: FLUENT mesh, velocity data, satellite imagery, SVG geometry
- Special case: Velocity multiplier increased (3.0×) to compensate for thick ice and 2D drag limitations in capturing natural bending failure modes

See the **"Example: Running the Confederation Bridge Simulation"** section below for step-by-step instructions on preparing the grid, running the simulation, and visualizing results. The same workflow applies to nares_strait.

---

## Configuration Details

The simulation JSON file controls all physics parameters:

### Grid and Domain
- `GridData`: Path to grid.h5 (created by plate_preparer)
- `CurrentVelocityData`: Path to grid_flow.h5 (FLUENT import or prescribed)
- `DimensionHorizontal`: Domain width in meters
- `InitialTimeStep`: Timestep size in seconds (constant throughout)

### Ice Rheology (Elliptic Failure Criterion)
- `IceCompressiveStrength`: Maximum pressure (Pa)
- `IceTensileStrength`: Tensile strength (minimum negative pressure)
- `IceShearStrength`: Maximum deviatoric stress (Pa)

### Plasticity (Drucker-Prager)
- `DP_phi`: Angle of internal friction (degrees)
- `DP_threshold_p`: Transition pressure between elastic and plastic

---

## Troubleshooting

**Post-processor issues on Wayland (Ubuntu 25.04)**
- Some display issues may occur with Wayland window manager
- Workaround: Switch to X11 or use XWayland compatibility

**Slow GUI rendering on weak GPUs**
- Use the GPU slow-down spinbox in simulation to throttle computation
- This prevents UI lag while maintaining accurate simulation

**Large grid (>4000 pixels) feels sluggish**
- Post-processor automatically disables slider tracking for large grids
- Release slider to load frames instead of dragging
- Consider using CLI version for large-scale simulations

---

## Example: Running the Confederation Bridge Simulation

The **Confederation Bridge** test case demonstrates a realistic FLUENT-derived ocean current simulation of sea ice dynamics near the Confederation Bridge in the Canadian Maritimes.

### Setup

The example uses FLUENT-generated flow data and is located in:
```
preparer/input/test_fluent_confederation_bridge/
```

**Files included:**
- `prepare_confederation_bridge_2k.json` - Configuration for grid preparation
- `cb.json` - Simulation configuration
- `FFF.1-1.cas.h5` - FLUENT mesh (CAS file in HDF5 format)
- `FFF.1-1-00000.dat.h5` - FLUENT velocity data (DAT file in HDF5 format)
- `cb_geometry_fluent.svg` - SVG file defining domain geometry and FLUENT region bounds
- `cb_2020-02-18_8000_final_2k.png` - Satellite image of ice coverage
- `land.png`, `ice.png`, `crushed.png` - Color maps for visualization

### Step 1: Prepare Grid and Flow Data

Run the **plate_preparer** to generate `grid.h5` and `grid_flow.h5`:

```bash
./build/plate_preparer preparer/input/test_fluent_confederation_bridge/prepare_confederation_bridge_2k.json
```

This will:
1. Load the satellite image and FLUENT geometry from SVG
2. Create the computational grid (`grid.h5`)
3. Import FLUENT velocity data and rasterize to grid resolution (`grid_flow.h5`)
4. Generate initial particle snapshot (`s00000.h5`)

**Output location:**
```
preparer/input/c_bridge_test_2k/
├── grid.h5           # Computational grid (created)
├── grid_flow.h5      # FLUENT-derived flow field (created)
└── s00000.h5         # Initial particle state (created)
```

### Step 2: Copy Simulation Configuration

Copy the simulation JSON to the output directory:

```bash
cp preparer/input/test_fluent_confederation_bridge/cb.json \
   preparer/input/c_bridge_test_2k/
```

Alternatively, create a symbolic link:
```bash
cd preparer/input/c_bridge_test_2k/
ln -s ../test_fluent_confederation_bridge/cb.json .
```

### Step 3: Run Simulation

Run the simulation from the output directory (where `grid.h5`, `grid_flow.h5`, and `s00000.h5` are located):

**GUI version (interactive):**
```bash
cd preparer/input/c_bridge_test_2k/
./build/gplate cb.json
```

Then in the GUI:
1. Click **"Load"** to load the configuration
2. Click **"Start"** to begin simulation
3. Watch the ice break up and move under FLUENT-derived currents

**CLI version (batch):**
```bash
cd preparer/input/c_bridge_test_2k/
./build/cplate cb.json
```

### Step 4: Visualize Results

Once simulation completes, visualize the results with the **visualizer**:

```bash
./build/visualizer
```

Then in the visualizer:
1. **File → Open Project** → select `preparer/input/c_bridge_test_2k/cb.json`
2. **File → Open Frames** → select the `output/frames/` directory
3. Use the **slider** to step through animation frames
4. Select visualization mode from the **dropdown**:
   - `grid_colors` - Original satellite image colors
   - `grid_P` - Pressure field
   - `grid_Q` - Deviatoric stress
   - `grid_vnorm` - Velocity magnitude
   - `str_vonMises` - Von Mises strain
   - `grid_ridges` - Ridge formation

### Configuration Details (cb.json)

Key parameters for the Confederation Bridge simulation:

```json
{
  "GridData": "grid.h5",
  "CurrentVelocityData": "grid_flow.h5",
  "Snapshot": "s00000.h5",

  "SimulationEndTime": 25000,      // 25000 seconds (~7 hours)
  "AnimationFramePeriod": 10,      // Save frame every 10 steps
  "InitialTimeStep": 0.005,        // 5 millisecond timestep

  "IceCompressiveStrength": 100e6, // Ice strength (Pa)
  "IceShearStrength": 1.0e6,

  "waterDragEffectiveLinear": 0.001,   // Ocean drag parameters
  "waterDragEffectiveQuadratic": 0.1,

  "nPartitions": 1                 // Single GPU (can increase for multi-GPU)
}
```

### Understanding the Results

The simulation will produce:

1. **output/frames/** - Pre-rendered grid visualization (f00000.h5, f00001.h5, ...)
2. **output/snapshots/** - Complete particle state for resuming (s00000.h5, s00001.h5, ...)
3. **output/multisink.txt** - Simulation log with performance metrics

In the post-processor, observe:
- **Ice motion** driven by FLUENT-derived ocean currents
- **Pressure buildup** in stress concentrations
- **Fracture and ridging** as ice yield strength is exceeded
- **Damage progression** shown in the thickness field

### Resume Interrupted Simulation

If the simulation is interrupted, resume it without re-preparing the grid:

```bash
cd preparer/input/c_bridge_test_2k/
./build/cplate cb.json
```

The simulator will:
1. Load the most recent snapshot automatically
2. Resume from where it left off
3. Continue saving frames and snapshots

No need to re-run plate_preparer!

---

## Citation

This work can be cited as follows:

Gribanov, I., Waseda, T., Taylor, R., & Turnbull, I. (2025, July). Application of the Material Point Method for Simulating Sea Ice Breakup. In Proceedings of the 28th International Conference on Port and Ocean Engineering under Arctic Conditions (POAC), St. John’s, Newfoundland and Labrador, Canada.
