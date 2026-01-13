# Class Reference: plateMPM API

API documentation for key classes in the plateMPM simulation framework.

---

## HostSideData

**Location:** `simulation/data_manager/host_side_data.h/cpp`

**Responsibility:** Manage all host-side data including grid, particles, parameters, and I/O.

### Public Members

```cpp
SimParams prms;                              // All simulation parameters
HostSideSOA hssoa;                          // Particle data (Structure-of-Arrays)
std::vector<double> host_grid_buffer;       // Grid data [gx*gy*16]
std::vector<uint8_t> landmask_buffer;       // Region indicator [gx*gy]
std::vector<uint8_t> original_image_colors_rgb;  // Background colors [h*w*3]
WindAndCurrentInterpolator waci;            // Flow field data
std::string output_directory;               // Where to save output
std::string data_directory;                 // Where grid/flow data located
```

### Key Methods

#### Initialization

**`void LoadGridDataFromFile(const std::string& gridFilePath)`**
- Loads grid structure from grid.h5
- Reads grid dimensions, cell size, landmask, colors
- Allocates `host_grid_buffer`
- **Called by:** Model::LoadParameterFile

**`void ReadPointsFromSnapshot(std::string fileNameSnapshotHDF5)`**
- Loads particle data from snapshot HDF5 file
- Restores particle positions, velocities, stress state
- Restores simulation step and time
- Allocates `hssoa` with proper capacity
- **Called by:** Model::LoadParameterFile

**`void AllocatePointArrays()`**
- Allocates 22 separate arrays for particle properties
- Capacity = nPtsInitial × (1 + extra_space_pts)
- Must be called after ReadPointsFromSnapshot
- **Called by:** Model::LoadParameterFile

#### Grid Management

**`void LoadGridDataFromFile(const std::string& gridFilePath)`**
- Reads grid metadata (dimensions, cell size, offsets)
- Loads landmask (modeled region indicator)
- Allocates and initializes grid buffer
- Called during initialization

#### Snapshot I/O

**`void SaveSnapshot(int frameNumber, double time, bool save_disabled_points)`**
- Saves complete particle state to HDF5
- Includes all properties needed to resume
- Used for checkpoints and resuming
- **Format:** `output/snapshots/s{frameNumber:05d}.h5`

**`void LoadFrameData(const std::string& framePath)`**
- Loads pre-rendered grid data from frame file
- Used by post-processor for visualization
- **Format:** `frames/f{frameNumber:05d}.h5`

#### Helper Methods

**`void ComputeHelperVariables()`**
- Calculates particle volume from grid and particle count
- Calculates particle mass (volume × density)
- Must be called before GPU transfer
- **Used by:** GPU initialization

**`void RemoveDisabledAndSort()`**
- Removes disabled particles from active buffer
- Sorts remaining particles for cache efficiency
- Maintains particle-to-partition mapping

---

## SimParams

**Location:** `simulation/parameters_sim.h/cpp`

**Responsibility:** Store and parse all simulation parameters from JSON configuration.

### Public Methods

**`std::map<std::string, std::string> ParseFile(const std::string& fileName)`**
- Parses JSON configuration file using RapidJSON
- Returns map with all JSON field values
- Validates required fields exist
- Throws exception on missing critical parameters
- **Returns:** Map with keys: "GridData", "CurrentVelocityData", "SimulationTitle", etc.

**`void Printout()`**
- Prints all parameters to logging system
- Useful for verification before simulation

### Key Parameters

**Domain:**
- `int GridXTotal, GridYTotal` - Grid dimensions
- `double DimensionHorizontal` - Physical domain width
- `double CellSize` - Physical grid cell size
- `int ModeledRegionOffsetX/Y` - Grid offset in image space

**Time Integration:**
- `double InitialTimeStep` - Timestep size (constant)
- `double SimulationTime` - Current simulation time
- `double SimulationEndTime` - When to stop
- `int SimulationStep` - Current step number

**Ice Rheology:**
- `double IceCompressiveStrength` - Max pressure
- `double IceTensileStrength` - Min (tension) pressure
- `double IceShearStrength` - Max deviatoric stress

**Plasticity:**
- `double DP_phi` - Drucker-Prager friction angle
- `double DP_threshold_p` - Elastic-plastic transition

**Material:**
- `double IceDensity` - Mass per volume
- `double InitializationImageSizeX/Y` - Original image dimensions
- `int PointsPerCell` - Target particles per grid cell

---

## HostSideSOA

**Location:** `simulation/hssoa/host_side_soa.h/cpp`

**Responsibility:** Manage particle data in Structure-of-Arrays format for GPU efficiency.

### Public Members

```cpp
std::vector<double> host_buffer;    // Single buffer: [nPtsArrays * capacity]
int nPtsInitial;                    // Number of valid particles
int capacity;                       // Total capacity (nPtsInitial + extra space)
int nPtsArrays;                     // Always 22 for current implementation
```

### Access Pattern

Particles accessed via `PtArrIdx` enum:

```cpp
// Position of particle p in array i:
size_t idx = p + capacity * (size_t)PtArrIdx::idx_posx;
double x = host_buffer[idx];

// Alternative: direct indexing
double x = GetDouble(p, PtArrIdx::idx_posx);
```

### Array Indices (22 total)

**Position and Velocity:**
- `idx_posx, idx_posy` - Position (2D)
- `idx_velx, idx_vely` - Velocity (2D)

**Deformation Gradient:**
- `idx_Fe00, idx_Fe01, idx_Fe10, idx_Fe11` - Elastic part (2×2)

**Stress:**
- `idx_P` - Pressure
- `idx_Q` - Deviatoric stress

**State Variables:**
- `idx_Jp_inv` - Inverse plastic compressibility
- `idx_thickness` - Damage (0.0 = crushed, 1.0 = intact)
- `idx_pt_color_R, idx_pt_color_G, idx_pt_color_B` - Color

**Utility:**
- `idx_integer_cell_idx` - Cell index for GPU
- `idx_utility_data` - General purpose flags

### Key Methods

**`void Allocate(int capacity)`**
- Allocates host_buffer with given capacity
- nPtsInitial must be set before calling
- Extra space reserved for particle generation

**`void convertToIntegerCellFormat(double cellsize_inv)`**
- Converts normalized coordinates to cell indices
- Multiplies positions by cell size inverse
- Used before GPU transfer

---

## Model

**Location:** `simulation/model.h/cpp`

**Responsibility:** Orchestrate simulation: load data, prepare GPU, run steps.

### Public Methods

**`void LoadParameterFile(std::string fileName, std::string resumeSnapshotFileName = "")`**
- Load configuration and initialize simulation
- Handles both fresh start and resume cases
- Initializes GPU
- Loads flow field if specified
- **Throws:** Exception on missing files or invalid config

**`void Prepare()`**
- Prepare GPU for simulation step
- Update GPU constants
- Load initial flow data
- Called before first step

**`bool Step()`**
- Execute one simulation timestep
- Returns true if simulation should continue
- Transfers updated grid to host for frame saves
- **Returns:** false when SimulationTime >= SimulationEndTime

**`void Printout()`**
- Print memory allocation and statistics

### Public Members

```cpp
HostSideData sim_data;          // All host-side data
GPU_Implementation5 gpu;        // GPU management and execution
SimParams prms;                 // Quick reference to parameters (copy)
```

### Workflow

```cpp
Model model;
model.LoadParameterFile(jsonPath);
model.Prepare();

while (model.Step()) {
    // Each step executes GPU kernels and saves data
}
```

---

## GPU_Implementation5

**Location:** `simulation/gpu_implementation5.h/cpp`

**Responsibility:** Manage GPU devices and execute simulation kernels.

### Public Methods

**`void initialize()`**
- Query CUDA devices
- Create GPU_Partition for each device
- Enable peer-to-peer access
- **Called by:** SplitIntoPartitionsAndTransferToDevice

**`void split_hssoa_into_partitions()`**
- Partition particles by X-coordinate
- Assign partition IDs
- Create index mapping for transfers
- **Called by:** SplitIntoPartitionsAndTransferToDevice

**`void SplitIntoPartitionsAndTransferToDevice()`**
- Complete GPU initialization sequence:
  1. ComputeHelperVariables()
  2. initialize()
  3. split_hssoa_into_partitions()
  4. allocate_device_arrays()
  5. transfer_to_device()
- **Called by:** Model::LoadParameterFile

**`bool Step()`**
- Execute one timestep on all GPU partitions
- P2G (point-to-grid), grid update, G2P, halo exchange
- **Returns:** false if simulation error

**`void transfer_from_device()`**
- Copy updated grid back to host
- Used when saving frame data

**`void update_constants()`**
- Transfer SimParams to GPU constant memory
- Called before simulation start

**`void transfer_wind_and_current_data_to_device()`**
- Copy flow field buffers to GPU
- Called when flow data updated

### Public Members

```cpp
std::vector<GPU_Partition> partitions;      // One per GPU device
HostSideData &hsd;                          // Reference to host data
```

---

## GPU_Partition

**Location:** `simulation/gpu_partition.h/cpp`

**Responsibility:** Manage computation on a single GPU device.

### Key Methods

**`void initialize()`**
- Setup CUDA streams and events
- Allocate on-device arrays
- Initialize PartitionParams structure

**`void transfer_to_device()`**
- Copy particle and grid data to GPU
- Uses pinned host memory for efficient DMA

**`void transfer_from_device()`**
- Copy updated grid back to host

**`bool Step()`**
- Execute kernels on this partition:
  1. Reset grid (set to zero)
  2. P2G kernel (points to grid)
  3. Grid update kernel (forces + constitutive)
  4. G2P kernel (grid to points)
  5. Halo exchange with neighbors
  6. Update particle velocity field
- **Returns:** false on CUDA error

### Members

```cpp
int Device;                 // CUDA device ID
HostSideData &hsd;         // Reference to host data
PartitionParams pparams;   // Partition metadata
```

---

## WindAndCurrentInterpolator

**Location:** `simulation/data_manager/windandcurrentinterpolator.h/cpp`

**Responsibility:** Load, manage, and interpolate ocean current and wind velocity fields. Uses a 3-frame ring buffer with asynchronous preloading for performance.

### Public Methods

**`void SetHDF5Path(const std::string& path)`**
- Load HDF5 file path and metadata
- Read dimensions: (num_frames, gx, gy)
- **Called by:** Model::LoadParameterFile

**`std::pair<bool, bool> SetTime(double time)`**
- Load frames needed for given simulation time
- Performs temporal interpolation between frames
- Manages 3-slot ring buffer to minimize reloads
- Asynchronously preloads the next frame (n+2) if sequential access detected
- **Returns:** `{ocean_changed, wind_changed}` pair
- **Called by:** Model::Prepare, Model::Step

**`void GetInterpolatedValue(int i, int j, double time, double &vx, double &vy)`**
- Get interpolated velocity at grid cell (i,j)
- Linear interpolation between frames
- Called by GPU kernels (via host wrapping if needed, though usually GPU accesses directly)

**`float* GetOceanDataPointer(int frameIdx, int component)`**
- Direct access to internal frame buffers for GPU transfer
- `component`: 0 for U (vx), 1 for V (vy)
- Returns pointer to the requested frame's data in the ring buffer

**`float* GetWindDataPointer(int frameIdx, int component)`**
- Direct access to internal wind frame buffers
- `component`: 0 for U (vx), 1 for V (vy)

### features

- **3-Frame Ring Buffer:** Keeps frames n, n+1, and n+2 resident when possible.
- **Async Preloading:** Uses `std::async` and `std::future` to load upcoming frames in the background during the current step's computation.
- **Smart Reuse:** `SetTime` intelligently maps logical frames to physical slots to avoid unnecessary copying.

### HDF5 Format Expected

```
water_current_vx[num_frames, gx, gy]  (FLOAT32)
  Attributes:
  - time_interval (DOUBLE)
  - loop_mode (INT) - 0=periodic, 1=hold_last

water_current_vy[num_frames, gx, gy]  (FLOAT32)
  (same attributes)
```

---

## VisualRepresentation

**Location:** `gui/vtk/visual_representation.h/cpp`

**Responsibility:** Manage VTK visualization actors and color mapping.

### Public Members

```cpp
vtkNew<vtkTextActor> actorText;         // Time display
vtkNew<vtkActor> raster_actor;          // Grid visualization
vtkNew<vtkActor> actor_region_boundary; // Region boundary outline
vtkNew<vtkScalarBarActor> scalarBar;    // Color scale
vtkNew<vtkActor> actor_debug_grid;      // Debug grid overlay

enum VisOpt {
    grid_colors, grid_mass, grid_Jpinv, grid_P, grid_Q,
    grid_vnorm, str_vonMises, grid_ridges
};
VisOpt VisualizingVariable;

double ranges[8];                       // Color scale ranges per mode
double transparency_coeffs[8];          // Transparency per mode
```

### Key Methods

**`void ChangeVisualizationOption(VisOpt option)`**
- Switch visualization mode
- Updates actor colors and scalar bar range
- **Called by:** GUI slider/dropdown

**`void SynchronizeTopology()`**
- Update actors to reflect current grid data
- Called after loading frame data

**`void UpdateTimeText()`**
- Update time display to HH:MM:SS format
- Uses `simulationTime` member variable
- **Called by:** Post-processor when frame loaded

---

## Key Data Structures

### SimParams::PtArrIdx (22 particle arrays)

```cpp
enum PtArrIdx {
    // Utility
    idx_utility_data = 0,
    idx_integer_cell_idx = 1,
    idx_integer_point_idx = 2,

    // Stress
    idx_P = 3,
    idx_Q = 4,
    idx_Jp_inv = 5,

    // Position
    idx_posx = 6,
    idx_posy = 7,

    // Velocity
    idx_velx = 8,
    idx_vely = 9,

    // Elastic deformation gradient (2x2)
    idx_Fe00 = 10, idx_Fe01 = 11,
    idx_Fe10 = 12, idx_Fe11 = 13,

    // ... (more arrays)

    idx_thickness = 19,
    idx_pt_color_R = 20,
    idx_pt_color_G = 21,
    idx_pt_color_B = 22,
};
```

### SimParams::GridArrIdx (16 grid arrays per cell)

```cpp
enum GridArrIdx {
    grid_idx_mass = 0,
    grid_idx_px = 1,
    grid_idx_py = 2,
    grid_idx_vis_pts_density = 3,
    grid_idx_vis_Jpinv = 4,
    grid_idx_vis_P = 5,
    grid_idx_vis_Q = 6,
    grid_idx_vis_strain_vonMises = 7,
    // ... (8 more)
};
```

---

## Common Patterns

### Accessing Particle Data

```cpp
// Get position of particle p
double x = hsd.hssoa.host_buffer[p + hsd.hssoa.capacity * (size_t)PtArrIdx::idx_posx];
double y = hsd.hssoa.host_buffer[p + hsd.hssoa.capacity * (size_t)PtArrIdx::idx_posy];

// Get pressure of particle p
double pressure = hsd.hssoa.host_buffer[p + hsd.hssoa.capacity * (size_t)PtArrIdx::idx_P];
```

### Accessing Grid Data

```cpp
// Get grid cell (i, j)
int idx = j + (size_t)i * prms.GridYTotal;  // Column-major

// Get mass at cell (i, j)
double mass = hsd.host_grid_buffer[idx + (size_t)SimParams::grid_idx_mass * gx * gy];
```

### Iterating Particles

```cpp
for (int p = 0; p < hsd.hssoa.nPtsInitial; p++) {
    double x = hsd.hssoa.host_buffer[p + capacity * idx_posx];
    double y = hsd.hssoa.host_buffer[p + capacity * idx_posy];
    // ... process particle
}
```

---

## Initialization Checklist

When adding features or modifying data structures:

1. Update `SimParams` (parameters_sim.h/cpp)
2. Add storage in `HostSideData` (host_side_data.h/cpp)
3. Allocate in `HostSideSOA` if particle data (host_side_soa.cpp)
4. Transfer to GPU in `GPU_Partition::transfer_to_device()`
5. Update GPU kernel code to use new data
6. Update visualization if data should be rendered

This reference covers the main API surface. See source code for additional methods and helpers.
