# Data Structures: HDF5 Files, Memory Layouts, and Indexing

Comprehensive reference for all file formats, memory organizations, and data access patterns in plateMPM.

---

## 1. HDF5 File Formats

### 1.1 Grid Definition File (grid.h5)

Created by `plate_preparer` during grid initialization.

**Purpose:** Store grid structure, landmask, and initial visualization colors

**Datasets:**

```
grid.h5
├── landmask [GridXTotal, GridYTotal] (UINT8)
│   └─ Values: 255 (water/ice), 0-254 (land)
│
├── color_grid [GridYTotal, GridXTotal, 3] (UINT8)
│   └─ RGB colors from original satellite image, Y-major layout
│
└── (Attributes)
    ├─ GridXTotal (INT32)
    ├─ GridYTotal (INT32)
    ├─ DimensionHorizontal (DOUBLE)
    ├─ CellSize (DOUBLE)
    ├─ ModeledRegionOffsetX (INT32)
    ├─ ModeledRegionOffsetY (INT32)
    └─ InitializationImageSize [2] (INT32)
        └─ [0] = ImageSizeX, [1] = ImageSizeY
```

**Layout Details:**

- **landmask**: Column-major (Fortran order)
  - Index: `idx = j + i * GridYTotal`
  - Range: i ∈ [0, GridXTotal), j ∈ [0, GridYTotal)

- **color_grid**: Interleaved RGB
  - Shape: [GridYTotal, GridXTotal, 3]
  - Access: `color_grid[j * GridXTotal + i][0..2]` for pixel (i,j)
  - Note: Y is first dimension (row-major for image compatibility)

**Example Read (C HDF5 API):**

```c
// Open file and datasets
hid_t file = H5Fopen("grid.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
hid_t landmask_ds = H5Dopen(file, "landmask", H5P_DEFAULT);

// Read attributes
int gx, gy;
hid_t attr_gx = H5Aopen(landmask_ds, "GridXTotal", H5P_DEFAULT);
H5Aread(attr_gx, H5T_NATIVE_INT, &gx);

// Read dataset
std::vector<uint8_t> landmask(gx * gy);
H5Dread(landmask_ds, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, landmask.data());

// Cleanup
H5Aclose(attr_gx);
H5Dclose(landmask_ds);
H5Fclose(file);
```

---

### 1.2 Snapshot File (s00000.h5, s00001.h5, ...)

Stores complete particle and grid state at specific simulation times.

**Purpose:** Checkpoint for resuming simulations, visualization playback, analysis

**Datasets:**

```
s00000.h5
├── pts_data [nPtsArrays, HSSOA_capacity] (FLOAT64)
│   └─ 22 separate arrays flattened into 2D dataset
│
├── grid_data [16, GridXTotal, GridYTotal] (FLOAT64)
│   └─ 16 grid arrays (if saving grid data)
│
└── (Attributes)
    ├─ nPtsInitial (INT32) - Number of valid particles
    ├─ SimulationStep (INT32) - Current step number
    ├─ SimulationTime (DOUBLE) - Elapsed time in seconds
    ├─ ParticleVolume (DOUBLE) - Volume per particle (m³)
    └─ GridXTotal, GridYTotal (INT32)
```

**pts_data Layout:**

22 arrays stored in flattened 2D HDF5 dataset:

```
Shape: [22, capacity]

Array index i corresponds to PtArrIdx enum:
Row 0:  idx_utility_data
Row 1:  idx_integer_cell_idx
Row 2:  idx_integer_point_idx
Row 3:  idx_P (pressure)
Row 4:  idx_Q (deviatoric stress)
Row 5:  idx_Jp_inv (inverse plasticity)
Row 6:  idx_posx
Row 7:  idx_posy
Row 8:  idx_velx
Row 9:  idx_vely
Row 10-13: idx_Fe00, idx_Fe01, idx_Fe10, idx_Fe11 (deformation gradient)
Row 14-16: idx_Fp00, idx_Fp01, idx_Fp10 (plastic deformation, if stored)
Row 17-18: idx_Fp11, idx_damage_parameter
Row 19: idx_thickness (damage/integrity 0.0-1.0)
Row 20: idx_pt_color_R
Row 21: idx_pt_color_G
Row 22: idx_pt_color_B (note: 23 rows total for indices 0-22)
```

**Access Pattern:**

```cpp
// Read snapshot
hid_t file = H5Fopen("s00000.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
hid_t pts_ds = H5Dopen(file, "pts_data", H5P_DEFAULT);

// Get dimensions
hsize_t dims[2];
H5Dget_space(pts_ds);
H5Sget_simple_extent_dims(space, dims, NULL);
int nPtsArrays = dims[0];  // Should be 22
int capacity = dims[1];

// Read all data
std::vector<double> buffer(nPtsArrays * capacity);
H5Dread(pts_ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data());

// Access particle p, property idx
double value = buffer[p + capacity * idx];

// Example: Get position of particle 42
double x = buffer[42 + capacity * 6];  // idx_posx = 6
double y = buffer[42 + capacity * 7];  // idx_posy = 7
```

---

### 1.3 Frame File (f00000.h5, f00001.h5, ...)

Pre-computed grid visualization data for efficient post-processing.

**Purpose:** Compact storage of grid-only data for animation generation, visualization

**Datasets:**

```
f00000.h5
├── grid_arrays [16, GridXTotal, GridYTotal] (FLOAT64, compressed)
│   └─ 16 visualization/state arrays
│
└── (Attributes)
    ├─ SimulationTime (DOUBLE) - Time step for this frame
    ├─ SimulationStep (INT32) - Step number
    ├─ GridXTotal, GridYTotal (INT32)
    └─ CompressionMethod (STRING) - HDF5 compression used
```

**16 Grid Arrays (GridArrIdx):**

```
Index 0:  grid_idx_mass - Total mass at cell
Index 1:  grid_idx_px - Momentum (x-component)
Index 2:  grid_idx_py - Momentum (y-component)
Index 3:  grid_idx_vis_pts_density - Count of points affecting cell
Index 4:  grid_idx_vis_Jpinv - Inverse plastic compressibility
Index 5:  grid_idx_vis_P - Pressure
Index 6:  grid_idx_vis_Q - Deviatoric stress
Index 7:  grid_idx_vis_strain_vonMises - Von Mises strain
Index 8:  grid_idx_vis_damage - Damage indicator
Index 9:  grid_idx_vis_velocity_norm - Velocity magnitude
Index 10-15: Additional state variables or reserved
```

**Layout:**

Column-major (Fortran order) - same as grid.h5

```cpp
// Access grid value at cell (i, j)
int idx = j + i * GridYTotal;  // Column-major

// Access array arr_idx at cell (i, j)
double value = grid_arrays[idx + arr_idx * (GridXTotal * GridYTotal)];
```

**Example Usage (Post-processor):**

```cpp
// Load frame file for visualization
LoadFrameData("output/frames/f00042.h5");
// → Sets host_grid_buffer with 16 arrays
// → Extracts SimulationTime attribute
// → Updates visualization actors

// Access pressure at cell (100, 50)
int gx = prms.GridXTotal;
int gy = prms.GridYTotal;
int idx = 50 + 100 * gy;
double pressure = host_grid_buffer[idx + 5 * (gx * gy)];  // arr_idx 5 = grid_idx_vis_P
```

---

### 1.4 Flow Field File (grid_flow.h5)

Ocean current velocity data for flow-dependent ice dynamics.

**Purpose:** Spatially and temporally varying flow field loaded during simulation

**Datasets:**

```
grid_flow.h5
├── water_current_vx [num_frames, GridXTotal, GridYTotal] (FLOAT32)
│   └─ X-component of current velocity (m/s)
│
├── water_current_vy [num_frames, GridXTotal, GridYTotal] (FLOAT32)
│   └─ Y-component of current velocity (m/s)
│
└── (Attributes on each dataset)
    ├─ time_interval (DOUBLE) - Seconds between frames
    ├─ loop_mode (INT32) - 0: periodic loop, 1: hold last frame
    └─ Compression info (if compressed)
```

**Coordinate System:**

- Same as grid: column-major, i ∈ [0, GridXTotal), j ∈ [0, GridYTotal)
- Physical velocities in m/s (applied to ice dynamics)

**Frame Access Pattern:**

```cpp
// Interpolate velocity at cell (i,j) at simulation time t
double vx = waci.GetInterpolatedValue(i, j, t, &vy);  // Returns vx, outputs vy

// Internal: bilinear interpolation between two frames
double frame_time = time / time_interval;
int frame_idx = (int)floor(frame_time);
double alpha = frame_time - frame_idx;

// Load frames frame_idx and frame_idx+1 if needed
// Interpolate: vx_interp = vx[frame_idx] + alpha * (vx[frame_idx+1] - vx[frame_idx])
```

**Generation Options:**

**Option 1: Constant Flow** (single frame)
```cpp
water_current_vx[1, gx, gy] = constant velocity
time_interval = 0.0
```

**Option 2: Wave Flow** (multiple frames)
```cpp
for frame in 0..num_frames:
    water_current_vx[frame, gx, gy] = wave_amplitude * sin(2π * frame / num_frames)
time_interval = total_simulation_time / num_frames
```

**Option 3: FLUENT Import** (single frame)
```cpp
water_current_vx[1, gx, gy] = rasterized FLUENT velocity
time_interval = 0.0
```

---

## 2. Host-Side Memory Layout

### 2.1 HostSideSOA: Particle Data in Structure-of-Arrays Format

All particle properties stored as separate arrays for GPU memory coalescing.

**Memory Organization:**

```cpp
class HostSideSOA {
    std::vector<double> host_buffer;  // Single contiguous buffer
    int nPtsInitial;                   // Number of valid particles
    int capacity;                      // Total allocated slots (>= nPtsInitial)
    int nPtsArrays = 22;               // Always 22 arrays
};

// Physical memory layout:
// host_buffer[0...capacity-1]              → Array 0 (idx_utility_data)
// host_buffer[capacity...2*capacity-1]     → Array 1 (idx_integer_cell_idx)
// ...
// host_buffer[21*capacity...22*capacity-1] → Array 21 (idx_pt_color_B)
```

**Capacity Management:**

```cpp
capacity = nPtsInitial * (1.0 + extra_space_pts);
// Default extra_space_pts ≈ 0.15 (15% buffer for particle generation)

// Valid particles: indices [0, nPtsInitial)
// Extra space: indices [nPtsInitial, capacity)
```

**Access Pattern (Host):**

```cpp
// Get position of particle p
size_t idx_x = p + capacity * PtArrIdx::idx_posx;
size_t idx_y = p + capacity * PtArrIdx::idx_posy;
double x = host_buffer[idx_x];
double y = host_buffer[idx_y];

// Get pressure of particle p
size_t idx_p = p + capacity * PtArrIdx::idx_P;
double pressure = host_buffer[idx_p];
```

**22 Array Indices (PtArrIdx):**

```cpp
enum PtArrIdx {
    idx_utility_data = 0,          // General-purpose flags
    idx_integer_cell_idx = 1,      // Which grid cell contains particle (GPU use)
    idx_integer_point_idx = 2,     // Point index in partition (GPU use)

    idx_P = 3,                     // Pressure (Pa)
    idx_Q = 4,                     // Deviatoric stress (Pa)
    idx_Jp_inv = 5,                // 1/det(Fp) - inverse plasticity

    idx_posx = 6,                  // Position X (m)
    idx_posy = 7,                  // Position Y (m)

    idx_velx = 8,                  // Velocity X (m/s)
    idx_vely = 9,                  // Velocity Y (m/s)

    idx_Fe00 = 10, idx_Fe01 = 11,  // Elastic deformation gradient F_e (2×2)
    idx_Fe10 = 12, idx_Fe11 = 13,

    idx_Fp00 = 14, idx_Fp01 = 15,  // Plastic deformation gradient F_p (2×2)
    idx_Fp10 = 16, idx_Fp11 = 17,

    idx_reserved_1 = 18,           // Reserved
    idx_thickness = 19,            // Damage: 0.0 (crushed) to 1.0 (intact)

    idx_pt_color_R = 20,           // Color from satellite image
    idx_pt_color_G = 21,
    idx_pt_color_B = 22
};
```

**Physical Properties:**

| Index | Name | Type | Range | Units | Meaning |
|-------|------|------|-------|-------|---------|
| 3 | P | double | -∞ to +∞ | Pa | Pressure (negative = compression) |
| 4 | Q | double | 0 to +∞ | Pa | Deviatoric stress magnitude |
| 5 | Jp_inv | double | 0 to +∞ | 1 | Inverse of plastic volume ratio |
| 6-7 | pos | double | varies | m | Position in domain |
| 8-9 | vel | double | varies | m/s | Velocity |
| 10-13 | Fe | double | varies | 1 | Elastic deformation gradient |
| 14-17 | Fp | double | varies | 1 | Plastic deformation gradient |
| 19 | thickness | double | 0 to 1 | 1 | Damage/integrity: 1=healthy, 0=crushed |
| 20-22 | color | uint8→double | 0 to 255 | RGB | Original satellite image color |

---

### 2.2 Host Grid Buffer

Grid data matching particle state and computational variables.

**Memory Organization:**

```cpp
class HostSideData {
    std::vector<double> host_grid_buffer;  // Size: GridXTotal * GridYTotal * 16
    // Layout: 16 separate arrays, each of size GridXTotal * GridYTotal
};

// Physical layout:
// host_grid_buffer[0..gx*gy-1]              → Array 0 (grid_idx_mass)
// host_grid_buffer[gx*gy..2*gx*gy-1]        → Array 1 (grid_idx_px)
// ...
// host_grid_buffer[15*gx*gy..16*gx*gy-1]    → Array 15
```

**Column-Major Indexing:**

```cpp
// Cell (i, j) → linear index
int idx = j + i * GridYTotal;  // j is fast-varying, i is slow-varying

// Example: Grid with GridXTotal=100, GridYTotal=60
// Cell (0, 0) → idx = 0
// Cell (0, 1) → idx = 1
// Cell (0, 59) → idx = 59
// Cell (1, 0) → idx = 60 (not 100!)
// Cell (1, 59) → idx = 119
```

**Array Indices (GridArrIdx):**

```cpp
enum GridArrIdx {
    grid_idx_mass = 0,                      // Total mass
    grid_idx_px = 1,                        // Momentum X
    grid_idx_py = 2,                        // Momentum Y
    grid_idx_vis_pts_density = 3,           // Count of points
    grid_idx_vis_Jpinv = 4,                 // Plastic invariant
    grid_idx_vis_P = 5,                     // Pressure
    grid_idx_vis_Q = 6,                     // Deviatoric stress
    grid_idx_vis_strain_vonMises = 7,       // Von Mises strain
    grid_idx_vis_damage = 8,                // Damage measure
    grid_idx_vis_velocity_norm = 9,         // Velocity magnitude
    grid_idx_reserved_10 = 10,
    grid_idx_reserved_11 = 11,
    grid_idx_reserved_12 = 12,
    grid_idx_reserved_13 = 13,
    grid_idx_reserved_14 = 14,
    grid_idx_reserved_15 = 15
};
```

**Access Pattern:**

```cpp
// Get mass at cell (i, j)
int gx = prms.GridXTotal;
int gy = prms.GridYTotal;
int idx = j + i * gy;
double mass = host_grid_buffer[idx + grid_idx_mass * (gx * gy)];

// Get pressure at cell (i, j)
double pressure = host_grid_buffer[idx + grid_idx_vis_P * (gx * gy)];

// Iterate all cells
for (int i = 0; i < gx; ++i) {
    for (int j = 0; j < gy; ++j) {
        int idx = j + i * gy;
        double mass = host_grid_buffer[idx];
        double px = host_grid_buffer[idx + gx * gy];
        double py = host_grid_buffer[idx + 2 * gx * gy];
    }
}
```

---

### 2.3 Landmask Buffer

Region indicator showing which grid cells are active.

**Organization:**

```cpp
std::vector<uint8_t> landmask_buffer;  // Size: GridXTotal * GridYTotal

// Column-major indexing, same as grid
int idx = j + i * GridYTotal;
uint8_t value = landmask_buffer[idx];
```

**Values:**

- `255`: Water/ice region (active, solved)
- `0-254`: Land region (inactive, not solved)

---

### 2.4 Original Image Colors

RGB colors from satellite image for visualization.

**Organization:**

```cpp
std::vector<uint8_t> original_image_colors_rgb;  // Size: ImageSizeX * ImageSizeY * 3

// Row-major indexing (Y-first, matching image convention)
int idx = (x + y * ImageSizeX) * 3;
uint8_t r = original_image_colors_rgb[idx + 0];
uint8_t g = original_image_colors_rgb[idx + 1];
uint8_t b = original_image_colors_rgb[idx + 2];
```

**Note:** ImageSize is original satellite image dimensions, not modeled grid dimensions

---

## 3. GPU Memory Layout

### 3.1 Device-Side Arrays (GPU_Partition)

Mirror of HostSideSOA on GPU with additional device-specific structures.

**GPU Allocation:**

```cuda
// Allocated in GPU_Partition::initialize()
double* d_particle_data[22];     // 22 separate arrays, each of size capacity
double* d_grid_data;             // Grid arrays, size 16 * gx_local * gy_local
double* d_grid_halo;             // Halo cells for inter-GPU communication
uint8_t* d_landmask;             // Landmask on GPU
```

**Memory Organization:**

Same SOA structure as host:
- Array i starts at offset `i * capacity * sizeof(double)`
- Particle p, property i stored at offset `p + i * capacity`

**Pinned Host Memory:**

Used for efficient PCIe transfers:

```cpp
// Allocate pinned memory for transfers
cudaMallocHost((void**)&pinned_buffer, num_bytes);
// Copy from regular host memory
memcpy(pinned_buffer, host_buffer, num_bytes);
// Async DMA to GPU
cudaMemcpyAsync(d_buffer, pinned_buffer, num_bytes,
                cudaMemcpyHostToDevice, stream);
```

---

### 3.2 Constant Memory (GPU)

SimParams copied to CUDA constant memory for efficient kernel access.

**Size:** ~2-4 KB (very limited)

**Contents:**

```cuda
__constant__ SimParams_device d_prms;

// Accessible in all kernels via d_prms.GridXTotal, d_prms.CellSize, etc.
// Much faster than reading from global memory (register vs DRAM)
```

---

## 4. Data Access Patterns and Idioms

### 4.1 Accessing Particle Data (Host)

```cpp
// Pattern 1: Direct indexing
size_t linear_idx = p + capacity * (size_t)PtArrIdx::idx_posx;
double x = hsd.hssoa.host_buffer[linear_idx];

// Pattern 2: Helper function (if available)
double x = hsd.hssoa.GetDouble(p, PtArrIdx::idx_posx);

// Pattern 3: Get all properties of one particle
struct Particle {
    double x, y, vx, vy, p, q;
};
Particle GetParticle(int p, const HostSideSOA& soa) {
    int cap = soa.capacity;
    return {
        soa.host_buffer[p + cap * idx_posx],
        soa.host_buffer[p + cap * idx_posy],
        soa.host_buffer[p + cap * idx_velx],
        soa.host_buffer[p + cap * idx_vely],
        soa.host_buffer[p + cap * idx_P],
        soa.host_buffer[p + cap * idx_Q]
    };
}
```

### 4.2 Accessing Grid Data (Host)

```cpp
// Get grid dimensions
int gx = prms.GridXTotal;
int gy = prms.GridYTotal;

// Access single cell value
auto GetGridValue = [&](int i, int j, int arr_idx) -> double {
    int idx = j + i * gy;
    return host_grid_buffer[idx + arr_idx * (gx * gy)];
};

double mass = GetGridValue(i, j, grid_idx_mass);
double pressure = GetGridValue(i, j, grid_idx_vis_P);
```

### 4.3 Iterating Particles

```cpp
// Iterate all valid particles
for (int p = 0; p < hsd.hssoa.nPtsInitial; ++p) {
    double x = hsd.hssoa.host_buffer[p + hsd.hssoa.capacity * idx_posx];
    double y = hsd.hssoa.host_buffer[p + hsd.hssoa.capacity * idx_posy];
    // Process particle p
}
```

### 4.4 Iterating Grid Cells

```cpp
// Iterate all grid cells (column-major)
int gx = prms.GridXTotal;
int gy = prms.GridYTotal;

for (int i = 0; i < gx; ++i) {
    for (int j = 0; j < gy; ++j) {
        int idx = j + i * gy;
        double mass = host_grid_buffer[idx];
        // Process cell (i, j)
    }
}

// Or iterate row-major (less cache-friendly, not recommended)
for (int j = 0; j < gy; ++j) {
    for (int i = 0; i < gx; ++i) {
        int idx = j + i * gy;  // Still column-major indexing!
        // Process cell (i, j)
    }
}
```

---

## 5. Coordinate Systems and Transformations

### 5.1 Image vs. Physical Coordinates

**Image Space:**
- Origin: top-left (0, 0)
- X-axis: points right
- Y-axis: points down
- Units: pixels (0 to ImageSizeX-1, 0 to ImageSizeY-1)

**Physical Space:**
- Origin: bottom-left
- X-axis: points right
- Y-axis: points up
- Units: meters (0 to DimensionHorizontal, 0 to DimensionVertical)

**Conversion:**

```cpp
// Image (px, py) → Physical (mx, my)
double mx = (px - OffsetX) * CellSize;
double my = (ImageSizeY - py - 1 - OffsetY) * CellSize;  // Y-flip

// Physical (mx, my) → Image (px, py)
int px = (int)(mx / CellSize) + OffsetX;
int py = ImageSizeY - (int)(my / CellSize) - 1 - OffsetY;  // Y-flip
```

### 5.2 Grid Cell Coordinates

**Cell (i, j):**
- i ∈ [0, GridXTotal)
- j ∈ [0, GridYTotal)

**Normalized Cell Coordinates:**
- Position within cell: (u, v) ∈ [0, 1)²
- Physical coordinates:
  ```cpp
  double x = (i + u) * CellSize + (OffsetX * CellSize);
  double y = (j + v) * CellSize + (OffsetY * CellSize);
  ```

---

## 6. Data Type Specifications

### 6.1 HDF5 Type Mappings

| C++ Type | HDF5 Type | Size | Notes |
|----------|-----------|------|-------|
| double | H5T_NATIVE_DOUBLE | 8 bytes | IEEE 754 double precision |
| float | H5T_NATIVE_FLOAT | 4 bytes | IEEE 754 single precision |
| int | H5T_NATIVE_INT | 4 bytes | 32-bit signed integer |
| uint8_t | H5T_NATIVE_UINT8 | 1 byte | 8-bit unsigned (0-255) |

### 6.2 Precision Notes

- **Particles**: Stored as `double` (8 bytes/value)
- **Grid**: Stored as `double` (high precision for accumulation)
- **Flow field**: Stored as `float` (4 bytes) - sufficient for velocity
- **Colors/masks**: Stored as `uint8_t` (0-255 range)

---

## 7. File I/O Patterns

### 7.1 Reading Grid Data

```cpp
// Standard HDF5 read pattern
hid_t file = H5Fopen("grid.h5", H5F_ACC_RDONLY, H5P_DEFAULT);

// Read landmask
hid_t landmask_ds = H5Dopen(file, "landmask", H5P_DEFAULT);
std::vector<uint8_t> landmask(gx * gy);
H5Dread(landmask_ds, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL,
        H5P_DEFAULT, landmask.data());
H5Dclose(landmask_ds);

// Read color grid
hid_t color_ds = H5Dopen(file, "color_grid", H5P_DEFAULT);
std::vector<uint8_t> colors(h * w * 3);
H5Dread(color_ds, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL,
        H5P_DEFAULT, colors.data());
H5Dclose(color_ds);

H5Fclose(file);
```

### 7.2 Reading Snapshot Data

```cpp
// Read particle data from snapshot
hid_t file = H5Fopen("s00000.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
hid_t pts_ds = H5Dopen(file, "pts_data", H5P_DEFAULT);

// Get dimensions
hsize_t dims[2];
H5Sget_simple_extent_dims(H5Dget_space(pts_ds), dims, NULL);

std::vector<double> buffer(dims[0] * dims[1]);
H5Dread(pts_ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
        H5P_DEFAULT, buffer.data());

// Read attributes
int nPtsInitial;
hid_t attr = H5Aopen(pts_ds, "nPtsInitial", H5P_DEFAULT);
H5Aread(attr, H5T_NATIVE_INT, &nPtsInitial);
H5Aclose(attr);

H5Dclose(pts_ds);
H5Fclose(file);
```

### 7.3 Writing Snapshot Data

```cpp
// Create snapshot file
hid_t file = H5Fcreate("s00001.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

// Create dataset with compression
hsize_t dims[2] = {22, capacity};
hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
H5Pset_deflate(dcpl, 4);  // Compression level 4

hid_t pts_space = H5Screate_simple(2, dims, NULL);
hid_t pts_ds = H5Dcreate(file, "pts_data", H5T_NATIVE_DOUBLE,
                         pts_space, H5P_DEFAULT, dcpl, H5P_DEFAULT);

// Write data
H5Dwrite(pts_ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
         H5P_DEFAULT, buffer.data());

// Write attributes
int nPtsInitial = 10000;
hid_t attr = H5Acreate(pts_ds, "nPtsInitial", H5T_NATIVE_INT,
                       H5Screate(H5S_SCALAR), H5P_DEFAULT);
H5Awrite(attr, H5T_NATIVE_INT, &nPtsInitial);
H5Aclose(attr);

// Cleanup
H5Pclose(dcpl);
H5Sclose(pts_space);
H5Dclose(pts_ds);
H5Fclose(file);
```

---

## 8. Performance Optimization Notes

### 8.1 Memory Access Patterns

**Good patterns:**
- Sequential access in fast-varying dimension (j for grid, p for particles)
- Column-major iteration (i loop outer, j loop inner)
- Coalesced GPU memory access (warps read adjacent indices)

**Bad patterns:**
- Random access across arrays (p jumps around)
- Row-major iteration of column-major arrays
- Unaligned data reads (indices not multiple of cache line)

### 8.2 HDF5 Compression

- **Type**: DEFLATE (gzip-compatible)
- **Level**: 4 (balance speed/ratio)
- **Chunking**: Automatic (HDF5 chooses)
- **Benefit**: Reduces file size by ~70-80% (important for frame storage)

### 8.3 Data Transfer Optimization

- **Pinned memory**: ~2-3× faster than pageable for large transfers
- **Async transfers**: Overlaps DMA with computation
- **Batch transfers**: Fewer cudaMemcpy calls, larger transfers
- **PCIe bandwidth**: ~11 GB/s per direction (PCIe 3.0 ×16)

---

## 9. Data Consistency and Synchronization

### 9.1 Host-Device Synchronization

**Point-to-Grid (P2G):**
```
Host buffer → GPU buffer (copy at initialization)
GPU computes P2G kernel
```

**Grid-to-Point (G2P):**
```
GPU computes G2P kernel
GPU → Host buffer (copy when saving frames)
```

**Update sequence:**
1. Transfer host_buffer to GPU (once at start)
2. Run P2G, update grid, G2P on GPU (every step)
3. Transfer grid_buffer back to host (when saving)
4. Don't transfer particles back (remain on GPU throughout)

### 9.2 Multi-GPU Synchronization

**Halo exchange:**
```
Partition 0           Halo           Partition 1
[grid]  ←→ [copied]  ←→  [grid]
         ↓ copy to GPU 1
         ← copy from GPU 0 halo
```

**Sequence:**
1. Each partition computes P2G in local region
2. Halo cells exchanged between neighbors
3. Grid update computed
4. G2P uses potentially halo-contributed grid values

---

## 10. Debugging Tips

### 10.1 Data Validation Checklist

- [ ] Grid dimensions match across all files (gx, gy)
- [ ] Particle positions within bounds: 0 ≤ x,y < grid dimensions (m)
- [ ] Particle velocities reasonable: |v| < 1 m/s (typically)
- [ ] Pressure reasonable: |P| < 10⁵ Pa (ice yield stress ~10⁶ Pa)
- [ ] Deformation gradient Fe close to identity (elastic only)
- [ ] Thickness ∈ [0, 1] (damage measure)
- [ ] Color RGB ∈ [0, 255] or [0, 1] depending on storage
- [ ] Landmask ∈ {0-254, 255} (255 = water/ice, 0-254 = land)
- [ ] Grid mass > 0 (accumulated from particles)

### 10.2 Common Issues

**Issue:** Particles outside domain after first step
- **Cause**: CellSize or OffsetX/Y incorrect
- **Check**: ComputeHelperVariables() correctly calculated ParticleVolume

**Issue:** Grid values all zero
- **Cause**: P2G kernel not running or particles not on GPU
- **Check**: host_buffer was transferred before P2G step

**Issue:** Landmask shows wrong region
- **Cause**: Coordinate system mismatch (image vs physical)
- **Check**: Y-flip applied correctly in plate_preparer

---

This reference covers all major data structures and layouts. For specific API details, see CLASS_REFERENCE.md. For architecture overview, see ARCHITECTURE.md.
