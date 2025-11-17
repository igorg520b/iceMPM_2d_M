# Executive Summary: Simulation Code Structure

## Overview
This document provides a structured analysis of the plateMPM simulation codebase, understanding how GPU_Implementation5 and HostSideData are used, where data (grid, flow, PNG images) is loaded, and how initialization works.

## Key Findings

### 1. Current System Status
**The codebase is 95% complete but initialization is disabled.**

- PNG-based initialization pipeline: FULLY IMPLEMENTED
- Snapshot saving/loading: FULLY IMPLEMENTED  
- GPU initialization: FULLY IMPLEMENTED
- Wind/current data loading: FULLY IMPLEMENTED
- **Integration in Model::LoadParameterFile(): COMMENTED OUT**

### 2. The Critical Missing Link

`Model::LoadParameterFile()` (simulation/model.cpp lines 157-196) contains all initialization calls commented out:
```cpp
// CURRENTLY COMMENTED OUT:
// snapshot.PrepareGrid(additionalFiles["InputPNG"], additionalFiles["InputMap"]);
// if(resumeSnapshotFileName.empty())
// {
//     snapshot.PopulatePoints(additionalFiles["InputMap"]);
// }
// else
// {
//     snapshot.ReadPointsFromSnapshot(resumeSnapshotFileName);
// }
// gpu.SplitIntoPartitionsAndTransferToDevice();
// if(additionalFiles.count("InputFlowVelocity"))
// {
//     prms.UseCurrentData = true;
//     wac_interpolator.OpenCustomHDF5(additionalFiles["InputFlowVelocity"]);
// }
```

This is the bottleneck preventing the system from functioning.

### 3. Data Flow Understanding

#### Where Grid Data (grid.h5) is Loaded/Created
**Created by:**
- `HostSideData::PrepareGrid()` (lines 224-364)
  - Reads PNG landmask and color images
  - Extracts water region bounding box
  - Calculates cell size from DimensionHorizontal parameter
  - Saves metadata to `output/{SimulationTitle}/grid.h5`

**Accessed by:**
- At resume: Grid metadata loaded from existing grid.h5
- At simulation run: Used only for validation/reference

**Content:**
- landmask: 2D array marking water/land regions
- color_grid: 3D array [Height][Width][3] for background coloring
- Attributes: grid dimensions, offsets, cell size, physical dimension

#### Where Flow Data (InputFlowVelocity) is Loaded
**Should be loaded by:**
- `Model::LoadParameterFile()` → calls `wac_interpolator.SetHDF5Path()`
- **Currently: NEVER CALLED** (commented out lines 180-184)

**Process:**
1. SetHDF5Path() stores HDF5 file path
2. LoadHDF5Metadata() reads dimensions and time interval
3. SetTime() loads appropriate frame pair from disk
4. GetInterpolatedValue() returns interpolated velocity

**Format Expected:**
- `water_current_vx[num_frames, gx, gy]` and `water_current_vy[num_frames, gx, gy]`
- Attributes: `time_interval` (seconds between frames), `loop_mode` (0=periodic, 1=hold)

#### Where PNG Initialization Happens
**Process:**
1. `Model::LoadParameterFile()` should call `sim_data.PrepareGridAndPoints()`
2. **Currently: NEVER CALLED** (commented out)

**What PrepareGridAndPoints() Does:**
- Loads 4 PNG files (landmask, color, ice mask, crushed mask)
- Validates dimensions match
- Flips images from PNG origin (top-left) to simulation origin (bottom-left)
- Calls PrepareGrid() → creates grid.h5
- Calls PopulatePoints() → generates particles via Poisson disk sampling
- Saves initial snapshot s00000.h5
- Allocates all host-side buffers

#### How GPU Initialization Happens
**Sequence (should be in LoadParameterFile()):**
1. `gpu.initialize()` - Query CUDA devices, create GPU_Partition objects
2. `gpu.split_hssoa_into_partitions()` - Divide points by X-coordinate
3. `gpu.transfer_to_device()` - Copy HostSideSOA and grid to GPU memory

**Currently:** Called implicitly somewhere (or never), needs to be explicit

### 4. GPU_Implementation5 and HostSideData Usage

#### GPU_Implementation5 Class
**Purpose:** Manages GPU computation and data transfers

**Key Methods:**
- `initialize()` - Setup CUDA devices and partitions
- `split_hssoa_into_partitions()` - Partition data across GPUs
- `transfer_to_device()` - Copy host data to GPU
- `transfer_from_device()` - Copy GPU results back to host
- `p2g()`, `update_nodes()`, `g2p()` - Simulation kernels
- `point_transfer()` - Multi-GPU point migration

**Member Variables:**
```cpp
HostSideData &hsd;              // Reference to host-side data
vector<GPU_Partition> partitions; // One per GPU device
```

#### HostSideData Class
**Purpose:** Manages all host-side simulation data

**Key Methods:**
- `PrepareGridAndPoints()` - Create simulation from PNG images
- `ReadPointsFromSnapshot()` - Load from HDF5 snapshot
- `SaveSnapshot()` - Write particle data to HDF5
- `SaveFrame()` - Write visualization frame to HDF5
- `AllocateGridArrays()` - Allocate grid buffers
- `AllocatePointArrays()` - Allocate point buffers

**Member Variables:**
```cpp
SimParams prms;                           // Simulation parameters
HostSideSOA hssoa;                       // Point data (SOA format)
vector<double> host_grid_buffer;         // Grid data
vector<uint8_t> landmask_buffer;         // Water/land mask
vector<uint8_t> original_image_colors_rgb; // Background colors
WindAndCurrentInterpolator waci;         // Flow field interpolation
```

### 5. Parameter Usage

**Parsed from JSON by `SimParams::ParseFile()`:**
```cpp
InputPNG              → File path to landmask PNG
InputMap              → File path to grid.h5 (or template)
InputFlowVelocity     → File path to water current HDF5
UseCurrentData        → Boolean flag to enable current
DimensionHorizontal   → Physical domain size in meters
```

**Currently Used:** None of these (LoadParameterFile() commented out)

**What Should Happen:**
1. InputPNG drives PNG loading via PrepareGridAndPoints()
2. InputFlowVelocity path passed to wac_interpolator.SetHDF5Path()
3. UseCurrentData controls whether wind/current is applied in simulation
4. DimensionHorizontal used to calculate cell size

### 6. Memory Allocation Sequence

```
Grid Buffers:
├─ landmask_buffer[GridX × GridY] (1 byte each)
├─ original_image_colors_rgb[3 × ImgWidth × ImgHeight] (1 byte each)
└─ host_grid_buffer[GridX × GridY × 16] (8 bytes each, DOUBLE)

Point Buffers (HostSideSOA):
├─ Capacity = nPtsInitial × (1 + extra_space_pts)  [default: 1.15×]
└─ host_buffer[Capacity × 22] (8 bytes each, DOUBLE)

GPU Memory (per partition):
├─ Grid arrays (with halo overlap)
├─ Point arrays
└─ Temporary buffers for computation
```

### 7. Initialization Modes

#### Fresh Start from PNG
```
JSON Config (InputPNG path)
  ↓
PrepareGridAndPoints()
  - Load PNG files
  - Generate particles from Poisson disk sampling
  - Filter by land/ice masks
  - Save grid.h5 and s00000.h5
  ↓
GPU Initialization
  - setup() → initialize devices
  - split_hssoa_into_partitions() → partition data
  - transfer_to_device() → copy to GPU
  ↓
Ready for simulation
```

#### Resume from Checkpoint
```
Resume Snapshot File (s00001.h5)
  ↓
ReadPointsFromSnapshot()
  - Load point data
  - Restore simulation state (step, time)
  ↓
Load Grid Metadata
  - From existing grid.h5
  ↓
GPU Initialization (same as above)
  ↓
Ready for simulation
```

### 8. Data Structures and Indexing

#### Point Array Layout (Structure-of-Arrays)
22 arrays, each of size [Capacity]:
- Index 0: utility_data (crushed flag)
- Index 1: integer_cell_idx
- Index 2: integer_point_idx  
- Index 3: P (pressure)
- Index 4: Q (stress)
- Index 5: Jp_inv
- Index 6-7: posx (X, Y position)
- Index 8-9: velx (X, Y velocity)
- Index 10-13: Fe (elastic deformation gradient)
- Index 14-17: Bp (plastic strain)
- Index 18: thickness
- Index 19-21: color RGB

#### Grid Array Layout (Column-Major)
16 arrays, each of size [GridX × GridY]:
- Index: idx = j + i × GridY
- Index 0: mass
- Index 1-2: px, py (momentum)
- Index 3-5: visualization RGB
- Index 6-11: visualization data (Jpinv, P, Q, strain metrics, density)
- Index 12-15: forces and current velocity

### 9. What's Working vs. What's Not

**Working:**
- Parameter parsing from JSON
- PNG image loading and processing
- Poisson disk sampling (with caching)
- Grid creation and saving to HDF5
- Point generation and filtering
- Snapshot serialization/deserialization
- GPU device detection and setup
- Data partitioning across GPUs
- Wind/current interpolation framework
- GUI and post-processing infrastructure

**Not Working:**
- PNG-based initialization invocation
- Snapshot-based resume functionality
- Flow field data loading at startup
- Complete initialization pipeline

### 10. What Needs to Change

**Single Critical Change:**
Uncomment and fix `Model::LoadParameterFile()` (simulation/model.cpp lines 137-198)

**Specific fixes needed:**
1. Use parseResult map returned from SimParams::ParseFile()
2. Call sim_data.PrepareGridAndPoints() with PNG file paths (fresh start case)
3. Call sim_data.ReadPointsFromSnapshot() with resume filename (resume case)
4. Call gpu.initialize(), gpu.split_hssoa_into_partitions(), gpu.transfer_to_device()
5. Call wac_interpolator.SetHDF5Path() if UseCurrentData is true

**Files to modify:**
- `simulation/model.cpp` - Re-enable initialization sequence
- Possibly `simulation/data_manager/host_side_data.h` - Adjust interfaces if needed

**Files that are already correct:**
- All other files are complete and functional

---

## Conclusion

The simulation infrastructure is essentially complete. The only issue is that the initialization sequence in `Model::LoadParameterFile()` has been commented out, preventing the system from actually loading data and starting simulations. The code to uncomment and restore is already present in the same file.

All the machinery for PNG-based initialization, snapshot saving/loading, grid/point creation, GPU transfers, and flow field interpolation is implemented and ready. It just needs to be called.

**Estimated effort to fully restore functionality:** 30-60 minutes to uncomment, test, and debug the initialization sequence.

