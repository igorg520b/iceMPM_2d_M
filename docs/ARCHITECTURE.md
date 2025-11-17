# System Architecture: plateMPM

A comprehensive overview of the plateMPM GPU-accelerated Material Point Method simulation for sea ice dynamics.

## 1. System Architecture

### Core Components

```
Model
├── HostSideData (sim_data)
│   ├── SimParams (prms) - All parameters from JSON
│   ├── HostSideSOA (hssoa) - Material points (Structure-of-Arrays)
│   ├── host_grid_buffer - Computational grid
│   ├── WindAndCurrentInterpolator (waci) - Flow field data
│   └── Various buffers (landmask, colors, RGB)
│
├── GPU_Implementation5 (gpu)
│   └── vector<GPU_Partition> partitions (one per GPU device)
│
└── Model::Step() - Main simulation loop
```

### Data Flow

```
JSON Configuration
    ↓
SimParams::ParseFile() → parameter map
    ↓
Model::LoadParameterFile()
    ├── Load grid from grid.h5 (created by plate_preparer)
    ├── Load initial particle state from s00000.h5
    ├── Initialize GPU partitions and transfer data
    └── Load flow field (grid_flow.h5)
    ↓
Model::Prepare()
    ├── Update GPU constants
    ├── Load flow data to GPU
    ↓
Model::Step() loop
    ├── Execute GPU simulation kernels
    ├── Transfer results to host (for visualization frames)
    └── Save frame data periodically
```

---

## 2. Parameter System (SimParams)

### Key Configuration Parameters

**Initialization:**
- `GridData`: Path to grid.h5 (created by plate_preparer)
- `CurrentVelocityData`: Path to grid_flow.h5 (flow field)
- `Snapshot`: Optional custom snapshot for resume

**Domain:**
- `DimensionHorizontal`: Physical domain width (meters)
- `InitialTimeStep`: Timestep size (seconds, constant)
- `GridXTotal`, `GridYTotal`: Computational grid dimensions
- `CellSize`: Physical size of grid cells

**Ice Rheology (Elliptic Failure):**
- `IceCompressiveStrength`: Maximum pressure
- `IceTensileStrength`: Tensile strength
- `IceShearStrength`: Maximum deviatoric stress

**Plasticity (Drucker-Prager):**
- `DP_phi`: Angle of internal friction
- `DP_threshold_p`: Elastic-plastic transition pressure

### Parameter Loading

```cpp
std::map<std::string, std::string> parseResult = prms.ParseFile(jsonPath);
```

Returns a map with all JSON fields for flexible access.

---

## 3. Data Loading Pipeline

### Stage 1: Grid Initialization (from plate_preparer output)

**Input:** grid.h5 file from plate_preparer

**Process:**
1. `LoadGridDataFromFile()` reads grid.h5
2. Extracts grid dimensions, cell size, offsets
3. Loads landmask and visualization colors
4. Allocates `host_grid_buffer` for grid arrays (16 arrays per cell)

**Grid Storage:**
- Column-major indexing: `idx = j + i * GridYTotal`
- 16 arrays per grid cell: pressure, stress, mass, velocity, etc.
- Landmask indicates modeled region (255) vs. excluded areas

### Stage 2: Particle State Loading

**Fresh Start:** Load from initial snapshot (s00000.h5)
```cpp
// Load initial particle state
hsd.ReadPointsFromSnapshot(snapshotPath);
hsd.AllocatePointArrays();
```

**Resume:** Load from saved checkpoint
```cpp
// Resume from interrupted simulation
hsd.ReadPointsFromSnapshot(checkpointPath);
hsd.AllocatePointArrays();
```

**Snapshot Format:**
- HDF5 dataset: `pts_data[nPtsArrays, HSSOA_size]`
- Attributes: nPtsInitial, SimulationStep, SimulationTime, ParticleVolume
- 22 arrays: position, velocity, deformation gradient, stress, color, thickness, etc.

### Stage 3: GPU Initialization

```cpp
// Setup GPU devices and partitions
gpu.SplitIntoPartitionsAndTransferToDevice();
  ├─ ComputeHelperVariables() - Calculate particle mass and volume
  ├─ initialize() - Setup CUDA devices
  ├─ split_hssoa_into_partitions() - Divide points by x-coordinate
  ├─ allocate_device_arrays() - Allocate GPU memory
  └─ transfer_to_device() - Copy data to GPU
```

**Partitioning Strategy:**
- Points divided along X-axis across available GPU devices
- Each partition processes points in X-range independently
- Halo exchange at partition boundaries for inter-GPU communication

### Stage 4: Flow Field Setup

**If CurrentVelocityData is specified:**
```cpp
waci.SetHDF5Path(grid_flow_path);  // Load flow file metadata
waci.SetTime(simulation_time);      // Load frames for current time
gpu.transfer_wind_and_current_data_to_device();  // Copy to GPU
```

**Flow Data Format:**
- HDF5 datasets: `water_current_vx[num_frames, gx, gy]` and `water_current_vy`
- Temporal interpolation between frames on GPU
- Updated whenever simulation time requires new frames

---

## 4. Grid Data Structure

### Host-Side Grid Buffer

**Allocation:**
```cpp
host_grid_buffer[GridXTotal * GridYTotal * 16]  // 16 arrays per cell
```

**16 Grid Arrays (SimParams::GridArrIdx):**
- `grid_idx_mass` - Material mass
- `grid_idx_px`, `grid_idx_py` - Momentum components
- `grid_idx_vis_Jpinv` - Plastic compressibility
- `grid_idx_vis_P` - Pressure
- `grid_idx_vis_Q` - Deviatoric stress
- `grid_idx_vis_pts_density` - Point density
- `grid_idx_vis_strain_vonMises` - Von Mises strain
- `grid_idx_*` - Additional state variables

**Indexing:**
- Grid coordinates (i, j) map to index: `idx = j + i * GridYTotal`
- Column-major layout for GPU memory coalescing
- Extends to neighboring cells (halo) in multi-GPU scenarios

### Landmask Buffer

```cpp
landmask_buffer[GridXTotal * GridYTotal]  // uint8
```

- `255`: Modeled area (water/ice)
- `0-254`: Land (excluded from simulation)

---

## 5. Material Point System (HostSideSOA)

### Data Layout

**Structure-of-Arrays (SOA) Format:**
```cpp
// 22 separate arrays, each of size [capacity]
Array<double> host_buffer[nPtsArrays * capacity];
```

**Index into arrays (SimParams::PtArrIdx):**
- Position: X and Y coordinates (physical space)
- Velocity: Vx and Vy components
- Deformation gradient: Fe (2×2 matrix, 4 components)
- Stress: P (pressure), Q (deviatoric stress)
- State: Jp_inv (plastic compressibility)
- Material: Color (RGB from original image)
- Damage: Thickness (0-1 for crushed ice)

**Benefits of SOA:**
- GPU memory coalescing for SIMD efficiency
- Cache-friendly access patterns
- Easy vectorization for kernels

### Point Properties

**Position & Velocity:**
- Normalized to cell coordinates (0-1 within cell)
- Converted to physical coordinates during initialization

**Deformation Gradient (Fe):**
- 2×2 matrix stored as 4 scalars (row-major)
- Initialized to identity (elastic, no deformation)

**Stress:**
- P: Pressure (Pa)
- Q: Deviatoric stress magnitude

**Material State:**
- Jp_inv: Inverse of plastic compressibility (1/det(Fp))
- Color: RGB from original image (for visualization)
- Thickness: 0.0 (damaged/crushed) to 1.0 (intact)

### Point Capacity Management

```cpp
capacity = nPtsInitial * (1 + extra_space_pts)  // Default: 1.15x
```

- Extra space reserved for particles generated during simulation
- Halo particles from neighboring partitions
- Prevents reallocation during long simulations

---

## 6. GPU_Implementation5: Multi-GPU Support

### Device Management

**Initialization:**
```cpp
gpu.initialize()
  ├─ Query CUDA devices (cudaGetDeviceCount)
  ├─ Log device properties (compute capability, memory)
  ├─ Create GPU_Partition for each available device
  └─ Enable peer-to-peer access between devices
```

### Data Partitioning

**Distribution Strategy:**
```cpp
gpu.split_hssoa_into_partitions()
  ├─ Sort points by X-coordinate
  └─ Divide into nGPU partitions
```

Each partition:
- Manages points in X-range [x_min, x_max)
- Owns GPU grid with halo cells for boundary points
- Communicates via halo exchange

### Data Transfer

**Host to Device:**
```cpp
gpu.transfer_to_device()
  ├─ For each partition:
  │  ├─ Allocate GPU memory
  │  ├─ Copy point arrays (HostSideSOA)
  │  └─ Copy grid arrays (host_grid_buffer)
```

**Device to Host:**
```cpp
gpu.transfer_from_device()
  └─ Copy updated grid back to host (for frame generation)
```

---

## 7. Simulation Lifecycle

### Initialization (Model::LoadParameterFile)

```
1. Parse JSON configuration
   ↓
2. Create output directory structure
   ↓
3. Load grid.h5
   ├─ Grid dimensions and cell size
   ├─ Landmask and visualization colors
   ↓
4. Load initial snapshot (s00000.h5)
   ├─ Particle positions and properties
   ├─ Initial stress state
   ↓
5. Allocate point arrays
   ├─ Reserve extra capacity for generated particles
   ↓
6. GPU Initialization (SplitIntoPartitionsAndTransferToDevice)
   ├─ Setup CUDA devices
   ├─ Partition points across GPUs
   ├─ Allocate GPU memory
   ├─ Transfer to GPU
   ↓
7. Load flow field (if specified)
   ├─ Load grid_flow.h5
   ├─ Setup temporal interpolation
```

### Preparation (Model::Prepare)

```
1. Update GPU constants (SimParams → GPU constant memory)
2. Load initial flow data to GPU
3. Ready for simulation step
```

### Main Loop (Model::Step)

```
for each timestep:
  1. Execute GPU kernels:
     ├─ Point-to-Grid (P2G)
     ├─ Grid update (forces, constitutive model)
     ├─ Grid-to-Point (G2P)
     └─ Halo exchange (multi-GPU)

  2. Periodically:
     ├─ Transfer grid to host
     ├─ Save frame data (f00000.h5, etc.)
     ├─ Save snapshot (s00000.h5, etc.)
     └─ Check for termination
```

---

## 8. File I/O

### Input Files

**grid.h5** (from plate_preparer)
- Datasets: landmask, color_grid
- Attributes: grid dimensions, offsets, cell size

**grid_flow.h5** (optional, from FLUENT or prescribed)
- Datasets: water_current_vx[frames, gx, gy], water_current_vy
- Attributes: time_interval, loop_mode

**s00000.h5** (initial snapshot)
- Dataset: pts_data[nPtsArrays, HSSOA_size]
- Attributes: particle count, time step, particle volume

### Output Files

**output/frames/**
- `f00000.h5, f00001.h5, ...`: Grid-only frame data
- Format: Compressed HDF5 with visualization arrays
- Size: Relatively compact

**output/snapshots/**
- `s00000.h5, s00001.h5, ...`: Full particle + grid state
- Format: Uncompressed HDF5 for fast I/O
- Size: Large (includes all particle data)

**output/logs/**
- `multisink.txt`: Combined console + file logging

---

## 9. Key Design Principles

### Separation of Concerns
- **Data**: HostSideData manages all host-side buffers and I/O
- **GPU**: GPU_Implementation5 manages GPU computation
- **Orchestration**: Model coordinates between data and GPU

### Memory Efficiency
- **Structure-of-Arrays**: Column-major layout for GPU coalescing
- **Compression**: HDF5 compression for frame storage
- **Extra Capacity**: Reserve space for particle generation

### Robustness
- **Snapshot-based Resume**: Full state saved periodically
- **File Validation**: Check existence before loading
- **Error Logging**: Comprehensive logging to file and console

### Scalability
- **Multi-GPU Support**: X-axis partitioning with halo exchange
- **Configurable**: Grid size, particle count, timestep all adjustable
- **Large Domain**: Tested up to 5000×3000 grids, 100M+ points

---

## 10. Critical Interdependencies

### HostSideData
- **Owns**: SimParams, HostSideSOA, grid/point buffers, WindAndCurrentInterpolator
- **Used by**: Model (orchestration), GPU_Implementation5 (GPU kernels)
- **Responsibilities**: All data loading, memory allocation, I/O

### GPU_Implementation5
- **References**: HostSideData (for access to host buffers)
- **Owns**: GPU_Partition objects and GPU memory
- **Responsibilities**: GPU initialization, data transfer, kernel execution

### Model
- **Owns**: HostSideData, GPU_Implementation5
- **Entry point**: All operations coordinated through Model
- **Responsibilities**: Orchestration, step loop, high-level control

---

## 11. Performance Considerations

### GPU Memory
- Partitioning reduces per-GPU memory requirement
- Halo cells add overhead (~5-10% extra grid)
- Point capacity > nPtsInitial (extra space for generation)

### Computation
- Explicit time integration (no linear solver)
- GPU-resident grid (minimal PCIe bandwidth)
- Periodic frame saves (asynchronous writes)

### Data Transfer
- Grid: Transferred only when saving frames
- Points: Transferred only at initialization
- Flow field: Cached; updated when time requires new frame

---

## 12. Files Organization

```
plateMPM/
├── simulation/
│   ├── model.cpp/h - Main orchestration
│   ├── data_manager/
│   │   ├── host_side_data.cpp/h - Data management and I/O
│   │   ├── windandcurrentinterpolator.cpp/h - Flow field
│   │   └── parameters_sim.cpp/h - Parameter parsing
│   ├── gpu_implementation5.cpp/h - GPU management
│   ├── gpu_partition.cpp/h - Per-GPU state
│   └── hssoa/
│       ├── host_side_soa.h - Point data structure
│       └── proxypoint2d.cpp/h - Point access utilities
│
├── preparer/
│   ├── mainwindow.cpp/h - Qt GUI for preparation
│   ├── parameterparser.cpp/h - JSON parsing (preparer)
│   └── flowfieldgenerator.cpp/h - Flow field generation
│
├── postprocessor/
│   ├── window/pp_mainwindow.cpp/h - Visualization GUI
│   ├── frame_utils.cpp/h - Frame file utilities
│   └── visual_representation.cpp/h - VTK visualization
│
├── cli/
│   └── main.cpp - Command-line entry point
│
└── docs/
    └── ARCHITECTURE.md (this file)
```

---

This architecture provides a clean separation between data management, GPU computation, and high-level orchestration, enabling efficient large-scale simulations while maintaining code clarity and extensibility.
