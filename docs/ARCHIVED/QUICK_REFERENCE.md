# Quick Reference: Simulation Code Structure

## Current State Summary
The PNG-based initialization pipeline is **fully implemented but not being called**. The Model::LoadParameterFile() method is essentially empty (all initialization code is commented out on lines 157-196 of model.cpp).

## What Exists (Ready to Use)
1. **HostSideData::PrepareGridAndPoints()** - Load PNG images, create grid and particles
2. **HostSideData::ReadPointsFromSnapshot()** - Load from HDF5 snapshot file
3. **GPU_Implementation5::initialize()** - Setup GPU devices
4. **GPU_Implementation5::split_hssoa_into_partitions()** - Partition data across GPUs
5. **GPU_Implementation5::transfer_to_device()** - Copy data to GPU
6. **WindAndCurrentInterpolator::SetHDF5Path()** - Load flow field data

## Critical Issue
Model::LoadParameterFile() does NOT use the parsed parameters:
- InputPNG (PNG image path) - parsed but unused
- InputMap (grid HDF5 path) - parsed but unused
- InputFlowVelocity (flow field path) - parsed but unused
- UseCurrentData (boolean flag) - parsed but unused

## What Needs to Happen
Replace commented code in Model::LoadParameterFile() (lines 157-196) with:

```cpp
void Model::LoadParameterFile(std::string fileName, std::string resumeSnapshotFileName)
{
    // Parse JSON and setup logging (already working)
    
    if (resumeSnapshotFileName.empty()) {
        // Fresh start: load from PNG images
        sim_data.PrepareGridAndPoints(
            parseResult["InputPNG"],
            parseResult["InputColor"],
            parseResult["InputIceMask"],
            parseResult["InputCrushedMask"],
            outputDir.string(),
            prms.DimensionHorizontal,
            pointsPerCell
        );
    } else {
        // Resume: load from snapshot
        // First load grid metadata, then points
        sim_data.ReadPointsFromSnapshot(resumeSnapshotFileName);
    }
    
    // Initialize GPU
    gpu.initialize();
    gpu.split_hssoa_into_partitions();
    gpu.transfer_to_device();
    
    // Load flow data if needed
    if (prms.UseCurrentData && parseResult.count("InputFlowVelocity")) {
        wac_interpolator.SetHDF5Path(parseResult["InputFlowVelocity"]);
    }
}
```

## Data Flow Diagram

### Fresh Initialization (PNG Images)
```
JSON Config (InputPNG, InputMap, InputFlowVelocity, UseCurrentData)
         ↓
HostSideData::PrepareGridAndPoints()
  ├─ Load 4 PNG files
  ├─ Create grid.h5
  ├─ Generate/filter particles
  ├─ Save s00000.h5 (initial snapshot)
  └─ Allocate host buffers
         ↓
GPU Initialization Chain
  ├─ gpu.initialize()           → setup CUDA devices
  ├─ gpu.split_hssoa_into_partitions()  → divide points by X-axis
  └─ gpu.transfer_to_device()   → copy to GPU
         ↓
Wind/Current Data (if enabled)
  └─ wac_interpolator.SetHDF5Path()
         ↓
Model::Prepare()
  └─ Setup GPU constants
         ↓
Simulation Loop
```

### Resume from Checkpoint
```
Resume Snapshot File (s00001.h5)
         ↓
HostSideData::ReadPointsFromSnapshot()
  ├─ Load point data
  ├─ Restore simulation state (step, time)
  └─ Allocate host buffers
         ↓
[Same GPU initialization chain as above]
```

## Key File Locations

| Component | Header | Implementation |
|-----------|--------|-----------------|
| Parameter parsing | `simulation/parameters_sim.h` | `simulation/parameters_sim.cpp` |
| Host-side data | `simulation/data_manager/host_side_data.h` | `simulation/data_manager/host_side_data.cpp` |
| GPU implementation | `simulation/gpu_implementation5.h` | `simulation/gpu_implementation5.cpp` |
| Wind/current | `simulation/data_manager/windandcurrentinterpolator.h` | `simulation/data_manager/windandcurrentinterpolator.cpp` |
| Entry point | `cli/main.cpp` | N/A |
| Model orchestration | `simulation/model.h` | `simulation/model.cpp` |

## Important Class Relationships

```
Model (orchestrator)
├─ owns HostSideData
│  ├─ owns SimParams
│  ├─ owns HostSideSOA (point array buffer)
│  ├─ owns WindAndCurrentInterpolator
│  └─ methods: PrepareGridAndPoints(), ReadPointsFromSnapshot(), SaveSnapshot()
│
├─ owns GPU_Implementation5
│  ├─ references HostSideData (NOT owns)
│  ├─ owns vector<GPU_Partition>
│  └─ methods: initialize(), split_hssoa_into_partitions(), transfer_to_device()
│
└─ owns WindAndCurrentInterpolator
   └─ methods: SetHDF5Path(), SetTime(), GetInterpolatedValue()
```

## PNG File Requirements

For fresh initialization, provide 4 PNG files:
1. **Landmask** - Black (0-127) = water/modeled, White (128-255) = land
2. **Color** - RGB image for particle coloring
3. **Ice Mask** - White (128-255) = ice, Black (0-127) = no ice
4. **Crushed Mask** - 255 = not crushed, 0-254 = crushed with thickness value

All must have identical dimensions.

## HDF5 Data Structures

### grid.h5 (created during PrepareGrid)
```
landmask [GridX, GridY] (UINT8)
  - Attributes: GridXTotal, GridYTotal, OffsetX, OffsetY, 
                InitImageSizeX, InitImageSizeY, CellSize, DimensionHorizontal

color_grid [Height, Width, 3] (UINT8)
```

### Snapshot (s{frame:05d}.h5)
```
pts_data [nPtsArrays, HSSOA_size] (DOUBLE)
  - Attributes: nPtsInitial, SimulationStep, SimulationTime, 
                HSSOA_size, ParticleVolume, nPtsArrays
```

### Flow Field (InputFlowVelocity)
```
water_current_vx [num_frames, gx, gy] (DOUBLE)
  - Attributes: time_interval, loop_mode

water_current_vy [num_frames, gx, gy] (DOUBLE)
  - Attributes: time_interval, loop_mode
```

## Memory Allocation Pattern

```cpp
// Allocate grid buffers
HostSideData::AllocateGridArrays()
├─ landmask_buffer[GridX * GridY]
├─ original_image_colors_rgb[3 * ImgWidth * ImgHeight]
└─ host_grid_buffer[GridX * GridY * 16]

// Allocate point buffers  
HostSideData::AllocatePointArrays()
├─ hssoa.Allocate(capacity = nPtsInitial * 1.15)
│  └─ host_buffer[capacity * 22]
└─ point_partitions[nPtsInitial]
```

## Point Array Indices (SimParams::PtArrIdx)

| Index | Name | Size | Description |
|-------|------|------|-------------|
| 0 | idx_utility_data | 1 | Crushed flag and other flags |
| 1 | integer_cell_idx | 1 | Cell index (X) |
| 2 | integer_point_idx | 1 | Point ID |
| 3 | idx_P | 1 | Pressure |
| 4 | idx_Q | 1 | Stress |
| 5 | idx_Jp_inv | 1 | Inverse Jacobian determinant |
| 6-7 | posx | 2 | Position (X, Y) |
| 8-9 | velx | 2 | Velocity (Vx, Vy) |
| 10-13 | Fe00-Fe33 | 4 | Elastic deformation gradient |
| 14-17 | Bp00-Bp33 | 4 | Plastic strain |
| 18 | idx_thickness | 1 | Ice thickness |
| 19-21 | idx_pt_color_RGB | 3 | Color (R, G, B) |

## Grid Array Indices (SimParams::GridArrayIndex)

| Index | Name | Description |
|-------|------|-------------|
| 0 | grid_idx_mass | Mass |
| 1-2 | grid_idx_px, py | Momentum |
| 3-5 | grid_idx_vis_r, g, b | RGB color for visualization |
| 6 | grid_idx_vis_Jpinv | Jacobian visualization |
| 7 | grid_idx_vis_P | Pressure visualization |
| 8 | grid_idx_vis_Q | Stress visualization |
| 9 | grid_idx_vis_strain_EqvGreenLagrange | Strain visualization |
| 10 | grid_idx_vis_strain_vonMises | Von Mises strain |
| 11 | grid_idx_vis_pts_density | Point density |
| 12-13 | grid_idx_fx, fy | Forces |
| 14-15 | grid_idx_current_vx, vy | Current velocity |

## Important Notes

1. **Column-Major Indexing**: Grid uses `idx = j + i * GridY` format
2. **Image Flipping**: PNG origin (top-left) vs simulation origin (bottom-left) requires vertical flip
3. **Poisson Caching**: Point generation is cached in `_data/poisson_cache/`
4. **Snapshot Auto-Save**: Initial snapshot (s00000.h5) auto-generated after PrepareGridAndPoints()
5. **Multi-GPU**: Points automatically partitioned by X-coordinate across available GPUs
6. **Particle Filtering**: Particles on boundaries and on land are filtered out
