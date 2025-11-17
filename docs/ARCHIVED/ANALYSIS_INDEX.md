# Code Analysis Documentation Index

This directory contains comprehensive analysis of the plateMPM simulation codebase structure. Start with the document most relevant to your needs:

## Documents Included

### 1. **EXECUTIVE_SUMMARY.md** - START HERE
**Best for:** Quick overview, understanding the big picture
- Current system status
- Critical missing link (commented out initialization)
- Data flow understanding
- What's working vs. not working
- 10 key findings
- Estimated effort to fix (30-60 minutes)

### 2. **QUICK_REFERENCE.md** - FOR DEVELOPERS
**Best for:** Developers implementing fixes
- Current code that exists and is ready to use
- What needs to happen (code snippet provided)
- Data flow diagrams
- File location reference table
- Class relationships
- PNG file requirements and HDF5 data structures
- Memory allocation patterns
- Point and grid array indices reference

### 3. **SIMULATION_STRUCTURE_ANALYSIS.md** - COMPREHENSIVE DEEP DIVE
**Best for:** Complete understanding, architectural decisions
- Detailed system overview (with ASCII diagrams)
- Parameter parsing mechanism
- PNG-based initialization (detailed walkthrough)
- Snapshot-based initialization (detailed walkthrough)
- GPU_Implementation5 class structure
- Initialization sequence in main()
- Wind and current data handling
- Data allocation hierarchy
- Current state and issues
- Required changes for snapshot-based init
- Design patterns and key interdependencies
- Memory requirements estimation
- 625 lines of detailed technical documentation

## Quick Navigation

### For "How do I..." Questions

**...understand how the system works?**
→ EXECUTIVE_SUMMARY.md sections 3-4

**...enable PNG-based initialization?**
→ QUICK_REFERENCE.md "What Needs to Happen" section

**...add new functionality?**
→ SIMULATION_STRUCTURE_ANALYSIS.md sections 5-7, 14

**...understand data structures?**
→ SIMULATION_STRUCTURE_ANALYSIS.md sections 8, 12

**...implement snapshot resume?**
→ SIMULATION_STRUCTURE_ANALYSIS.md section 10

**...find where something is implemented?**
→ QUICK_REFERENCE.md "Key File Locations" table

**...understand parameter flow?**
→ SIMULATION_STRUCTURE_ANALYSIS.md sections 2-3

**...debug GPU initialization?**
→ SIMULATION_STRUCTURE_ANALYSIS.md sections 5, 9

## Key Findings Summary

### The Critical Issue
`Model::LoadParameterFile()` (simulation/model.cpp lines 157-196) has all initialization code commented out. This single function needs to be restored to enable the entire system.

### What Exists (Ready to Use)
1. PNG image loading and grid creation
2. Poisson disk sampling for particle generation
3. Snapshot serialization/deserialization
4. GPU device initialization
5. Data partitioning for multi-GPU
6. Wind/current field interpolation

### What's Missing
The glue code that calls these components in `Model::LoadParameterFile()`

### Effort Required
Approximately 30-60 minutes to uncomment, test, and debug

## Code Organization

```
plateMPM/
├─ simulation/
│  ├─ model.h/cpp              ← MAIN ENTRY POINT (needs fixing)
│  ├─ parameters_sim.h/cpp     ← Parameter parsing
│  ├─ gpu_implementation5.h/cpp ← GPU management
│  └─ data_manager/
│     ├─ host_side_data.h/cpp       ← Grid & point management
│     ├─ windandcurrentinterpolator.h/cpp ← Flow field
│     └─ [other utilities]
├─ cli/
│  └─ main.cpp                 ← Entry point (already correct)
├─ gui/                        ← GUI components
├─ postprocessor/              ← Visualization
└─ preparer/                   ← Data preparation tools
```

## Key Concepts Explained

### Structure-of-Arrays (SOA) Layout
Points stored in 22 separate arrays rather than one array of structs:
- Allows GPU memory coalescing
- Enables SIMD vectorization
- Slightly more complex but critical for performance

### Column-Major Indexing
Grids use `idx = j + i * GridY` format:
- Matches Fortran/scientific computing conventions
- Different from C/C++ row-major convention
- Consistent throughout codebase

### HDF5 Data Format
Used for persistent storage:
- Grid metadata (grid.h5)
- Snapshots (s00000.h5, s00001.h5, ...)
- Frames (f00000.h5, ...)
- Wind/current data (flow_velocities.h5)

### Multi-GPU Partitioning
Points automatically divided by X-coordinate:
- Each GPU gets a slice of the domain
- Halo exchange at boundaries
- Automatic rebalancing as particles move

## Important Parameters

**From JSON Configuration:**
- `InputPNG` → Path to landmask PNG image
- `InputMap` → Path to grid HDF5 (or unused)
- `InputFlowVelocity` → Path to flow field HDF5
- `UseCurrentData` → Enable/disable current effects
- `DimensionHorizontal` → Physical domain size

**Simulation Control:**
- `SaveSnapshots` → Enable snapshot saving
- `SnapshotPeriod` → Frames between snapshots
- `SimulationEndTime` → When to stop simulation
- `AnimationFramePeriod` → Frames between visualization outputs

## Related Files

- `SIMULATION_STRUCTURE_ANALYSIS.md` - Full technical documentation (625 lines)
- `QUICK_REFERENCE.md` - Developer quick reference
- `EXECUTIVE_SUMMARY.md` - High-level overview
- Input examples in `input/n5k/`, `input/n700/`, etc.

---

**Generated:** 2025-11-11
**Project:** plateMPM - GPU-based Ice Dynamics Simulation
**Language:** C++/CUDA
**Status:** 95% complete, initialization disabled
