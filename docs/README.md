# plateMPM Documentation Index

Technical reference documentation for the plateMPM GPU-accelerated sea ice simulation framework.

## Quick Navigation

### For Users
- **[../README.md](../README.md)** - Complete workflow guide for all 4 stages (prepare, simulate, visualize, compress)
- **[../preparer/README.md](../preparer/README.md)** - plate_preparer domain-specific guide

### For Developers

#### System Design
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture, data flow, and design principles
  - 12 major sections covering the complete framework
  - Core components: Model, HostSideData, GPU_Implementation5, GPU_Partition
  - Initialization, preparation, and simulation lifecycle
  - Multi-GPU support and halo exchange strategy

#### API Reference
- **[CLASS_REFERENCE.md](CLASS_REFERENCE.md)** - API documentation for key classes
  - HostSideData: Grid and particle state management
  - SimParams: Parameter parsing and storage
  - HostSideSOA: Particle data structure (22 arrays)
  - Model: Orchestration and simulation control
  - GPU_Implementation5: GPU initialization and management
  - GPU_Partition: Per-GPU computation
  - WindAndCurrentInterpolator: Flow field loading
  - VisualRepresentation: VTK visualization

#### Data Formats
- **[DATA_STRUCTURES.md](DATA_STRUCTURES.md)** - HDF5 files, memory layouts, indexing conventions
  - HDF5 file formats: grid.h5, snapshots, frames, flow data
  - Host-side memory: HostSideSOA (22 particle arrays), grid buffers
  - GPU memory layout and data transfer patterns
  - Coordinate systems and transformations
  - 22 particle array indices and 16 grid array indices
  - Data access patterns and performance optimization

#### Feature Documentation
- **[FLUENT_IMPORT.md](FLUENT_IMPORT.md)** - Ocean current import from CFD simulations
  - Configuration and JSON parameters
  - FluentFlowImporter class design
  - 4-step workflow: geometry extraction, grid loading, transformation, rasterization
  - Coordinate system conversion (scientific → image space)
  - Integration with flow field generation pipeline

### Historical Documentation
- **[ARCHIVED/](ARCHIVED/)** - Previous analysis documents (kept for reference)
  - EXECUTIVE_SUMMARY.md - Original high-level overview
  - QUICK_REFERENCE.md - Old quick reference
  - ANALYSIS_INDEX.md - Original index of analysis files

---

## Documentation Philosophy

### Separation of Concerns

**User Documentation** (`../README.md`, `../preparer/README.md`)
- How to use the tools (step-by-step workflows)
- Configuration examples and troubleshooting
- Command-line instructions
- Typical use cases

**Technical Documentation** (this directory)
- System design and architecture
- Class APIs and method signatures
- Data structure specifications
- Implementation details
- Developer extension guide

### Coverage

This documentation covers:
- ✅ All 4 pipeline stages (prepare, simulate, visualize, compress)
- ✅ All major classes and their responsibilities
- ✅ Complete data structure specifications
- ✅ HDF5 file format details
- ✅ GPU memory management and optimization
- ✅ FLUENT CFD integration feature

### Future Additions

Possible extensions (not yet documented):
- DEVELOPMENT.md - Build instructions, git workflow, testing strategy
- PERFORMANCE.md - Profiling, optimization tips, scaling analysis
- TROUBLESHOOTING_ADVANCED.md - Deep debugging techniques

---

## Key Concepts

### Material Point Method (MPM)
- Hybrid Eulerian-Lagrangian approach combining grid + particles
- Each particle carries material state; grid transfers momentum
- P2G (point-to-grid), grid update, G2P (grid-to-point) cycle

### Structure-of-Arrays (SOA)
- 22 separate particle arrays instead of array of structures
- GPU memory coalescing efficiency
- Column-major indexing for cache friendliness

### Multi-GPU Architecture
- X-axis partitioning of particles across GPU devices
- Halo exchange at partition boundaries for communication
- Independent computation per partition

### Coordinate Systems
- **Physical**: Bottom-left origin, Y points up, meters
- **Image**: Top-left origin, Y points down, pixels
- **Grid**: Column-major indexing (j fast, i slow)

---

## Development Workflow

### Adding a New Feature

1. **Update parameters**: Modify `SimParams` in `simulation/parameters_sim.h`
2. **Add storage**: Add field to `HostSideData` in `simulation/data_manager/host_side_data.h`
3. **GPU transfer**: Update `GPU_Partition::transfer_to_device()` to copy new data
4. **GPU kernel**: Modify simulation kernels to use new data
5. **Visualization**: Update `VisualRepresentation` if data should be rendered
6. **Testing**: Create test case with known input/output
7. **Documentation**: Update relevant .md files in `docs/`

See [ARCHITECTURE.md](ARCHITECTURE.md) section 11 "Initialization Checklist" for complete details.

### Extending Post-Processor

1. **Add UI control**: New slider/checkbox in `pp_mainwindow.cpp`
2. **Add visualization**: New color mapping in `VisualRepresentation`
3. **Add frame loading**: New data extraction in `HostSideData::LoadFrameData()`
4. **Connect signals**: Wire UI → visualization updates

See [CLASS_REFERENCE.md](CLASS_REFERENCE.md) for UI and visualization classes.

---

## File Statistics

| File | Size | Content |
|------|------|---------|
| ARCHITECTURE.md | ~14 KB | System design, 12 major sections |
| CLASS_REFERENCE.md | ~14 KB | API reference, 8+ classes |
| DATA_STRUCTURES.md | ~25 KB | Format specs, 10 sections |
| FLUENT_IMPORT.md | ~24 KB | CFD integration, complete workflow |

**Total**: ~77 KB of technical documentation

---

## Quick References by Topic

### "How do I access particle data?"
→ See [DATA_STRUCTURES.md](DATA_STRUCTURES.md) section 4.1 "Accessing Particle Data"

### "How do I access grid data?"
→ See [DATA_STRUCTURES.md](DATA_STRUCTURES.md) section 4.2 "Accessing Grid Data"

### "What do the 22 particle arrays store?"
→ See [DATA_STRUCTURES.md](DATA_STRUCTURES.md) section 2.1 "HostSideSOA" (array index table)

### "What's the column-major indexing scheme?"
→ See [DATA_STRUCTURES.md](DATA_STRUCTURES.md) section 2.2 "Host Grid Buffer"

### "How does the simulation run?"
→ See [ARCHITECTURE.md](ARCHITECTURE.md) section 7 "Simulation Lifecycle"

### "How is FLUENT data imported?"
→ See [FLUENT_IMPORT.md](FLUENT_IMPORT.md) section 2 "Implementation Architecture"

### "What classes do I need to modify to add a feature?"
→ See [CLASS_REFERENCE.md](CLASS_REFERENCE.md) section "Initialization Checklist"

### "What are the HDF5 file formats?"
→ See [DATA_STRUCTURES.md](DATA_STRUCTURES.md) section 1 "HDF5 File Formats"

---

## Related Resources

### Source Code Locations
- **Core simulation**: `simulation/model.cpp`, `simulation/gpu_implementation5.cpp`
- **Data management**: `simulation/data_manager/host_side_data.cpp`
- **Post-processor**: `postprocessor/window/pp_mainwindow.cpp`, `visual_representation.cpp`
- **FLUENT import**: `preparer/fluentflowimporter.cpp`

### External Dependencies
- **CUDA**: GPU computation backend
- **VTK**: Visualization library, FLUENT grid reading
- **HDF5**: File I/O format
- **Eigen**: Vector/matrix math
- **Qt**: GUI framework (preparer, post-processor)
- **spdlog**: Logging library

### Building
```bash
cd /path/to/plateMPM
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

---

## Version Information

- **Framework**: plateMPM (GPU-accelerated Material Point Method)
- **Last Updated**: November 2024
- **Documentation Status**: Complete (all major features covered)
- **GPU Support**: NVIDIA CUDA (multi-GPU capable)

---

This documentation is the authoritative technical reference. For workflows and user guides, refer to the main [README.md](../README.md).
