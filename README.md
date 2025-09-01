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

For compute servers or headless environments, it's sufficient to build just the command-line simulation tool:
```
cmake -B build -S .
cmake --build build -j
```
Feel free to build in a directory of your choice and/or configure with `ccmake` instead of `cmake`. Note that if the server is not running a Ubuntu-like system, you may have to build the required libraries and make them available to `cmake`.

The CLI version of the project, titled `cm2m`, generates the relatively compact HDF5 output files, which can be downloaded form the server and visualized later. This option is best suited for large simulations (>10M points).

### 🎯 CUDA Architecture Flags

In `CMakeLists.txt`, you may find lines like the following:
```
set_target_properties(gm2m PROPERTIES CUDA_ARCHITECTURES "80;89")
```

You can customize this line to match your GPU's compute capability. The minimum supported version is 6.1, although this hasn't been explicitly tested.

### 🪟 Building on Windows (Experimental)
While not officially tested, the project should be buildable on Windows using MSVC and QtCreator. Assuming that the required libraries are present and accessible, the project should be able to compile and run in Windows.



## Running the Simulaiton

The `input` directory contains sample simulation setups. Copy this directory into the `build` directory, then run the GUI simulation with the following command:
```
./gm2m input/n700/nares700.json
```
Alternatively, run the gm2m project from QtCreator and set `input/n700/nares700.json` as the command line argument in 'Run Settings'. Once the window appears, select 'Reset Camera' from the menu. You should see the simulated area. To run the simulation, click the 'Play' button in the toolbar or select 'Start/Pause' from the menu. To see whether the simulation is running, from the drop-down list select one of the parameters to visualize. Some parameters are:
- `grid_Jpinv`: shows 1/Jₚ = det(Fₚ)⁻¹, which is the measure of the relative surface density. Blue color shows the dispersal of material (lower density), where red shows the accumulated material (ridges).
- `grid_P`: in-plane pressure in the ice sheet. Set the range in the first spin box to 5.0, which will correspond to [0, 10^5.0]. The second spin box introduces a manual slow-down for GPU computation (useful for weaker GPUs that also need to draw the user interface).
- `grid_Q`: deviatoric stress
- `partitions`: split of material points between different GPU partitions. If the system has more than 1 GPU device, partitions are allocated on different devices.

![Screenshot of the GUI version](/screenshots/n700.png)

### 📁 Output Files
Simulation results are saved to `ouput` directory. The sub-directories include:
- `frames`: contain custom HDF5 files with 'rendered' and compressed results for Jp^-1, P, Q, vx, vy. These files can be 'rendered' into image at post-processing.
- `snapshots`: these files are backup of full points' data, from which a simulation can be resumed.

### 🔁 Resuming a Simulation
To resume a simulation (after unexpected stop), run: 
```
./cm2m input/n700/nares700.json --resume 'output/n700/snapshots/s00120.h5'
```
Resuming currently works for CLI version and wasn't fully tested in the GUI version.

The list of included sample setups:

- `n700`: Nares Strait simulation with 700x463 grid (note that actual modelled area may be smaller). Approx. 230,000 material points are generated dynamically at first run.
- `n2k`: Higher-resolution 2000×1300 Nares Strait setup.
- `n5k`: Large-scale 5000×3000 Nares Strait simulation.
- `cb`: Experimental setup modeling the Confederation Bridge region.

The CLI version `./cm2m` can be run in the same way as the GUI version.

## Configuration

The `.json` file allows to configure the simulation.

- `InputPNG`: PNG image of the simulated area. At this stage it is mostly used for visualization, i.e., it assigns color to the landmass and moving material points. The image must be 3-channel (RGB), not 4-channel (sorry).
- `InputMap`: A custom HDF5 file generated at pre-processing stage that defines the simulated region and the initial distribution of ice.
- `InputFlowVelocity`: Velocity map generated as pre-processing that is used as the force source. The file is a custom HDF5 format, produced from the FLUENT output at pre-processing.
- `DimensionHorizontal`: Horizontal span in meters (or other units of your choice).
- `InitialTimeStep`: Time step in seconds. This value remains constant and does not change.
- `IceCompressiveStrength`:Parameter of the elliptic failure curve - maximum pressure.
- `IceTensileStrength`:Minimum pressure / tensile strength.
- `IceShearStrength`:Max Q (deviatoric stress) value of the elliptic curve.
- `DP_phi`, `DP_threshold_p`: Drucker-Prager parameters for the linear portion of the yield curve (see plasticity model)



## Output

Output files are written in custom HDF5 format. The 'frames' are compressed with OpenJPEG 2k for faster download from a server. The output files can be post-processed to generate animation frames. Run the post-processing tool as follows

```
./pp input/n700/nares700.json --frames 'output/n700/frames'
```
After cropping/zooming on the frame, select 'Generate All' from the menu to generate the animation frames. 
* At this time the post-processor has some issues when running on Wayland (Ubuntu 25.04). Working on a fix.

## Citation

This work can be cited as follows:

Gribanov, I., Waseda, T., Taylor, R., & Turnbull, I. (2025, July). Application of the Material Point Method for Simulating Sea Ice Breakup. In Proceedings of the 28th International Conference on Port and Ocean Engineering under Arctic Conditions (POAC), St. John’s, Newfoundland and Labrador, Canada.
