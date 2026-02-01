cmake_minimum_required(VERSION 3.20.2)

project(inPlaneMPM LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -use_fast_maths --expt-relaxed-constexpr")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

option(BUILD_CLI_VERSION "Build CLI version" ON)
option(BUILD_GUI_VERSION "Build Qt version" OFF)
option(BUILD_POSTPROCESSOR "Build the tool for visualization" OFF)
option(BUILD_PREPARER "Build the tool for preparing the data" OFF)
option(BUILD_PREPARER_CLI "Build the CLI tool for preparing the data" OFF)
option(BUILD_COMPRESSOR "Build the CLI compressor utility" OFF)
option(BUILD_ADDITIONAL_UTILITIES "Build additional utilities" OFF)

option(ENABLE_NAN_CHECKS "Perform NaN checks during the simulation" OFF)

if(BUILD_GUI_VERSION OR BUILD_POSTPROCESSOR OR BUILD_PREPARER)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
endif()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED COMPONENTS CXX C)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(PNG REQUIRED)


if(BUILD_GUI_VERSION)
# Qt GUI executable

include(FetchContent)
FetchContent_Declare(
    PBrowser
    GIT_REPOSITORY https://github.com/igorg520b/PBrowser.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(PBrowser)


      find_package(VTK 9.1 REQUIRED COMPONENTS
      CommonColor
      CommonCore
      FiltersSources
      FiltersGeometry
      InteractionStyle
      RenderingContextOpenGL2
      RenderingCore
      RenderingFreeType
      RenderingGL2PSOpenGL2
      RenderingOpenGL2
      IOGeometry
      RenderingAnnotation
      RenderingLabel
      IOImage
      ViewsQt
      FiltersFlowPaths
    )
    find_package(Qt5 COMPONENTS Widgets REQUIRED)
    add_compile_definitions(QT_NO_KEYWORDS)

    add_executable(gplate
        gui/main.cpp

        gui/windows/mainwindow.cpp
        gui/windows/mainwindow.h
        gui/windows/mainwindow.ui
        icons.qrc

        gui/vtk/visual_representation.cpp
        gui/vtk/visual_representation.h

        gui/backgroundworker.cpp
        gui/backgroundworker.h
        gui/parameters_wrapper.h

        # simulation
        simulation/parameters_sim.h
        simulation/parameters_sim.cpp
        simulation/model.cpp
        simulation/model.h

        simulation/givens.cuh
        simulation/helper_math.cuh
        simulation/gpu_implementation5.h
        simulation/gpu_implementation5.cpp

        simulation/hssoa/host_side_soa.h
        simulation/hssoa/host_side_soa.cpp
        simulation/hssoa/proxypoint2d.h
        simulation/hssoa/proxypoint2d.cpp

        simulation/gpu_partition.h
        simulation/partition_params.h
        simulation/gpu_partition.cpp
        simulation/kernel_declarations.cuh
        simulation/kernels.cu

        simulation/data_manager/host_side_data.h
        simulation/data_manager/host_side_data.cpp
        simulation/data_manager/windandcurrentinterpolator.cpp
        simulation/data_manager/windandcurrentinterpolator.h
        simulation/data_manager/poisson_disk_sampling.h

        gui/colormap.h
        gui/colormap.cpp
        simulation/data_manager/intinterval.h
    )
    set_source_files_properties(simulation/gpu_partition.cpp simulation/kernels.cu simulation/kernel_declarations.cuh PROPERTIES LANGUAGE CUDA)

    message(${PBrowser_SOURCE_DIR})
    target_include_directories(gplate PRIVATE ${PBrowser_SOURCE_DIR} ${PBrowser_SOURCE_DIR}/propertybrowser)
    target_include_directories(gplate PRIVATE ${PBrowser_SOURCE_DIR}/objectpropertybrowser)
    target_link_libraries(gplate PRIVATE PBrowser)

    set_target_properties(gplate PROPERTIES CUDA_ARCHITECTURES "80;89")
    set_target_properties(gplate PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    target_include_directories(gplate PRIVATE gui gui/windows gui/vtk simulation simulation/hssoa simulation/data_manager External External/rapidjson)
    target_link_libraries(gplate PRIVATE Qt5::Widgets Qt5::Core)

    # Eigen
    find_package(Eigen3 REQUIRED NO_MODULE)
    target_link_libraries(gplate PRIVATE Eigen3::Eigen)

    # VTK
    target_link_libraries(gplate PRIVATE ${VTK_LIBRARIES})

    # HDF5
    target_link_libraries(gplate PRIVATE ${HDF5_LIBRARIES})
    target_include_directories(gplate PRIVATE ${HDF5_INCLUDE_DIRS})

    # spdlog
    target_link_libraries(gplate PRIVATE spdlog::spdlog)

    # OpenMP
    target_compile_options(gplate PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
    target_link_libraries(gplate PRIVATE gomp)

    vtk_module_autoinit(
    TARGETS gplate
    MODULES ${VTK_LIBRARIES}
    )

    if(ENABLE_NAN_CHECKS)
        target_compile_definitions(gplate PRIVATE ENABLE_NAN_CHECKS)
    endif()

endif()




if(BUILD_CLI_VERSION)

# CLI version
add_executable(cplate
    cli/main.cpp

    # simulation
    simulation/parameters_sim.h
    simulation/parameters_sim.cpp
    simulation/model.cpp
    simulation/model.h

    simulation/givens.cuh
    simulation/helper_math.cuh
    simulation/gpu_implementation5.h
    simulation/gpu_implementation5.cpp

    simulation/hssoa/host_side_soa.h
    simulation/hssoa/host_side_soa.cpp
    simulation/hssoa/proxypoint2d.h
    simulation/hssoa/proxypoint2d.cpp

    simulation/gpu_partition.h
    simulation/partition_params.h
    simulation/gpu_partition.cpp
    simulation/kernel_declarations.cuh
    simulation/kernels.cu

    simulation/data_manager/host_side_data.h
    simulation/data_manager/host_side_data.cpp
    simulation/data_manager/windandcurrentinterpolator.cpp
    simulation/data_manager/windandcurrentinterpolator.h

    simulation/data_manager/poisson_disk_sampling.h

    gui/colormap.h
    gui/colormap.cpp

    simulation/data_manager/intinterval.h

)
set_source_files_properties(simulation/gpu_partition.cpp simulation/kernels.cu simulation/kernel_declarations.cuh PROPERTIES LANGUAGE CUDA)

set_target_properties(cplate PROPERTIES CUDA_ARCHITECTURES "80;89")
set_target_properties(cplate PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(cplate PRIVATE Eigen3::Eigen spdlog::spdlog)
target_link_libraries(cplate PRIVATE fmt::fmt)

target_include_directories(cplate PRIVATE simulation simulation/hssoa simulation/data_manager External External/rapidjson)


if(HDF5_INCLUDE_DIRS)
    target_link_libraries(cplate PRIVATE ${HDF5_LIBRARIES})
else()
    target_link_libraries(cplate PRIVATE hdf5_cpp-shared)
endif()

target_include_directories(cplate PRIVATE ${HDF5_INCLUDE_DIRS})

target_compile_options(cplate PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
target_link_libraries(cplate PRIVATE gomp)

endif()




if(BUILD_POSTPROCESSOR)
    find_package(VTK 9.1 REQUIRED COMPONENTS
      CommonColor
      CommonCore
      FiltersSources
      FiltersGeometry
      InteractionStyle
      RenderingContextOpenGL2
      RenderingCore
      RenderingFreeType
      RenderingGL2PSOpenGL2
      RenderingOpenGL2
      IOGeometry
      RenderingAnnotation
      RenderingLabel
      IOImage
      ViewsQt
      FiltersFlowPaths
    )
    find_package(Qt5 COMPONENTS Widgets REQUIRED)
    add_compile_definitions(QT_NO_KEYWORDS)

    add_executable(visualizer
    	postprocessor/pp_main.cpp
    	postprocessor/window/visualizer_mainwindow.cpp
    	postprocessor/window/visualizer_mainwindow.h
        postprocessor/window/render_selector_dialog.cpp
        postprocessor/window/render_selector_dialog.h
        icons.qrc

        gui/vtk/visual_representation.cpp
        gui/vtk/visual_representation.h

        simulation/parameters_sim.h
        simulation/parameters_sim.cpp

        gui/colormap.h
        gui/colormap.cpp

        simulation/hssoa/host_side_soa.h
        simulation/hssoa/host_side_soa.cpp
        simulation/hssoa/proxypoint2d.h
        simulation/hssoa/proxypoint2d.cpp

        simulation/data_manager/windandcurrentinterpolator.cpp
        simulation/data_manager/windandcurrentinterpolator.h

        simulation/data_manager/intinterval.h

        postprocessor/frame_utils.h
        postprocessor/frame_utils.cpp

        simulation/data_manager/host_side_data.h
        simulation/data_manager/host_side_data.cpp
    )

    target_include_directories(visualizer PRIVATE postprocessor postprocessor/window gui External gui/vtk)
    target_include_directories(visualizer PRIVATE simulation simulation/hssoa simulation/data_manager)
    target_link_libraries(visualizer PRIVATE Qt5::Widgets Qt5::Core)

    # Eigen
    find_package(Eigen3 REQUIRED NO_MODULE)
    target_link_libraries(visualizer PRIVATE Eigen3::Eigen)

    # VTK
    target_link_libraries(visualizer PRIVATE ${VTK_LIBRARIES})

    # HDF5
    target_link_libraries(visualizer PRIVATE ${HDF5_LIBRARIES})
    target_include_directories(visualizer PRIVATE ${HDF5_INCLUDE_DIRS})

    # spdlog
    target_link_libraries(visualizer PRIVATE spdlog::spdlog)

    vtk_module_autoinit(
    TARGETS visualizer
    MODULES ${VTK_LIBRARIES}
    )

    target_compile_options(visualizer PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
    target_link_libraries(visualizer PRIVATE gomp)

    if(ENABLE_NAN_CHECKS)
        target_compile_definitions(cplate PRIVATE ENABLE_NAN_CHECKS)
    endif()

endif()






if(BUILD_PREPARER)

    find_package(VTK 9.1 REQUIRED COMPONENTS
    CommonCore
    IOGeometry
    FiltersSources
    FiltersGeometry
    IOImage
    InteractionStyle
    RenderingContextOpenGL2
    RenderingCore
    RenderingFreeType
    RenderingGL2PSOpenGL2
    RenderingOpenGL2
    RenderingAnnotation
    RenderingLabel
    ViewsQt
    FiltersFlowPaths
    )

    add_executable(preparer
        preparer/preparer_main.cpp
        preparer/preparer_mainwindow.cpp
        preparer/preparer_mainwindow.h

        preparer/parameterparser.h
        preparer/parameterparser.cpp

        preparer/data_preparer.h
        preparer/data_preparer.cpp

        preparer/flowfieldgenerator.cpp
        preparer/flowfieldgenerator.h



        simulation/data_manager/intinterval.h

        gui/vtk/visual_representation.cpp
        gui/vtk/visual_representation.h
        gui/colormap.h
        gui/colormap.cpp
        simulation/parameters_sim.h
        simulation/parameters_sim.cpp

        simulation/hssoa/host_side_soa.h
        simulation/hssoa/host_side_soa.cpp
        simulation/hssoa/proxypoint2d.h
        simulation/hssoa/proxypoint2d.cpp

        simulation/data_manager/host_side_data.h
        simulation/data_manager/host_side_data.cpp

        simulation/data_manager/windandcurrentinterpolator.cpp
        simulation/data_manager/windandcurrentinterpolator.h
        simulation/data_manager/poisson_disk_sampling.h
    )

    # Note: preparer no longer uses CUDA for Kelvin wake computation
    # Kelvin wake is computed on CPU using OpenMP

    target_link_libraries(preparer PRIVATE spdlog::spdlog Eigen3::Eigen ${HDF5_LIBRARIES} ${VTK_LIBRARIES})
    target_link_libraries(preparer PRIVATE Qt${QT_VERSION_MAJOR}::Widgets )

    target_include_directories(preparer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${HDF5_INCLUDE_DIRS})
    target_include_directories(preparer PRIVATE External preparer/vtk preparer/svg preparer/processing preparer gui/vtk gui simulation simulation/hssoa simulation/data_manager)

    # OpenMP support for parallel Kelvin wake computation
    target_compile_options(preparer PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
    target_link_libraries(preparer PRIVATE gomp)

    target_link_libraries(preparer PRIVATE PNG::PNG)

    vtk_module_autoinit(
    TARGETS preparer
    MODULES ${VTK_LIBRARIES}
    )

endif()




if(BUILD_PREPARER_CLI)
    find_path(NANOSVG_INCLUDE_DIR NAMES nanosvg.h PATH_SUFFIXES nanosvg)

    add_executable(preparer_cli
        preparer/cli/cli_preparer_main.cpp
        preparer/data_preparer.cpp
        simulation/data_manager/host_side_data.cpp
        simulation/parameters_sim.cpp
        preparer/parameterparser.cpp
        simulation/hssoa/host_side_soa.cpp
        gui/colormap.cpp
        simulation/data_manager/windandcurrentinterpolator.cpp
        simulation/hssoa/proxypoint2d.cpp
    )

    target_link_libraries(preparer_cli PRIVATE spdlog::spdlog Eigen3::Eigen ${HDF5_LIBRARIES})
    target_link_libraries(preparer_cli PRIVATE PNG::PNG)
    target_include_directories(preparer_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${HDF5_INCLUDE_DIRS})
    target_include_directories(preparer_cli PRIVATE External preparer/vtk preparer/svg preparer/processing preparer gui/vtk gui simulation simulation/hssoa simulation/data_manager)

    # OpenMP support for parallel Kelvin wake computation
    target_compile_options(preparer_cli PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
    target_link_libraries(preparer_cli PRIVATE gomp)

endif()


if(BUILD_COMPRESSOR)
    add_executable(compressor
        compressor/core.cpp
        compressor/core.cpp
        compressor/main_serial.cpp
    )
    target_include_directories(compressor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${HDF5_INCLUDE_DIRS})
    target_link_libraries(compressor PRIVATE ${HDF5_LIBRARIES})

endif()



if(BUILD_ADDITIONAL_UTILITIES)

    add_executable(verify_sort
        _utilities/main_verify_sort.cpp

        simulation/parameters_sim.h
        simulation/parameters_sim.cpp
        simulation/hssoa/host_side_soa.h
        simulation/hssoa/host_side_soa.cpp
        simulation/hssoa/proxypoint2d.h
        simulation/hssoa/proxypoint2d.cpp
    )
    target_include_directories(verify_sort PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${HDF5_INCLUDE_DIRS})
    target_include_directories(verify_sort PRIVATE _utilities simulation simulation/hssoa simulation/data_manager External)
    target_link_libraries(verify_sort PRIVATE ${HDF5_LIBRARIES} spdlog::spdlog Eigen3::Eigen fmt::fmt)


    add_executable(verify_omp
    _utilities/main_verify_omp.cpp
    )
    target_compile_options(verify_omp PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
    target_link_libraries(verify_omp PRIVATE gomp)

    add_executable(snapshot_compressor
        _utilities/main_snapshot_compressor.cpp
        simulation/parameters_sim.h
        simulation/parameters_sim.cpp
        simulation/hssoa/host_side_soa.h
        simulation/hssoa/host_side_soa.cpp
        simulation/hssoa/proxypoint2d.h
        simulation/hssoa/proxypoint2d.cpp
        simulation/data_manager/host_side_data.cpp
        simulation/data_manager/windandcurrentinterpolator.cpp
        gui/colormap.cpp
    )
    target_include_directories(snapshot_compressor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${HDF5_INCLUDE_DIRS})
    target_include_directories(snapshot_compressor PRIVATE _utilities simulation simulation/hssoa simulation/data_manager External gui)
    target_link_libraries(snapshot_compressor PRIVATE ${HDF5_LIBRARIES} spdlog::spdlog Eigen3::Eigen fmt::fmt)



endif()
