cmake_minimum_required(VERSION 3.20.2)

project(WaveIce_MPM LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -use_fast_maths")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

option(BUILD_CLI_VERSION "Build CLI version" ON)
option(BUILD_GUI_VERSION "Build Qt version" OFF)
option(BUILD_POSTPROCESSOR "Build the tool for visualization" OFF)
option(BUILD_PREPARER "Build the tool for preparing the data" OFF)

if(BUILD_GUI_VERSION OR BUILD_POSTPROCESSOR OR BUILD_PREPARER)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
endif()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenJPEG REQUIRED)



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
    )
    find_package(Qt5 COMPONENTS Widgets REQUIRED)
    add_compile_definitions(QT_NO_KEYWORDS)

    add_executable(gm2m
        gui/main.cpp

        gui/windows/mainwindow.cpp
        gui/windows/mainwindow.h
        gui/windows/mainwindow.ui
        icons.qrc

        gui/vtk/vtk_representation.cpp
        gui/vtk/vtk_representation.h

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
        simulation/gpu_partition.cpp
        simulation/kernels.cuh
        
        simulation/data_manager/snapshotmanager.h
        simulation/data_manager/snapshotmanager.cpp
        simulation/data_manager/windandcurrentinterpolator.cpp
        simulation/data_manager/windandcurrentinterpolator.h
        simulation/data_manager/poisson_disk_sampling.h

        gui/colormap.h
        gui/colormap.cpp
    )
    set_source_files_properties(simulation/gpu_partition.cpp PROPERTIES LANGUAGE CUDA)
    
    message(${PBrowser_SOURCE_DIR})
    target_include_directories(gm2m PRIVATE ${PBrowser_SOURCE_DIR} ${PBrowser_SOURCE_DIR}/propertybrowser)
    target_include_directories(gm2m PRIVATE ${PBrowser_SOURCE_DIR}/objectpropertybrowser)
    target_link_libraries(gm2m PRIVATE PBrowser)

    set_target_properties(gm2m PROPERTIES CUDA_ARCHITECTURES "80;89")
    set_target_properties(gm2m PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    target_include_directories(gm2m PRIVATE gui gui/windows gui/vtk simulation simulation/hssoa simulation/data_manager External External/rapidjson)
    target_link_libraries(gm2m PRIVATE Qt5::Widgets Qt5::Core)

    # Eigen
    find_package(Eigen3 REQUIRED NO_MODULE)
    target_link_libraries(gm2m PRIVATE Eigen3::Eigen)

    # VTK
    target_link_libraries(gm2m PRIVATE ${VTK_LIBRARIES})

    # HDF5
    target_link_libraries(gm2m PRIVATE ${HDF5_LIBRARIES})
    target_include_directories(gm2m PRIVATE ${HDF5_INCLUDE_DIRS})

    # spdlog
    target_link_libraries(gm2m PRIVATE spdlog::spdlog)

    target_include_directories(gm2m PRIVATE ${OPENJPEG_INCLUDE_DIRS})
    target_link_libraries(gm2m PRIVATE ${OPENJPEG_LIBRARIES})

    # OpenMP
    target_compile_options(gm2m PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
    target_link_libraries(gm2m PRIVATE gomp)

    vtk_module_autoinit(
    TARGETS gm2m
    MODULES ${VTK_LIBRARIES}
    )

endif()




if(BUILD_CLI_VERSION)

# CLI version
add_executable(cm2m
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
    simulation/gpu_partition.cpp
    simulation/kernels.cuh


    simulation/data_manager/snapshotmanager.h
    simulation/data_manager/snapshotmanager.cpp
    simulation/data_manager/windandcurrentinterpolator.cpp
    simulation/data_manager/windandcurrentinterpolator.h

    simulation/data_manager/poisson_disk_sampling.h

    gui/colormap.h
    gui/colormap.cpp

)
set_source_files_properties(simulation/gpu_partition.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(cm2m PROPERTIES CUDA_ARCHITECTURES "80;89")
set_target_properties(cm2m PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(cm2m PRIVATE Eigen3::Eigen spdlog::spdlog)
target_link_libraries(cm2m PRIVATE fmt::fmt)

target_include_directories(cm2m PRIVATE simulation simulation/hssoa simulation/data_manager External External/rapidjson)


if(HDF5_INCLUDE_DIRS)
    target_link_libraries(cm2m PRIVATE ${HDF5_LIBRARIES})
else()
    target_link_libraries(cm2m PRIVATE hdf5_cpp-shared)
endif()

target_include_directories(cm2m PRIVATE ${HDF5_INCLUDE_DIRS})

target_include_directories(cm2m PRIVATE ${OPENJPEG_INCLUDE_DIRS})
target_link_libraries(cm2m PRIVATE ${OPENJPEG_LIBRARIES})

target_compile_options(cm2m PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
target_link_libraries(cm2m PRIVATE gomp)

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
    )
    find_package(Qt5 COMPONENTS Widgets REQUIRED)
    add_compile_definitions(QT_NO_KEYWORDS)

    add_executable(pp
    	postprocessor/pp_main.cpp
    	postprocessor/window/pp_mainwindow.cpp
    	postprocessor/window/pp_mainwindow.h
    	postprocessor/window/pp_mainwindow.ui
        icons.qrc
        
        postprocessor/vtk/vtk_visualization.cpp
        postprocessor/vtk/vtk_visualization.h

        # simulation
        simulation/parameters_sim.h
        simulation/parameters_sim.cpp

        postprocessor/framedata.h
        postprocessor/framedata.cpp
        postprocessor/generalgriddata.h
        postprocessor/generalgriddata.cpp

        gui/colormap.h
        gui/colormap.cpp

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
        simulation/gpu_partition.cpp
        simulation/kernels.cuh

        simulation/data_manager/snapshotmanager.h
        simulation/data_manager/snapshotmanager.cpp
        simulation/data_manager/windandcurrentinterpolator.cpp
        simulation/data_manager/windandcurrentinterpolator.h
    )

    target_include_directories(pp PRIVATE postprocessor postprocessor/window postprocessor/vtk gui External)
    target_include_directories(pp PRIVATE simulation simulation/hssoa simulation/data_manager)
    target_link_libraries(pp PRIVATE Qt5::Widgets Qt5::Core)

    # Eigen
    find_package(Eigen3 REQUIRED NO_MODULE)
    target_link_libraries(pp PRIVATE Eigen3::Eigen)

    # VTK
    target_link_libraries(pp PRIVATE ${VTK_LIBRARIES})

    # HDF5
    target_link_libraries(pp PRIVATE ${HDF5_LIBRARIES})
    target_include_directories(pp PRIVATE ${HDF5_INCLUDE_DIRS})

    # spdlog
    target_link_libraries(pp PRIVATE spdlog::spdlog)

    vtk_module_autoinit(
    TARGETS pp
    MODULES ${VTK_LIBRARIES}
    )

    target_include_directories(pp PRIVATE ${OPENJPEG_INCLUDE_DIRS})
    target_link_libraries(pp PRIVATE ${OPENJPEG_LIBRARIES})

    set_target_properties(pp PROPERTIES CUDA_ARCHITECTURES "80;89")
    set_target_properties(pp PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)


    target_compile_options(pp PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
    target_link_libraries(pp PRIVATE gomp)

endif()






if(BUILD_PREPARER)

    find_path(NANOSVG_INCLUDE_DIR NAMES nanosvg.h PATH_SUFFIXES nanosvg)
    if(NOT NANOSVG_INCLUDE_DIR)
        message(FATAL_ERROR "nanosvg.h not found. Please install libnanosvg-dev.")
    endif()

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
    )

    add_executable(MPM_setup_prepare
        preparer/main.cpp
        preparer/mainwindow.cpp
        preparer/mainwindow.h
        preparer/mainwindow.ui

        preparer/processing/satelliteimageprocessor.h 
        preparer/processing/satelliteimageprocessor.cpp
        preparer/svg/bezierpath.h 
        preparer/svg/bezierpath.cpp
        preparer/parameterparser.h 
        preparer/parameterparser.cpp
        preparer/processing/flowdataprocessor.h 
        preparer/processing/flowdataprocessor.cpp
        preparer/vtk/vtkFLUENTCFFCustomReader.h 
        preparer/vtk/vtkFLUENTCFFCustomReader.cxx
        preparer/processing/mainimageimporter.h 
        preparer/processing/mainimageimporter.cpp
    )

    target_link_libraries(MPM_setup_prepare PRIVATE spdlog::spdlog Eigen3::Eigen ${HDF5_LIBRARIES} ${VTK_LIBRARIES})
    target_link_libraries(MPM_setup_prepare PRIVATE Qt${QT_VERSION_MAJOR}::Widgets )

    target_include_directories(MPM_setup_prepare PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${NANOSVG_INCLUDE_DIR} ${HDF5_INCLUDE_DIRS})
    target_include_directories(MPM_setup_prepare PRIVATE External preparer/vtk preparer/svg preparer/processing preparer)

    vtk_module_autoinit(
    TARGETS MPM_setup_prepare
    MODULES ${VTK_LIBRARIES}
    )

endif()

