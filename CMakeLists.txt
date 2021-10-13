cmake_minimum_required(VERSION 3.14)
project(parallel-triangle-counter LANGUAGES CXX)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

    set(CMAKE_CXX_STANDARD 17)

    # Let's ensure -std=c++xx instead of -std=g++xx
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Let's nicely support folders in IDEs
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    # Add build type Asan for using AddressSanitizer
    include(cmake/AddAsanBuildType.cmake)
endif()

option(USE_SUBMODULES "This options prevents loading dependencies via FetchContent, but assumes that they are loaded as git submodules" OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

if(NOT USE_SUBMODULES)
    include(FetchContent)

    FetchContent_Declare(
        CLI11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v1.9.1
        )
    FetchContent_MakeAvailable(CLI11)

    FetchContent_Declare(
        json 
        GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent.git
        GIT_TAG v3.10.2
        )
    FetchContent_GetProperties(json)
    if (NOT json_POPULATED)
        FetchContent_Populate(json)
        add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()

    FetchContent_Declare(
        kagen
        GIT_REPOSITORY https://github.com/sebalamm/KaGen.git
        GIT_TAG 1fe87f7131ca740d803b764d2fab2fae819abf33
        )
    FetchContent_GetProperties(kagen)
    if(NOT kagen_POPULATED)
        FetchContent_Populate(kagen)
        set(kagen_POPULATED FALSE)
    endif()
    FetchContent_Declare(
            tlx
            GIT_REPOSITORY https://github.com/tlx/tlx.git
            SOURCE_DIR ${kagen_SOURCE_DIR}/extlib/sampling/extlib/tlx/tlx
    )
    if(NOT tlx_POPULATED)
        FetchContent_Populate(tlx)
    endif()
else()
    add_subdirectory(extern/cli11)
    #add_subdirectory(extern/tlx)
    set(kagen_SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/kagen)
endif()
if(NOT kagen_POPULATED)
    add_subdirectory("${kagen_SOURCE_DIR}")
    set(KAGEN_INCLUDE_DIRS
            ${kagen_SOURCE_DIR}/extlib/sampling/sampling
            ${kagen_SOURCE_DIR}/extlib/sampling/extlib/tlx/tlx
            ${kagen_SOURCE_DIR}/app
            ${kagen_SOURCE_DIR}/include
            ${kagen_SOURCE_DIR}/include/generators
            ${kagen_SOURCE_DIR}/include/generators/barabassi
            ${kagen_SOURCE_DIR}/include/generators/geometric
            ${kagen_SOURCE_DIR}/include/generators/geometric/delaunay
            ${kagen_SOURCE_DIR}/include/generators/geometric/rgg
            ${kagen_SOURCE_DIR}/include/generators/gnm
            ${kagen_SOURCE_DIR}/include/generators/gnp
            ${kagen_SOURCE_DIR}/include/generators/hyperbolic
            ${kagen_SOURCE_DIR}/include/io
            ${kagen_SOURCE_DIR}/include/tools
            ${kagen_SOURCE_DIR}/interface/)
    set(KAGEN_LINK_LIBRARIES sampling tlx)
    set(kagen_POPULATED TRUE)
endif()

#FetchContent_MakeAvailable(tlx)

find_package(sparsehash REQUIRED)
find_package(Boost)
#find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
# Library usage:
#target_link_libraries(<openmp-app> PUBLIC OpenMP::OpenMP_CXX)
#target_link_libraries(<mpi-app> PUBLIC MPI::MPI_CXX)

# The compiled library code is here
add_subdirectory(src)

# The executable code is here
add_subdirectory(apps)

add_subdirectory(extern)
