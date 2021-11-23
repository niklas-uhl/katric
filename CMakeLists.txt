cmake_minimum_required(VERSION 3.14)
project(parallel-triangle-counter LANGUAGES CXX)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

    set(CMAKE_CXX_STANDARD 17)

    # Let's ensure -std=c++xx instead of -std=g++xx
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Let's nicely support folders in IDEs
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    # Add build type Asan for using AddressSanitizer
    #include(cmake/AddAsanBuildType.cmake)
    include(CTest)
endif()

option(USE_SUBMODULES "This options prevents loading dependencies via FetchContent, but assumes that they are loaded as git submodules" OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

find_package(sparsehash REQUIRED)
add_library(sparsehash INTERFACE)
target_include_directories(sparsehash INTERFACE ${SPARSEHASH_INCLUDE_DIR})

find_package(Boost)
#find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")


# set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

if(NOT USE_SUBMODULES)
    include(FetchContent)

    FetchContent_Declare(
        catch
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v2.13.7
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(catch)

    FetchContent_Declare(
        CLI11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v1.9.1
        GIT_PROGRESS TRUE
        )
    FetchContent_MakeAvailable(CLI11)

    FetchContent_Declare(
        debug_assert
        GIT_REPOSITORY https://github.com/foonathan/debug_assert.git
        GIT_PROGRESS TRUE
        )
    FetchContent_MakeAvailable(debug_assert)

    FetchContent_Declare(
      backward-cpp
      GIT_REPOSITORY https://github.com/bombela/backward-cpp.git
      GIT_TAG v1.6
      GIT_PROGRESS TRUE
      )
    FetchContent_GetProperties(backward-cpp)
    if (NOT backward-cpp_POPULATED)
      FetchContent_Populate(backward-cpp)
      add_subdirectory(${backward-cpp_SOURCE_DIR} EXCLUDE_FROM_ALL)
    endif()

    FetchContent_Declare(
      fmt
      GIT_REPOSITORY https://github.com/fmtlib/fmt.git
      GIT_TAG 8.0.1
      GIT_PROGRESS TRUE
      )
    FetchContent_GetProperties(fmt)
    if (NOT fmt_POPULATED)
      FetchContent_Populate(fmt)
      add_subdirectory(${fmt_SOURCE_DIR} EXCLUDE_FROM_ALL)
    endif()

    FetchContent_Declare(
      cereal
      GIT_REPOSITORY https://github.com/USCiLab/cereal.git
      GIT_PROGRESS TRUE
      )
    FetchContent_GetProperties(cereal)
    if (NOT cereal_POPULATED)
      FetchContent_Populate(cereal)
      add_subdirectory(${cereal_SOURCE_DIR} EXCLUDE_FROM_ALL)
    endif()

    FetchContent_Declare(
        kagen
        GIT_REPOSITORY https://github.com/niklas-uhl/KaGen.git
        GIT_TAG origin/update-sampling
        )
    FetchContent_GetProperties(kagen)
    if(NOT kagen_POPULATED)
        FetchContent_Populate(kagen)
        set(kagen_POPULATED FALSE)
    endif()
    # FetchContent_Declare(
    #         tlx
    #         GIT_REPOSITORY https://github.com/tlx/tlx.git
    #         SOURCE_DIR ${kagen_SOURCE_DIR}/extlib/sampling/extlib/tlx/tlx
    # )
    # if(NOT tlx_POPULATED)
    #     FetchContent_Populate(tlx)
    # endif()
else()
    add_subdirectory(extern/cli11)
    #add_subdirectory(extern/tlx)
    set(kagen_SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/kagen)
endif()
find_package(CGAL)
if(NOT kagen_POPULATED)
    #add_subdirectory("${kagen_SOURCE_DIR}" EXCLUDE_FROM_ALL)
    set(KAGEN_INCLUDE_DIRS
        ${kagen_SOURCE_DIR}/extlib/sampling/sampling
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
    add_subdirectory(${kagen_SOURCE_DIR}/extlib/sampling EXCLUDE_FROM_ALL)
    add_library(kagen INTERFACE)
    target_include_directories(kagen INTERFACE ${KAGEN_INCLUDE_DIRS})
    target_link_libraries(kagen INTERFACE sampling tlx CGAL::CGAL)
    #set(KAGEN_LINK_LIBRARIES sampling tlx)
    set(kagen_POPULATED TRUE)
endif()

#FetchContent_MakeAvailable(tlx)

# Library usage:
#target_link_libraries(<openmp-app> PUBLIC OpenMP::OpenMP_CXX)
#target_link_libraries(<mpi-app> PUBLIC MPI::MPI_CXX)

# The compiled library code is here
add_subdirectory(src)

# The executable code is here
add_subdirectory(apps)

add_subdirectory(extern)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()
