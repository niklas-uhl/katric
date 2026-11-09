cmake_minimum_required(VERSION 3.25)
project(cetric LANGUAGES C CXX)

option(CETRIC_BUILD_APPS "Build the katric and shmetric executables" ON)
option(CETRIC_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
option(CETRIC_USE_OPENMP "Enable OpenMP-based parallelization (--parallelization-method omp_for)" ON)
option(CETRIC_BINARY_INTERSECTION_RECURSIVE "Use recursion in binary search intersection" OFF)
option(CETRIC_ENABLE_FORMAT_CMAKE "Fetch Format.cmake and provide the format/check-format/fix-format targets" OFF)
set(CETRIC_CXX_FLAGS "" CACHE STRING "Additional compile options for katric's own targets (semicolon-separated)")
set(CETRIC_ACTIVE_LOG_LEVEL "INFO" CACHE STRING
    "Compile-time spdlog level; SPDLOG_DEBUG/SPDLOG_TRACE calls below it are compiled out")
set_property(CACHE CETRIC_ACTIVE_LOG_LEVEL PROPERTY STRINGS TRACE DEBUG INFO WARN ERROR CRITICAL OFF)
set(KASSERT_ASSERTION_LEVEL $<IF:$<CONFIG:Debug>,30,0> CACHE STRING "KASSERT assertion level")

# MPI and OpenMP always come from the system, TBB preferably (see extern/), everything else is fetched.
find_package(MPI REQUIRED)
if(CETRIC_USE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

add_subdirectory(extern)

# Compile options for katric's own executables and tests (warnings are set by the dev preset). Linked PRIVATE, so they
# never leak to consumers of cetric-lib.
add_library(cetric_cxx_flags INTERFACE)
target_compile_options(cetric_cxx_flags INTERFACE ${CETRIC_CXX_FLAGS})

# The algorithms are header-only.
add_library(cetric-lib INTERFACE)
file(GLOB_RECURSE CETRIC_HEADERS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/*.h")
target_sources(cetric-lib INTERFACE FILE_SET HEADERS BASE_DIRS include FILES ${CETRIC_HEADERS})
target_compile_features(cetric-lib INTERFACE cxx_std_20)
target_link_libraries(
  cetric-lib
  INTERFACE absl::flat_hash_map
            absl::flat_hash_set
            absl::hash
            tlx
            cereal::cereal
            magic_enum::magic_enum
            fmt::fmt
            spdlog::spdlog
            kassert::kassert
            kamping::types
            graph-io
            message-queue
            # last: a system TBB adds e.g. /opt/homebrew/include, which must not shadow pinned fmt/spdlog/abseil headers
            MPI::MPI_CXX
            TBB::tbb)
# kassert only reads the level from a KASSERT_ASSERTION_LEVEL property on each consuming target, never from the cache
# variable, so the variable is passed on as a definition here.
target_compile_definitions(cetric-lib INTERFACE SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${CETRIC_ACTIVE_LOG_LEVEL}
                                                KASSERT_ASSERTION_LEVEL=${KASSERT_ASSERTION_LEVEL})
if(CETRIC_USE_OPENMP)
  target_link_libraries(cetric-lib INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(cetric-lib INTERFACE CETRIC_HAS_OPENMP)
endif()
if(CETRIC_BINARY_INTERSECTION_RECURSIVE)
  target_compile_definitions(cetric-lib INTERFACE CETRIC_BINARY_INTERSECTION_RECURSIVE)
endif()

if(CETRIC_BUILD_APPS)
  add_subdirectory(apps)
endif()

if(CETRIC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
