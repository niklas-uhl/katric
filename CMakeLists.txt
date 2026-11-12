cmake_minimum_required(VERSION 3.25)
project(cetric LANGUAGES C CXX)

option(CETRIC_BUILD_APPS "Build the katric executable" ON)
option(CETRIC_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
option(CETRIC_BINARY_INTERSECTION_RECURSIVE "Use recursion in binary search intersection" OFF)
option(CETRIC_ENABLE_CLANG_TIDY "Run clang-tidy (.clang-tidy) on katric's own sources while building" OFF)
option(CETRIC_FETCH_TBB "Build the pinned oneTBB from source (shared) instead of using the system's" OFF)
option(CETRIC_ENABLE_FORMAT_CMAKE "Fetch Format.cmake and provide the format/check-format/fix-format targets" OFF)
set(CETRIC_CXX_FLAGS "" CACHE STRING "Additional compile options for katric's own targets (semicolon-separated)")
set(CETRIC_ACTIVE_LOG_LEVEL "INFO" CACHE STRING
    "Compile-time spdlog level; SPDLOG_DEBUG/SPDLOG_TRACE calls below it are compiled out")
set_property(CACHE CETRIC_ACTIVE_LOG_LEVEL PROPERTY STRINGS TRACE DEBUG INFO WARN ERROR CRITICAL OFF)
set(KASSERT_ASSERTION_LEVEL $<IF:$<CONFIG:Debug>,30,0> CACHE STRING "KASSERT assertion level")

# MPI and TBB come from the system (TBB from source with CETRIC_FETCH_TBB), everything else is fetched.
find_package(MPI REQUIRED)

add_subdirectory(extern)

# Compile options for katric's own executables and tests (warnings are set by the dev preset). Linked PRIVATE, so they
# never leak to consumers of cetric-lib.
add_library(cetric_cxx_flags INTERFACE)
target_compile_options(cetric_cxx_flags INTERFACE ${CETRIC_CXX_FLAGS})

# clang-tidy command for katric's own executables and tests (their CXX_CLANG_TIDY property; empty = off). The headers
# in include/cetric are checked through these sources (HeaderFilterRegex in .clang-tidy). -std=c++20 is passed
# explicitly: compilers that default to C++20 (GCC >= 16) get no -std flag from CMake, so clang-tidy would parse katric
# as C++17.
set(CETRIC_CLANG_TIDY_COMMAND "")
if(CETRIC_ENABLE_CLANG_TIDY)
  find_program(CETRIC_CLANG_TIDY_EXE NAMES clang-tidy REQUIRED)
  set(CETRIC_CLANG_TIDY_COMMAND ${CETRIC_CLANG_TIDY_EXE} --extra-arg=-std=c++20)
endif()

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
            magic_enum::magic_enum
            fmt::fmt
            spdlog::spdlog
            kassert::kassert
            kamping::types
            KaGen::KaGen
            message-queue
            caliper
            adiak
            # last: a system TBB adds e.g. /opt/homebrew/include, which must not shadow pinned fmt/spdlog/abseil headers
            MPI::MPI_CXX
            TBB::tbb)
# KASSERT_ASSERTION_LEVEL is applied by kassert itself (it is the default level of the instance, see extern/).
target_compile_definitions(cetric-lib INTERFACE SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${CETRIC_ACTIVE_LOG_LEVEL})
if(CETRIC_BINARY_INTERSECTION_RECURSIVE)
  target_compile_definitions(cetric-lib INTERFACE CETRIC_BINARY_INTERSECTION_RECURSIVE)
endif()

add_subdirectory(src)

if(CETRIC_BUILD_APPS)
  add_subdirectory(apps)
endif()

if(CETRIC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
