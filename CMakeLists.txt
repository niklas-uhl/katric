cmake_minimum_required(VERSION 3.25)
project(katric LANGUAGES C CXX)

option(KATRIC_BUILD_APPS "Build the katric executable" ON)
option(KATRIC_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
option(KATRIC_BINARY_INTERSECTION_RECURSIVE "Use recursion in binary search intersection" OFF)
option(KATRIC_ENABLE_CLANG_TIDY "Run clang-tidy (.clang-tidy) on katric's own sources while building" OFF)
option(KATRIC_FETCH_TBB "Build the pinned oneTBB from source (shared) instead of using the system's" OFF)
option(KATRIC_ENABLE_FORMAT_CMAKE "Fetch Format.cmake and provide the format/check-format/fix-format targets" OFF)
# Competitors behind katric's benchmark interface (--algorithm <name>, doc/competitors.md), each opt-in
option(KATRIC_COMPETITOR_TRIC "Build TriC (Ghosh & Halappanavar) into katric as --algorithm tric" OFF)
option(KATRIC_COMPETITOR_XXTRIC "Build xxtric's CPU path (Ghosh, TriC's successor) into katric as --algorithm xxtric" OFF)
option(KATRIC_COMPETITOR_LCC "Build LCC (Strausz et al., with CLaMPI) into katric as --algorithm lcc; GPL-3.0, makes the binary GPL" OFF)
option(KATRIC_COMPETITOR_HAVOQGT "Build HavoqGT's triangle count (Pearce et al.) into katric as --algorithm havoqgt; needs Boost" OFF)
option(KATRIC_COMPETITOR_TRIPOLL "Build TriPoll (Steil et al., on YGM) into katric as --algorithm tripoll; needs Boost" OFF)
set(KATRIC_CXX_FLAGS "" CACHE STRING "Additional compile options for katric's own targets (semicolon-separated)")
set(KATRIC_ACTIVE_LOG_LEVEL "INFO" CACHE STRING
    "Compile-time spdlog level; SPDLOG_DEBUG/SPDLOG_TRACE calls below it are compiled out")
set_property(CACHE KATRIC_ACTIVE_LOG_LEVEL PROPERTY STRINGS TRACE DEBUG INFO WARN ERROR CRITICAL OFF)
set(KASSERT_ASSERTION_LEVEL $<IF:$<CONFIG:Debug>,30,0> CACHE STRING "KASSERT assertion level")

# MPI and TBB come from the system (TBB from source with KATRIC_FETCH_TBB), everything else is fetched.
find_package(MPI REQUIRED)

add_subdirectory(extern)

# Compile options for katric's own executables and tests (warnings are set by the dev preset). Linked PRIVATE, so they
# never leak to consumers of katric-lib.
add_library(katric_cxx_flags INTERFACE)
target_compile_options(katric_cxx_flags INTERFACE ${KATRIC_CXX_FLAGS})

# clang-tidy command for katric's own executables and tests (their CXX_CLANG_TIDY property; empty = off). The headers
# in include/katric are checked through these sources (HeaderFilterRegex in .clang-tidy). -std=c++20 is passed
# explicitly: compilers that default to C++20 (GCC >= 16) get no -std flag from CMake, so clang-tidy would parse katric
# as C++17.
set(KATRIC_CLANG_TIDY_COMMAND "")
if(KATRIC_ENABLE_CLANG_TIDY)
  find_program(KATRIC_CLANG_TIDY_EXE NAMES clang-tidy REQUIRED)
  set(KATRIC_CLANG_TIDY_COMMAND ${KATRIC_CLANG_TIDY_EXE} --extra-arg=-std=c++20)
endif()

# The algorithms are header-only.
add_library(katric-lib INTERFACE)
file(GLOB_RECURSE KATRIC_HEADERS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/*.h")
target_sources(katric-lib INTERFACE FILE_SET HEADERS BASE_DIRS include FILES ${KATRIC_HEADERS})
target_compile_features(katric-lib INTERFACE cxx_std_20)
target_link_libraries(
  katric-lib
  INTERFACE absl::flat_hash_map
            absl::flat_hash_set
            absl::hash
            magic_enum::magic_enum
            fmt::fmt
            spdlog::spdlog
            kassert::kassert
            kamping::types
            kamping::v2
            dstl::dstl
            KaGen::KaGen
            message-queue
            caliper
            adiak
            # last: a system TBB adds e.g. /opt/homebrew/include, which must not shadow pinned fmt/spdlog/abseil headers
            MPI::MPI_CXX
            TBB::tbb)
# KASSERT_ASSERTION_LEVEL is applied by kassert itself (it is the default level of the instance, see extern/).
target_compile_definitions(katric-lib INTERFACE SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${KATRIC_ACTIVE_LOG_LEVEL})
if(KATRIC_BINARY_INTERSECTION_RECURSIVE)
  target_compile_definitions(katric-lib INTERFACE KATRIC_BINARY_INTERSECTION_RECURSIVE)
endif()

add_subdirectory(src)

if(KATRIC_BUILD_APPS)
  add_subdirectory(apps)
endif()

if(KATRIC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
