include(FetchContent)
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON) # targets found via FIND_PACKAGE_ARGS must be visible outside extern/
# All dependencies are fetched here. To work on one locally, configure with
# -DFETCHCONTENT_SOURCE_DIR_<NAME>=/path/to/checkout (e.g. ..._GRAPH-IO), see CMakeUserPresets.json.template.
# Every dependency is pinned to a release tag or a commit SHA (with the branch name as comment), never to a branch:
# FetchContent does not reliably update a branch checkout, so a branch pin silently builds against stale code.

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11
  GIT_TAG v2.5.0
  GIT_SHALLOW ON
  SYSTEM EXCLUDE_FROM_ALL)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG 12.2.1
  GIT_SHALLOW ON
  SYSTEM EXCLUDE_FROM_ALL)

# spdlog uses the (header-only) fmt fetched above
set(SPDLOG_FMT_EXTERNAL_HO ON)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog
  GIT_TAG v1.16.0
  GIT_SHALLOW ON
  SYSTEM EXCLUDE_FROM_ALL)

# Caliper: annotation-based performance profiling (phase timings), configured at runtime (--caliper-config)
set(WITH_MPI ON CACHE BOOL "" FORCE)
set(WITH_TOOLS ON CACHE BOOL "" FORCE)
set(WITH_ADIAK ON CACHE BOOL "" FORCE)
set(WITH_DOCS OFF CACHE BOOL "" FORCE)
# Adiak: run metadata (config, counters) recorded next to the Caliper profile
set(ENABLE_MPI ON CACHE BOOL "" FORCE)
set(ENABLE_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  adiak
  GIT_REPOSITORY https://github.com/LLNL/adiak
  GIT_TAG v0.5.0
  GIT_SHALLOW ON
  SYSTEM EXCLUDE_FROM_ALL
  OVERRIDE_FIND_PACKAGE)
FetchContent_Declare(
  caliper
  GIT_REPOSITORY https://github.com/LLNL/Caliper
  GIT_TAG v2.15.0
  GIT_SHALLOW ON
  SYSTEM EXCLUDE_FROM_ALL)

set(MAGIC_ENUM_OPT_BUILD_EXAMPLES OFF)
set(MAGIC_ENUM_OPT_BUILD_TESTS OFF)
FetchContent_Declare(
  magic_enum
  GIT_REPOSITORY https://github.com/Neargye/magic_enum
  GIT_TAG v0.9.7
  GIT_SHALLOW ON
  SYSTEM EXCLUDE_FROM_ALL)

FetchContent_Declare(
  kassert
  GIT_REPOSITORY https://github.com/kamping-site/kassert
  GIT_TAG v1.1.0
  GIT_SHALLOW ON
  SYSTEM EXCLUDE_FROM_ALL)

# Hash tables.
set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_BUILD_TESTING OFF)
FetchContent_Declare(
  absl
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp
  GIT_TAG 20250814.0
  GIT_SHALLOW ON
  # GCC on macOS: abseil defines _XOPEN_SOURCE, which hides quick_exit from <cstdlib>. The patch only changes the
  # condition for __APPLE__, so it is applied on all platforms.
  PATCH_COMMAND ${CMAKE_COMMAND} -DPATCH=${PROJECT_SOURCE_DIR}/cmake/patches/abseil_macos_no_define_xopen_source.patch
                -P ${PROJECT_SOURCE_DIR}/cmake/apply_patch.cmake
  SYSTEM EXCLUDE_FROM_ALL)


# oneTBB comes from the system (module, package manager); on a cluster, load the TBB module before configuring.
# KATRIC_FETCH_TBB builds the pinned version from source instead, as a shared library (static builds are unsupported
# upstream). Declared here, made available below.
if(KATRIC_FETCH_TBB)
  set(TBB_TEST OFF)
  set(TBB_EXAMPLES OFF)
  set(TBB_STRICT OFF) # do not build TBB itself with -Werror
  set(TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH ON) # tbbbind (NUMA support) is not needed
  FetchContent_Declare(
    TBB
    GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB
    GIT_TAG v2023.2.0
    GIT_SHALLOW ON
    SYSTEM EXCLUDE_FROM_ALL)
endif()

# GoogleTest + MPI test helpers (only downloaded if tests are built, see the end of this file). Declared before
# kamping-v2, which declares its own (older) KaTestrophe: the first declaration wins.
FetchContent_Declare(
  KaTestrophe
  GIT_REPOSITORY https://github.com/kamping-site/KaTestrophe.cmake
  GIT_TAG v1.2.0
  GIT_SHALLOW ON
  SYSTEM EXCLUDE_FROM_ALL)

# KaGen: katric's input (--kagen_option_string).
set(KAGEN_BUILD_APPS OFF)
set(KAGEN_BUILD_TOOLS OFF)
set(KAGEN_BUILD_EXAMPLES OFF)
set(KAGEN_BUILD_TESTS OFF)
set(KAGEN_USE_CGAL OFF)
set(KAGEN_USE_SPARSEHASH OFF)
set(KAGEN_USE_XXHASH ON)
FetchContent_Declare(
  kagen
  GIT_REPOSITORY https://github.com/KarlsruheGraphGeneration/KaGen
  GIT_TAG v1.3.0
  GIT_SHALLOW ON
  SYSTEM EXCLUDE_FROM_ALL)

# KaMPIng (v1, master): provides kamping::types (MPI datatype mapping, used by katric) and kamping::kamping (used by the
# spdlog adapter). kamping-v2 and kamping-spdlog-adapter reuse these targets instead of fetching their own copies.
set(KAMPING_ENABLE_REFLECTION OFF)
set(KAMPING_ENABLE_SERIALIZATION OFF)
FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping
  GIT_TAG v0.3.2
  GIT_SHALLOW ON
  SYSTEM EXCLUDE_FROM_ALL)

# MPI-aware logging setup (rank pattern, root-only and gather sinks) on top of spdlog.
FetchContent_Declare(
  kamping-spdlog-adapter
  GIT_REPOSITORY https://github.com/kamping-site/kamping-spdlog-adapter
  GIT_TAG v1.0.0
  GIT_SHALLOW ON
  SYSTEM EXCLUDE_FROM_ALL)

# Reflection (Boost.PFR, as_struct datatypes) is not needed by katric.
set(KAMPING_V2_ENABLE_REFLECTION OFF)
FetchContent_Declare(
  kamping-v2
  GIT_REPOSITORY https://github.com/kamping-site/kamping-v2
  GIT_TAG 7634a78a29ce0620d371dd56cb868dfd5d56d46d # fix/redistribute-empty-windows (PR #126: empty windows, 64-bit displacements; on main 0f90666); back to main once merged
  SYSTEM EXCLUDE_FROM_ALL)

set(MESSAGE_QUEUE_BLOCKING_RECEIVE ON CACHE BOOL "Receive messages blocking")
# Legacy branch of message-queue (the predecessor of BriefKAsten).
FetchContent_Declare(
  message-queue
  GIT_REPOSITORY https://github.com/niklas-uhl/message-queue
  GIT_TAG a36c73f84cac5a12d5137eb007adda2a0bbe10ec # katric-legacy
  SYSTEM)

# katric's own kassert instance (kassert::kassert, used by no other dependency). Its default level applies to every target
# linking to it; a single target can override it with the KASSERT_ASSERTION_LEVEL property.
block()
  set(KASSERT_DEFAULT_ASSERTION_LEVEL ${KASSERT_ASSERTION_LEVEL})
  FetchContent_MakeAvailable(kassert)
endblock()

# All dependencies are static libraries. Caliper declares option(BUILD_SHARED_LIBS ... TRUE), which would turn every
# dependency made available after it into a shared library: on Linux, KaGen as a shared library then fails to link
# its static, non-PIC xxHash. Caliper uses policy CMP0077, so this normal variable takes precedence over its option,
# and it also hides a stale BUILD_SHARED_LIBS=ON cache entry in existing build directories.
set(BUILD_SHARED_LIBS OFF)

# Caliper builds its own tests (and googletest) if BUILD_TESTING is on
block()
  set(BUILD_TESTING OFF)
  # Caliper's MPI wrapper generator (wrap.py) fails on Python 3.15 ("Cannot use capturing groups in re.Scanner");
  # pick an older interpreter, which Caliper's own find_package(Python) then reuses.
  find_package(Python 3.6...<3.15 COMPONENTS Interpreter REQUIRED)
  FetchContent_MakeAvailable(adiak)
  # Adiak's BLT setup redirects the output directories of *all* targets into its own build directory
  unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY CACHE)
  unset(LIBRARY_OUTPUT_PATH CACHE)
  unset(EXECUTABLE_OUTPUT_PATH CACHE)
  # Caliper's find_package(adiak) is redirected to the fetched project; it expects adiak_LIBRARIES and reads the
  # LOCATION property of its first entry, which is only allowed on imported targets.
  add_library(adiak-imported INTERFACE IMPORTED GLOBAL)
  target_link_libraries(adiak-imported INTERFACE adiak)
  file(WRITE ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/adiak-extra.cmake "set(adiak_LIBRARIES adiak-imported)\nset(adiak_INCLUDE_DIRS ${adiak_SOURCE_DIR}/include)\n")
  FetchContent_MakeAvailable(caliper)
endblock()
# the build-tree target does not export its include directories
target_include_directories(caliper SYSTEM INTERFACE $<BUILD_INTERFACE:${caliper_SOURCE_DIR}/include> $<BUILD_INTERFACE:${caliper_BINARY_DIR}/include>)

FetchContent_MakeAvailable(cli11 fmt spdlog)
if(KATRIC_FETCH_TBB)
  block()
    set(BUILD_SHARED_LIBS ON) # see the declaration of TBB
    FetchContent_MakeAvailable(TBB)
  endblock()
else()
  find_package(TBB 2021.5 CONFIG GLOBAL)
  if(NOT TBB_FOUND)
    message(FATAL_ERROR "oneTBB (>= 2021.5) not found: load the TBB module or install it (or point TBB_DIR at its CMake "
                        "config), or configure with -DKATRIC_FETCH_TBB=ON to build it from source")
  endif()
  message(STATUS "TBB: ${TBB_VERSION} (${TBB_DIR})")
endif()
FetchContent_MakeAvailable(absl magic_enum kamping kamping-spdlog-adapter kamping-v2 kagen message-queue)

# GCC on AArch64 emits a -Wpsabi note for every function passing std::pair<double, double> (ABI change in GCC 10.1).
# It is irrelevant as long as nothing is mixed with objects from GCC < 10.1, and it floods the KaGen build.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(KaGen PRIVATE -Wno-psabi)
endif()


# Tooling and app/test-only dependencies.
if(KATRIC_ENABLE_FORMAT_CMAKE)
  FetchContent_Declare(
    Format.cmake
    GIT_REPOSITORY https://github.com/TheLartians/Format.cmake
    GIT_TAG v1.8.3)
  FetchContent_MakeAvailable(Format.cmake)
endif()

if(KATRIC_BUILD_APPS)
  # Commit hash and dirty state of the working tree, embedded into the executables (git.h).
  FetchContent_Declare(
    cmake_git_version_tracking
    GIT_REPOSITORY https://github.com/niklas-uhl/cmake-git-version-tracking
    GIT_TAG v1.0.0
    GIT_SHALLOW ON
    SYSTEM EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(cmake_git_version_tracking)
endif()

# Competitors (doc/competitors.md): forks with a branch `katric` that provides a library target.
if(KATRIC_COMPETITOR_TRIC)
  FetchContent_Declare(
    tric
    GIT_REPOSITORY https://github.com/niklas-uhl/tric
    GIT_TAG f1dec08186333a765985f6d2837cca0254bdb593 # katric (on sg0/tric master 6939dbe, archived 2026-10)
    SYSTEM EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(tric)
endif()
if(KATRIC_COMPETITOR_XXTRIC)
  FetchContent_Declare(
    xxtric
    GIT_REPOSITORY https://github.com/niklas-uhl/xxtric
    GIT_TAG 45215321b44f2434d4787a24e52586ca6cf888a6 # katric (on sg0/xxtric main eb797a3)
    SYSTEM EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(xxtric)
endif()
if(KATRIC_COMPETITOR_LCC)
  # GPL-3.0 (LCC and CLaMPI): binaries built with it are GPL and not distributed (doc/competitors.md, decision 5).
  # LCC's CMake fetches CLaMPI (branch katric of niklas-uhl/CLaMPI) itself.
  FetchContent_Declare(
    lcc
    GIT_REPOSITORY https://github.com/niklas-uhl/lcc
    GIT_TAG cf0f9f16fdd395dee03dd9d91532ff43518d8030 # katric (on fvella/lcc master 42c30b0)
    SYSTEM EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(lcc)
endif()
if(KATRIC_COMPETITOR_HAVOQGT)
  FetchContent_Declare(
    havoqgt
    GIT_REPOSITORY https://github.com/niklas-uhl/havoqgt
    GIT_TAG a06e177c58e675cafb82ef0af995e9ddbf7bcd7a # katric (on llnl/havoqgt master 2e8b2a8)
    SYSTEM EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(havoqgt)
endif()
if(KATRIC_COMPETITOR_GRAPE)
  # The fork fetches glog (pinned) itself.
  FetchContent_Declare(
    grape
    GIT_REPOSITORY https://github.com/niklas-uhl/libgrape-lite
    GIT_TAG 682485d906f6e0a600723866f46728a53243cb92 # katric (on alibaba/libgrape-lite master beaa6e8)
    SYSTEM EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(grape)
endif()
if(KATRIC_COMPETITOR_TOM2D)
  FetchContent_Declare(
    tom2d
    GIT_REPOSITORY https://github.com/niklas-uhl/2DTriangleCounting
    GIT_TAG db9fbd9bb87afc878ba9099d4f1a9593f3a47b6f # katric (on KarypisLab/2DTriangleCounting master f704a53)
    SYSTEM EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(tom2d)
endif()
if(KATRIC_COMPETITOR_TRIPOLL)
  # The fork pins YGM (master) and cereal by SHA and takes spdlog and Boost from here when they exist.
  FetchContent_Declare(
    tripoll
    GIT_REPOSITORY https://github.com/niklas-uhl/tripoll
    GIT_TAG 437572c8b809e23fa1d1e4cb60872921a3e5d868 # katric (on LLNL/tripoll main ccfe8a0)
    SYSTEM EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(tripoll)
endif()

if(KATRIC_BUILD_TESTS)
  # Only register MPI tests with rank counts this machine can run, instead of oversubscribing. A cache entry, so that
  # tests/ sees it (a normal variable would only live in this directory scope).
  set(KATESTROPHE_MAX_CORES ${MPIEXEC_MAX_NUMPROCS} CACHE STRING "Skip MPI tests with more ranks than this")
  FetchContent_MakeAvailable(KaTestrophe)
endif()
