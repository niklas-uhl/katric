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
  GIT_TAG 11.0.2
  GIT_SHALLOW ON
  # no FIND_PACKAGE_ARGS: the code does not compile with fmt 12 yet (e.g. Homebrew's)
  SYSTEM EXCLUDE_FROM_ALL)

# spdlog uses the (header-only) fmt fetched above
set(SPDLOG_FMT_EXTERNAL_HO ON)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog
  GIT_TAG v1.16.0
  GIT_SHALLOW ON
  SYSTEM EXCLUDE_FROM_ALL)

set(JUST_INSTALL_CEREAL ON)
FetchContent_Declare(
  cereal
  GIT_REPOSITORY https://github.com/USCiLab/cereal
  GIT_TAG v1.3.2
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
  GIT_TAG v1.0.0
  GIT_SHALLOW ON
  SYSTEM EXCLUDE_FROM_ALL)

FetchContent_Declare(
  tlx
  GIT_REPOSITORY https://github.com/tlx/tlx
  # newer than v0.6.1: raises the too-old cmake_minimum_required (needed for CMake >= 4)
  GIT_TAG 2dd63ab839909f0b43cd02108a1869c4670b2c8e
  SYSTEM EXCLUDE_FROM_ALL)

FetchContent_Declare(
  sparsehash
  GIT_REPOSITORY https://github.com/niklas-uhl/sparsehash-cmake
  GIT_TAG 06457c996e01f4f222f6e9f9f4be088ce1e3901c # master
  SYSTEM EXCLUDE_FROM_ALL)

# header-only, no releases: pinned to a commit
FetchContent_Declare(
  utils_tm
  GIT_REPOSITORY https://github.com/toobiased/utils_tm
  GIT_TAG 40b3893ee8dc4896aca2313cbd8ac3c4551dc1de
  SYSTEM EXCLUDE_FROM_ALL)

# oneTBB is usually provided by the system (module, package manager): find_package is tried first, and it is only
# fetched if that fails. Force fetching with -DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER.
set(TBB_TEST OFF)
set(TBB_EXAMPLES OFF)
set(TBB_STRICT OFF) # do not build TBB itself with -Werror
set(TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH ON) # tbbbind (NUMA support) is not needed
FetchContent_Declare(
  TBB
  GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB
  GIT_TAG v2023.2.0
  GIT_SHALLOW ON
  SYSTEM EXCLUDE_FROM_ALL
  FIND_PACKAGE_ARGS 2021.5 CONFIG)

# GoogleTest + MPI test helpers (only downloaded if tests are built, see the end of this file). Declared before
# kamping-v2, which declares its own (older) KaTestrophe: the first declaration wins.
FetchContent_Declare(
  KaTestrophe
  GIT_REPOSITORY https://github.com/kamping-site/KaTestrophe.cmake
  GIT_TAG v1.2.0
  GIT_SHALLOW ON
  SYSTEM EXCLUDE_FROM_ALL)

# KaGen is not used by katric directly yet, but it is declared here so that
# graph-io (and later katric itself) build against exactly this version.
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
# spdlog adapter). Pinned to a commit so that kamping-v2 does not fetch an unpinned `main` of kamping-types; kamping-v2
# and kamping-spdlog-adapter reuse these targets instead of fetching their own copies.
set(KAMPING_ENABLE_REFLECTION OFF)
set(KAMPING_ENABLE_SERIALIZATION OFF)
FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping
  GIT_TAG 001f21510a7edef33248a9d4063073a7b05b5696 # main
  SYSTEM EXCLUDE_FROM_ALL)

# MPI-aware logging setup (rank pattern, root-only and gather sinks) on top of spdlog.
FetchContent_Declare(
  kamping-spdlog-adapter
  GIT_REPOSITORY https://github.com/kamping-site/kamping-spdlog-adapter
  GIT_TAG v1.0.0
  GIT_SHALLOW ON
  SYSTEM EXCLUDE_FROM_ALL)

FetchContent_Declare(
  kamping-v2
  GIT_REPOSITORY https://github.com/kamping-site/kamping-v2
  GIT_TAG 21e1c3030592ea7f9e3e1b25c7edac1e2dea6dfc # main
  SYSTEM EXCLUDE_FROM_ALL)

set(GRAPH_IO_MMAP ON CACHE BOOL "use mmap")
FetchContent_Declare(
  graph-io
  GIT_REPOSITORY https://github.com/niklas-uhl/graph-io
  GIT_TAG 44a1cd56b61f975a77289576619d61730c723b63 # fix-modern-toolchain
  SYSTEM)

set(MESSAGE_QUEUE_BLOCKING_RECEIVE ON CACHE BOOL "Receive messages blocking")
# Legacy branch of message-queue (BriefKAsten) which reuses the graph-io target.
FetchContent_Declare(
  message-queue
  GIT_REPOSITORY https://github.com/niklas-uhl/message-queue
  GIT_TAG 57b0b8a6129a78dc277a992b04f818038c9669f8 # katric-legacy
  SYSTEM)

# kagen has to be available before graph-io, which then reuses its target.
FetchContent_MakeAvailable(cli11 fmt spdlog TBB cereal magic_enum kassert tlx sparsehash utils_tm kamping kamping-spdlog-adapter kamping-v2 kagen graph-io message-queue)

add_library(utils_tm INTERFACE)
target_include_directories(utils_tm INTERFACE ${utils_tm_SOURCE_DIR})

# Tooling and app/test-only dependencies.
if(CETRIC_ENABLE_FORMAT_CMAKE)
  FetchContent_Declare(
    Format.cmake
    GIT_REPOSITORY https://github.com/TheLartians/Format.cmake
    GIT_TAG v1.8.3)
  FetchContent_MakeAvailable(Format.cmake)
endif()

if(CETRIC_BUILD_APPS)
  # Commit hash and dirty state of the working tree, embedded into the executables (git.h).
  FetchContent_Declare(
    cmake_git_version_tracking
    GIT_REPOSITORY https://github.com/niklas-uhl/cmake-git-version-tracking
    GIT_TAG v1.0.0
    GIT_SHALLOW ON
    SYSTEM EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(cmake_git_version_tracking)
endif()

if(CETRIC_BUILD_TESTS)
  # Only register MPI tests with rank counts this machine can run, instead of oversubscribing. A cache entry, so that
  # tests/ sees it (a normal variable would only live in this directory scope).
  set(KATESTROPHE_MAX_CORES ${MPIEXEC_MAX_NUMPROCS} CACHE STRING "Skip MPI tests with more ranks than this")
  FetchContent_MakeAvailable(KaTestrophe)
endif()
