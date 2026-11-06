include(FetchContent)
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON) # targets of system packages (FIND_PACKAGE_ARGS) must be visible outside extern/


# All dependencies are fetched here. To work on one locally, configure with
# -DFETCHCONTENT_SOURCE_DIR_<NAME>=/path/to/checkout (e.g. ..._GRAPH-IO).

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11
  GIT_TAG v2.4.2
  GIT_SHALLOW ON
  SYSTEM EXCLUDE_FROM_ALL FIND_PACKAGE_ARGS 2.4)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG 11.0.2
  GIT_SHALLOW ON
  # no FIND_PACKAGE_ARGS: the code does not compile with fmt 12 yet (e.g. Homebrew's)
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

set(GRAPH_IO_MMAP ON CACHE BOOL "use mmap")
FetchContent_Declare(
  graph-io
  GIT_REPOSITORY https://github.com/niklas-uhl/graph-io
  GIT_TAG fix-modern-toolchain
  SYSTEM)

set(MESSAGE_QUEUE_BLOCKING_RECEIVE ON CACHE BOOL "Receive messages blocking")
# Legacy branch of message-queue (BriefKAsten) which reuses the graph-io target.
FetchContent_Declare(
  message-queue
  GIT_REPOSITORY https://github.com/niklas-uhl/message-queue
  GIT_TAG katric-legacy
  SYSTEM)

# kagen has to be available before graph-io, which then reuses its target.
FetchContent_MakeAvailable(cli11 fmt cereal magic_enum kassert tlx sparsehash utils_tm kagen graph-io message-queue)

add_library(utils_tm INTERFACE)
target_include_directories(utils_tm INTERFACE ${utils_tm_SOURCE_DIR})
