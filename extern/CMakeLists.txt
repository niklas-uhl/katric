add_subdirectory(cli11)
add_subdirectory(backward-cpp EXCLUDE_FROM_ALL)
add_subdirectory(fmt EXCLUDE_FROM_ALL)
add_subdirectory(cereal EXCLUDE_FROM_ALL)
add_subdirectory(magic_enum EXCLUDE_FROM_ALL)
#add_subdirectory(bytehamster_util EXCLUDE_FROM_ALL)
#add_subdirectory(kagen)
add_subdirectory(kassert)
set(MESSAGE_QUEUE_BLOCKING_RECEIVE ON CACHE BOOL "Receive messages blocking")
set(GRAPH_IO_MMAP ON CACHE BOOL "use mmap")

# graph-io is fetched here (instead of the copy nested in message-queue).
# For local development use -DFETCHCONTENT_SOURCE_DIR_GRAPH-IO=/path/to/graph-io
include(FetchContent)
FetchContent_Declare(
  graph-io
  GIT_REPOSITORY https://github.com/niklas-uhl/graph-io
  GIT_TAG        fix-modern-toolchain
  GIT_SUBMODULES_RECURSE TRUE
)
FetchContent_MakeAvailable(graph-io)

if(NOT TARGET tlx)
  add_subdirectory(tlx EXCLUDE_FROM_ALL)
endif()

# Pinned to a legacy branch which reuses the graph-io target provided above.
# For local development use -DFETCHCONTENT_SOURCE_DIR_MESSAGE-QUEUE=/path/to/message-queue
FetchContent_Declare(
  message-queue
  GIT_REPOSITORY https://github.com/niklas-uhl/message-queue
  GIT_TAG        katric-legacy
)
FetchContent_MakeAvailable(message-queue)

add_library(utils_tm INTERFACE)
target_include_directories(utils_tm INTERFACE utils_tm)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/extern/sanitizers-cmake/cmake")
find_package(Sanitizers)
