add_subdirectory(spooky)
add_subdirectory(cli11)
add_subdirectory(backward-cpp EXCLUDE_FROM_ALL)
add_subdirectory(fmt EXCLUDE_FROM_ALL)
add_subdirectory(cereal EXCLUDE_FROM_ALL)
add_subdirectory(magic_enum EXCLUDE_FROM_ALL)
add_subdirectory(catch)
#add_subdirectory(bytehamster_util EXCLUDE_FROM_ALL)
if(NOT TARGET gtest)
  add_subdirectory(googletest EXCLUDE_FROM_ALL)
endif()
#add_subdirectory(kagen)
add_subdirectory(kassert)
set(MESSAGE_QUEUE_BLOCKING_RECEIVE ON CACHE BOOL "Receive messages blocking")
set(GRAPH_IO_MMAP ON CACHE BOOL "use mmap")
add_subdirectory(message-queue)
add_library(utils_tm INTERFACE)
target_include_directories(utils_tm INTERFACE utils_tm)

set(gtest-mpi-listener_SOURCE_DIR ${CMAKE_SOURCE_DIR}/extern/gtest-mpi-listener)
add_library(gtest-mpi-listener INTERFACE)
target_include_directories(gtest-mpi-listener INTERFACE "${gtest-mpi-listener_SOURCE_DIR}/include")
target_link_libraries(gtest-mpi-listener INTERFACE MPI::MPI_CXX gtest gmock)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/extern/sanitizers-cmake/cmake")
find_package(Sanitizers)
