add_subdirectory(spooky)
add_subdirectory(cli11)
add_subdirectory(debug_assert)
add_subdirectory(backward-cpp EXCLUDE_FROM_ALL)
add_subdirectory(fmt EXCLUDE_FROM_ALL)
add_subdirectory(cereal EXCLUDE_FROM_ALL)
add_subdirectory(catch EXCLUDE_FROM_ALL)
add_subdirectory(googletest EXCLUDE_FROM_ALL)

set(gtest-mpi-listener_SOURCE_DIR ${CMAKE_SOURCE_DIR}/extern/gtest-mpi-listener)
add_library(gtest-mpi-listener INTERFACE)
target_include_directories(gtest-mpi-listener INTERFACE "${gtest-mpi-listener_SOURCE_DIR}/include")
target_link_libraries(gtest-mpi-listener INTERFACE MPI::MPI_CXX gtest gmock)

find_package(CGAL)
set(kagen_SOURCE_DIR kagen)
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
