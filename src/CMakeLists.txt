# The algorithms, one translation unit per algorithm and communication policy, so they compile in parallel and an
# edit to the driver does not recompile the edge iterator.
add_library(
  cetric-algorithms STATIC
  patric_queue.cpp
  patric_grid.cpp
  cetric_queue.cpp
  cetric_grid.cpp
  cetric_x_queue.cpp
  cetric_x_grid.cpp)
target_link_libraries(cetric-algorithms PUBLIC cetric-lib PRIVATE cetric_cxx_flags)
set_target_properties(cetric-algorithms PROPERTIES CXX_CLANG_TIDY "${CETRIC_CLANG_TIDY_COMMAND}")
