# The algorithm, one translation unit per communication policy, so they compile in parallel and an edit to the driver
# does not recompile the other.
add_library(cetric-algorithms STATIC katric_queue.cpp katric_grid.cpp)
target_link_libraries(cetric-algorithms PUBLIC cetric-lib PRIVATE cetric_cxx_flags)
set_target_properties(cetric-algorithms PROPERTIES CXX_CLANG_TIDY "${CETRIC_CLANG_TIDY_COMMAND}")
