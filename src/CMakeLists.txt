# The algorithm, one translation unit per communication policy, so they compile in parallel and an edit to the driver
# does not recompile the other.
add_library(katric-algorithms STATIC katric_queue.cpp katric_grid.cpp)
target_link_libraries(katric-algorithms PUBLIC katric-lib PRIVATE katric_cxx_flags)
set_target_properties(katric-algorithms PROPERTIES CXX_CLANG_TIDY "${KATRIC_CLANG_TIDY_COMMAND}")
