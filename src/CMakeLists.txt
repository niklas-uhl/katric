# The algorithm in its own translation unit, so the harness does not instantiate the phases.
add_library(katric-algorithms STATIC katric.cpp)
target_link_libraries(katric-algorithms PUBLIC katric-lib PRIVATE katric_cxx_flags)
set_target_properties(katric-algorithms PROPERTIES CXX_CLANG_TIDY "${KATRIC_CLANG_TIDY_COMMAND}")
