add_executable(katric katric.cpp git_hash.cpp)
target_link_libraries(katric PRIVATE cetric-lib cetric_cxx_flags CLI11::CLI11 kamping::spdlog-adapter
                                     cmake_git_version_tracking KaGen::KaGen)

set_target_properties(katric PROPERTIES CXX_CLANG_TIDY "${CETRIC_CLANG_TIDY_COMMAND}")
