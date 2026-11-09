add_executable(katric katric.cpp git_hash.cpp)
target_link_libraries(katric PRIVATE cetric-lib cetric_cxx_flags CLI11::CLI11 kamping::spdlog-adapter
                                     cmake_git_version_tracking)

add_executable(shmetric shmetric.cpp git_hash.cpp)
target_link_libraries(shmetric PRIVATE cetric-lib cetric_cxx_flags CLI11::CLI11 cmake_git_version_tracking)
