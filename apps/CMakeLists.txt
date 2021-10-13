add_executable(cetric cetric.cpp)

target_link_libraries(cetric PRIVATE cetric-lib CLI11::CLI11 MPI::MPI_CXX)
