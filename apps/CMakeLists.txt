add_executable(cetric cetric.cpp)

target_link_libraries(cetric PRIVATE cetric-lib CLI11::CLI11 MPI::MPI_CXX)
#target_link_libraries(cetric PUBLIC ${BACKWARD_ENABLE})
target_link_libraries(cetric PUBLIC -ldw -ldl)
target_compile_definitions(cetric PUBLIC BACKWARD_HAS_DW)

