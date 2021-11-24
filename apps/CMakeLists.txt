add_executable(cetric cetric.cpp)

target_link_libraries(cetric PRIVATE cetric-lib CLI11::CLI11 MPI::MPI_CXX)
#target_link_libraries(cetric PUBLIC ${BACKWARD_ENABLE})
target_link_libraries(cetric PUBLIC ${BACKWARD_LIBRARIES})
target_compile_definitions(cetric PUBLIC ${BACKWARD_DEFINITIONS})

