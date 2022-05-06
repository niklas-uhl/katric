add_executable(cetric cetric.cpp)
add_executable(shared_memory_counter shared_memory_counter.cpp)

target_link_libraries(cetric PRIVATE cetric-lib CLI11::CLI11 MPI::MPI_CXX)
#target_link_libraries(cetric PUBLIC ${BACKWARD_ENABLE})
target_link_libraries(cetric PUBLIC ${BACKWARD_LIBRARIES})
target_compile_definitions(cetric PUBLIC ${BACKWARD_DEFINITIONS})

target_link_libraries(shared_memory_counter PRIVATE cetric-lib CLI11::CLI11)
target_link_libraries(cetric PUBLIC ${BACKWARD_LIBRARIES})
target_compile_definitions(cetric PUBLIC ${BACKWARD_DEFINITIONS})
