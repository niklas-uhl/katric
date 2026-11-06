# MPI tests: KaTestrophe::main initializes MPI and aggregates GoogleTest output across ranks.
function(cetric_add_mpi_test TARGET)
    cmake_parse_arguments(CETRIC_MPI_TEST "" "" "FILES;CORES" ${ARGN})
    add_executable(${TARGET} ${CETRIC_MPI_TEST_FILES})
    target_link_libraries(${TARGET} PRIVATE KaTestrophe::main cetric-lib)
    katestrophe_add_mpi_test(${TARGET} CORES ${CETRIC_MPI_TEST_CORES})
endfunction()

# Sequential tests.
function(cetric_add_test TARGET)
    cmake_parse_arguments(CETRIC_TEST "" "" "FILES" ${ARGN})
    add_executable(${TARGET} ${CETRIC_TEST_FILES})
    target_link_libraries(${TARGET} PRIVATE GTest::gmock GTest::gtest_main cetric-lib)
    gtest_discover_tests(${TARGET})
endfunction()

cetric_add_mpi_test(graph_datastructure_test FILES graph_datastructure_test.cpp CORES 2 4 8 16)
cetric_add_mpi_test(load_balancing_test FILES load_balancing_test.cpp CORES 2 4 8 16)
cetric_add_mpi_test(orientation_test FILES orientation_test.cpp CORES 4)
cetric_add_mpi_test(communicator_test FILES gather_test.cpp CORES 2 4 8 16)

cetric_add_test(rank_encoded_node_id_test FILES rank_encoded_node_id_test.cpp)
cetric_add_test(intersection_test FILES intersection_test.cpp)
