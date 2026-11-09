include(GoogleTest)

# MPI tests: KaTestrophe::main initializes MPI and aggregates GoogleTest output across ranks. Rank counts above
# KATESTROPHE_MAX_CORES (set in extern/) are skipped.
add_executable(graph_datastructure_test graph_datastructure_test.cpp)
target_link_libraries(graph_datastructure_test PRIVATE KaTestrophe::main cetric-lib cetric_cxx_flags)
katestrophe_add_mpi_test(graph_datastructure_test CORES 2 4 8 16)

add_executable(load_balancing_test load_balancing_test.cpp)
target_link_libraries(load_balancing_test PRIVATE KaTestrophe::main cetric-lib cetric_cxx_flags)
katestrophe_add_mpi_test(load_balancing_test CORES 2 4 8 16)

add_executable(orientation_test orientation_test.cpp)
target_link_libraries(orientation_test PRIVATE KaTestrophe::main cetric-lib cetric_cxx_flags)
katestrophe_add_mpi_test(orientation_test CORES 4)

add_executable(communicator_test gather_test.cpp)
target_link_libraries(communicator_test PRIVATE KaTestrophe::main cetric-lib cetric_cxx_flags)
katestrophe_add_mpi_test(communicator_test CORES 2 4 8 16)

# Sequential tests.
add_executable(rank_encoded_node_id_test rank_encoded_node_id_test.cpp)
target_link_libraries(rank_encoded_node_id_test PRIVATE GTest::gmock GTest::gtest_main cetric-lib cetric_cxx_flags)
gtest_discover_tests(rank_encoded_node_id_test)

add_executable(intersection_test intersection_test.cpp)
target_link_libraries(intersection_test PRIVATE GTest::gmock GTest::gtest_main cetric-lib cetric_cxx_flags)
gtest_discover_tests(intersection_test)
