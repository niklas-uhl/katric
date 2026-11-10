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

# Sequential tests. The first launch of a freshly linked binary can take seconds on macOS (more than gtest's default
# discovery timeout of 5 s during a parallel build).
add_executable(rank_encoded_node_id_test rank_encoded_node_id_test.cpp)
target_link_libraries(rank_encoded_node_id_test PRIVATE GTest::gmock GTest::gtest_main cetric-lib cetric_cxx_flags)
gtest_discover_tests(rank_encoded_node_id_test DISCOVERY_TIMEOUT 60)

add_executable(intersection_test intersection_test.cpp)
target_link_libraries(intersection_test PRIVATE GTest::gmock GTest::gtest_main cetric-lib cetric_cxx_flags)
gtest_discover_tests(intersection_test DISCOVERY_TIMEOUT 60)

# End-to-end tests: run katric on the example graphs with each algorithm configuration and check the triangle count.
if(TARGET katric)
  set(example_graphs "triangle=1" "hiv=45" "com-amazon=667129")
  set(example_configs "cetric" "ditric" "ditric-grid" "hybrid")
  set(args_cetric --config ${PROJECT_SOURCE_DIR}/configs/cetric.toml)
  set(args_ditric --config ${PROJECT_SOURCE_DIR}/configs/ditric.toml)
  set(args_ditric-grid --config ${PROJECT_SOURCE_DIR}/configs/ditric.toml --config
                       ${PROJECT_SOURCE_DIR}/configs/2d-grid.toml)
  set(args_hybrid --config ${PROJECT_SOURCE_DIR}/configs/cetric.toml --config ${PROJECT_SOURCE_DIR}/configs/hybrid.toml
                  --num-threads 2)
  foreach(entry IN LISTS example_graphs)
    string(REPLACE "=" ";" entry "${entry}")
    list(GET entry 0 graph)
    list(GET entry 1 expected)
    foreach(config IN LISTS example_configs)
      foreach(ranks 2 4)
        if(KATESTROPHE_MAX_CORES AND ranks GREATER KATESTROPHE_MAX_CORES)
          continue()
        endif()
        set(command ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${KATESTROPHE_MPIEXEC_OVERSUBSCRIBE_FLAG}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:katric> ${PROJECT_SOURCE_DIR}/examples/${graph}.metis
                    --input-format metis ${args_${config}})
        add_test(NAME katric.${graph}.${config}.${ranks}cores
                 COMMAND ${CMAKE_COMMAND} "-DCOMMAND=${command}" -DEXPECTED=${expected} -P
                         ${CMAKE_CURRENT_SOURCE_DIR}/check_triangle_count.cmake)
      endforeach()
    endforeach()
  endforeach()
endif()
