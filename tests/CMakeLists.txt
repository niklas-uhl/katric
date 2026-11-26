include(GoogleTest)

# MPI tests: KaTestrophe::main initializes MPI and aggregates GoogleTest output across ranks. Rank counts above
# KATESTROPHE_MAX_CORES (set in extern/) are skipped.
add_executable(graph_datastructure_test graph_datastructure_test.cpp)
target_link_libraries(graph_datastructure_test PRIVATE KaTestrophe::main katric-lib katric_cxx_flags)
katestrophe_add_mpi_test(graph_datastructure_test CORES 2 4 8 16)

add_executable(load_balancing_test load_balancing_test.cpp)
target_link_libraries(load_balancing_test PRIVATE KaTestrophe::main katric-lib katric_cxx_flags)
katestrophe_add_mpi_test(load_balancing_test CORES 2 4 8 16)

add_executable(orientation_test orientation_test.cpp)
target_link_libraries(orientation_test PRIVATE KaTestrophe::main katric-lib katric_cxx_flags)
katestrophe_add_mpi_test(orientation_test CORES 4)

# Sequential tests. The first launch of a freshly linked binary can take seconds on macOS (more than gtest's default
# discovery timeout of 5 s during a parallel build).
add_executable(intersection_test intersection_test.cpp)
target_link_libraries(intersection_test PRIVATE GTest::gmock GTest::gtest_main katric-lib katric_cxx_flags)
gtest_discover_tests(intersection_test DISCOVERY_TIMEOUT 60)

set_target_properties(
  graph_datastructure_test load_balancing_test orientation_test intersection_test PROPERTIES CXX_CLANG_TIDY "${KATRIC_CLANG_TIDY_COMMAND}")

# End-to-end tests: run katric on the example graphs with each algorithm configuration and check the triangle count
# (against the expected value here and, via --verify, against the sequential reference count).
if(TARGET katric)
  # Inputs as KaGen option strings ($<SEMICOLON>: a plain ; would split the CMake list). Files are read edge-balanced
  # (formerly read-edge-partitioned in the configs). rgg2d gives the same graph for 2 and 4 ranks; KaGen's GNM and RHG
  # graphs depend on the number of PEs.
  set(example_graphs "triangle=1" "hiv=45" "com-amazon=667129" "rgg2d=413610")
  foreach(graph triangle hiv com-amazon)
    set(kagen_${graph} "file$<SEMICOLON>filename=${PROJECT_SOURCE_DIR}/examples/${graph}.metis$<SEMICOLON>input_format=metis$<SEMICOLON>distribution=balance-edges")
  endforeach()
  set(kagen_rgg2d "rgg2d$<SEMICOLON>n=16384$<SEMICOLON>m=131072$<SEMICOLON>seed=7")
  set(example_configs "cetric" "ditric" "ditric-grid" "ditric-dense-exchange" "cetric-grid-exchange"
      "ditric-trimming" "cetric-trimming")
  set(args_cetric --config ${PROJECT_SOURCE_DIR}/configs/cetric.toml)
  set(args_ditric --config ${PROJECT_SOURCE_DIR}/configs/ditric.toml)
  set(args_ditric-grid --config ${PROJECT_SOURCE_DIR}/configs/ditric.toml --config
                       ${PROJECT_SOURCE_DIR}/configs/2d-grid.toml)
  set(args_ditric-dense-exchange --config ${PROJECT_SOURCE_DIR}/configs/ditric.toml --degree-exchange dense)
  set(args_cetric-grid-exchange --config ${PROJECT_SOURCE_DIR}/configs/cetric.toml --degree-exchange grid)
  set(args_ditric-trimming --config ${PROJECT_SOURCE_DIR}/configs/ditric.toml --message-trimming)
  set(args_cetric-trimming --config ${PROJECT_SOURCE_DIR}/configs/cetric.toml --message-trimming)
  # competitors (doc/competitors.md), each only if compiled in. Their tests get a timeout of their own, so that a
  # competitor that hangs (TriC's buffered variants did on tiny inputs) fails fast instead of after ctest's 1500 s.
  # The slowest so far takes ~75 s in a Debug build (com-amazon, tric hash2 with sparse ranges, 2 ranks, ctest -j4).
  set(KATRIC_COMPETITOR_TEST_TIMEOUT 180 CACHE STRING "Timeout in seconds of each competitor ctest")
  set(competitor_configs "")
  if(KATRIC_COMPETITOR_TRIC)
    foreach(variant plain buffered hash2 inrecv)
      list(APPEND example_configs tric-${variant})
      list(APPEND competitor_configs tric-${variant})
      set(args_tric-${variant} --algorithm tric --tric-variant ${variant} --iterations 2)
    endforeach()
    foreach(variant buffered hash2)
      list(APPEND example_configs tric-${variant}-sparse-ranges)
      list(APPEND competitor_configs tric-${variant}-sparse-ranges)
      set(args_tric-${variant}-sparse-ranges --algorithm tric --tric-variant ${variant} --tric-sparse-ranges
                                             --iterations 2)
    endforeach()
  endif()
  if(KATRIC_COMPETITOR_LCC)
    # small CLaMPI caches (the default allocates 8 GiB per rank up front)
    list(APPEND example_configs lcc lcc-degree-reduction)
    list(APPEND competitor_configs lcc lcc-degree-reduction)
    set(args_lcc --algorithm lcc --lcc-cache-size 67108864 --iterations 2)
    set(args_lcc-degree-reduction --algorithm lcc --lcc-cache-size 67108864 --lcc-degree-reduction --iterations 2)
  endif()
  if(KATRIC_COMPETITOR_XXTRIC)
    list(APPEND example_configs xxtric xxtric-buffer-size)
    list(APPEND competitor_configs xxtric xxtric-buffer-size)
    set(args_xxtric --algorithm xxtric --iterations 2)
    set(args_xxtric-buffer-size --algorithm xxtric --xxtric-buffer-size 8000 --iterations 2)
  endif()
  if(KATRIC_COMPETITOR_HAVOQGT)
    list(APPEND example_configs havoqgt havoqgt-stats)
    list(APPEND competitor_configs havoqgt havoqgt-stats)
    set(args_havoqgt --algorithm havoqgt --iterations 2)
    set(args_havoqgt-stats --algorithm havoqgt --havoqgt-stats)
  endif()
  if(KATRIC_COMPETITOR_TRIPOLL)
    list(APPEND example_configs tripoll-push-only tripoll-push-pull tripoll-nlnr)
    list(APPEND competitor_configs tripoll-push-only tripoll-push-pull tripoll-nlnr)
    set(args_tripoll-push-only --algorithm tripoll --tripoll-variant push-only --iterations 2)
    set(args_tripoll-push-pull --algorithm tripoll --tripoll-variant push-pull --iterations 2)
    set(args_tripoll-nlnr --algorithm tripoll --tripoll-routing nlnr)
  endif()
  # the hybrid configuration (configs/hybrid.toml, --num-threads > 1) is disabled until the hybrid path is re-ported
  # (doc/refactoring-plan.md)
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
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:katric> --verify ${args_${config}})
        add_test(NAME katric.${graph}.${config}.${ranks}cores
                 COMMAND ${CMAKE_COMMAND} "-DCOMMAND=${command}" "-DKAGEN_OPTION_STRING=${kagen_${graph}}"
                         -DEXPECTED=${expected} -P ${CMAKE_CURRENT_SOURCE_DIR}/check_triangle_count.cmake)
        if(config IN_LIST competitor_configs)
          set_tests_properties(katric.${graph}.${config}.${ranks}cores PROPERTIES TIMEOUT
                                                                                 ${KATRIC_COMPETITOR_TEST_TIMEOUT})
          # one OpenMP thread per rank (LCC's kernel uses OpenMP); the ranks already use the cores
          set_tests_properties(katric.${graph}.${config}.${ranks}cores PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1)
        endif()
      endforeach()
    endforeach()
  endforeach()
endif()
