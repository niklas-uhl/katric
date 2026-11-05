add_library(mpi-gtest-main EXCLUDE_FROM_ALL mpi-gtest-main.cpp)
target_link_libraries(mpi-gtest-main PUBLIC gtest gtest-mpi-listener gmock)
target_include_directories(mpi-gtest-main PUBLIC ${BACKWARD_INCLUDE_DIRS})
target_link_libraries(mpi-gtest-main PUBLIC ${BACKWARD_LIBRARIES})
target_compile_definitions(mpi-gtest-main PUBLIC ${BACKWARD_DEFINITIONS})

add_library(gtest-main-with-backward EXCLUDE_FROM_ALL gtest-main-with-backward.cpp)
target_link_libraries(gtest-main-with-backward PUBLIC gtest gmock)
target_include_directories(gtest-main-with-backward PUBLIC ${BACKWARD_INCLUDE_DIRS})
target_link_libraries(gtest-main-with-backward PUBLIC ${BACKWARD_LIBRARIES})
target_compile_definitions(gtest-main-with-backward PUBLIC ${BACKWARD_DEFINITIONS})

function(nuhl_add_mpi_test_executable)
    cmake_parse_arguments(
        NUHL_ADD_MPI_TEST
        ""
        "TARGET"
        "FILES"
        ${ARGN}
        )
    add_executable(${NUHL_ADD_MPI_TEST_TARGET} "${NUHL_ADD_MPI_TEST_FILES}")
    target_link_libraries(${NUHL_ADD_MPI_TEST_TARGET} PUBLIC mpi-gtest-main)
endfunction()

function(nuhl_add_mpi_test NUHL_ADD_TEST_TARGET)
    cmake_parse_arguments(
        NUHL_ADD_TEST
        ""
        ""
        "CORES"
        ${ARGN}
        )
    if(NOT NUHL_MPI_TEST_CORES)
        set(NUHL_MPI_TEST_CORES ${MPIEXEC_MAX_NUMPROCS})
    endif()
    foreach(p ${NUHL_ADD_TEST_CORES})
        if(${p} LESS_EQUAL ${MPIEXEC_MAX_NUMPROCS})
            set(TEST_NAME "${NUHL_ADD_TEST_TARGET}_${p}cores")
            add_test(NAME "${TEST_NAME}" COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${p} $<TARGET_FILE:${NUHL_ADD_TEST_TARGET}> WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        endif()
    endforeach()
endfunction()

nuhl_add_mpi_test_executable(TARGET graph_datastructure_test FILES graph_datastructure_test.cpp)
target_link_libraries(graph_datastructure_test PRIVATE cetric-lib)
nuhl_add_mpi_test(graph_datastructure_test CORES 2 4 8 16)

nuhl_add_mpi_test_executable(TARGET load_balancing_test FILES load_balancing_test.cpp)
target_link_libraries(load_balancing_test PRIVATE cetric-lib)
nuhl_add_mpi_test(load_balancing_test CORES 2 4 8 16)

nuhl_add_mpi_test_executable(TARGET orientation_test FILES orientation_test.cpp)
target_link_libraries(orientation_test PRIVATE cetric-lib)
nuhl_add_mpi_test(orientation_test CORES 4)

nuhl_add_mpi_test_executable(TARGET communicator_test FILES gather_test.cpp)
target_link_libraries(communicator_test PRIVATE cetric-lib)
nuhl_add_mpi_test(communicator_test CORES 2 4 8 16)

add_executable(rank_encoded_node_id_test rank_encoded_node_id_test.cpp)
target_link_libraries(rank_encoded_node_id_test PRIVATE gtest-main-with-backward cetric-lib)
gtest_discover_tests(rank_encoded_node_id_test)

add_executable(intersection_test intersection_test.cpp)
target_link_libraries(intersection_test PRIVATE gtest-main-with-backward cetric-lib)
gtest_discover_tests(intersection_test)
