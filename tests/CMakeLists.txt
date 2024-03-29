add_library(mpi-gtest-main EXCLUDE_FROM_ALL mpi-gtest-main.cpp)
target_link_libraries(mpi-gtest-main PUBLIC gtest gtest-mpi-listener gmock)
target_include_directories(mpi-gtest-main PUBLIC ${BACKWARD_INCLUDE_DIRS})
target_link_libraries(mpi-gtest-main PUBLIC ${BACKWARD_LIBRARIES})
target_compile_definitions(mpi-gtest-main PUBLIC ${BACKWARD_DEFINITIONS})

set(TEST_MAIN_MPI mpi-catch2-main.cpp)
add_library(mpi-catch2-main EXCLUDE_FROM_ALL ${TEST_MAIN_MPI})
target_link_libraries(mpi-catch2-main PUBLIC MPI::MPI_CXX Catch2::Catch2)

function(nuhl_add_mpi_test_executable)
    cmake_parse_arguments(
        NUHL_ADD_MPI_TEST
        "CATCH;GTEST"
        "TARGET"
        "FILES"
        ${ARGN}
        )
    if(NOT DEFINED NUHL_ADD_MPI_TEST_CATCH AND NOT DEFINED NUHL_ADD_MPI_TEST_GTEST)
        set(NUHL_ADD_MPI_TEST_CATCH TRUE)
    endif()

    add_executable(${NUHL_ADD_MPI_TEST_TARGET} "${NUHL_ADD_MPI_TEST_FILES}")
    if (NUHL_ADD_MPI_TEST_CATCH)
        target_link_libraries(${NUHL_ADD_MPI_TEST_TARGET} PUBLIC mpi-catch2-main)
    elseif(NUHL_ADD_MPI_TEST_GTEST)
        target_link_libraries(${NUHL_ADD_MPI_TEST_TARGET} PUBLIC mpi-gtest-main)
    endif()
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

nuhl_add_mpi_test_executable(GTEST TARGET graph_datastructure_test FILES graph_datastructure_test.cpp)
target_link_libraries(graph_datastructure_test PRIVATE cetric-lib)
nuhl_add_mpi_test(graph_datastructure_test CORES 2 4 8 16)

nuhl_add_mpi_test_executable(GTEST TARGET load_balancing_test FILES load_balancing_test.cpp)
target_link_libraries(load_balancing_test PRIVATE cetric-lib)
nuhl_add_mpi_test(load_balancing_test CORES 2 4 8 16)

nuhl_add_mpi_test_executable(GTEST TARGET orientation_test FILES orientation_test.cpp)
target_link_libraries(orientation_test PRIVATE cetric-lib)
nuhl_add_mpi_test(orientation_test CORES 4)

nuhl_add_mpi_test_executable(GTEST TARGET communicator_test FILES gather_test.cpp)
target_link_libraries(communicator_test PRIVATE cetric-lib)
nuhl_add_mpi_test(communicator_test CORES 2 4 8 16)

add_executable(rank_encoded_node_id_test rank_encoded_node_id_test.cpp)
target_link_libraries(rank_encoded_node_id_test PRIVATE Catch2::Catch2WithMain cetric-lib)

add_library(catch2_main_with_backward EXCLUDE_FROM_ALL catch2_main_with_backward.cpp)
target_link_libraries(catch2_main_with_backward PUBLIC Catch2::Catch2)
target_include_directories(catch2_main_with_backward PUBLIC ${BACKWARD_INCLUDE_DIRS})
target_link_libraries(catch2_main_with_backward PUBLIC ${BACKWARD_LIBRARIES})
target_compile_definitions(catch2_main_with_backward PUBLIC ${BACKWARD_DEFINITIONS})

add_executable(intersection_test intersection_test.cpp)
target_link_libraries(intersection_test PRIVATE catch2_main_with_backward cetric-lib)
