FetchContent_Declare(
  catch
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.7)

FetchContent_MakeAvailable(catch)

set(TEST_MAIN test_main.cpp)
set(TEST_MAIN_MPI test_main_mpi.cpp)

function(nuhl_add_mpi_test)
    cmake_parse_arguments(
        NUHL_ADD_MPI_TEST
        ""
        "TARGET;CORES"
        "FILES"
        ${ARGN}
        )
    if(NOT ${NUHL_ADD_MPI_TEST_CORES}) 
        set(NUHL_ADD_MPI_TEST_CORES ${MPIEXEC_MAX_NUMPROCS})
    endif()

    add_executable(${NUHL_ADD_MPI_TEST_TARGET} ${TEST_MAIN_MPI} "${NUHL_ADD_MPI_TEST_FILES}")
    target_link_libraries(${NUHL_ADD_MPI_TEST_TARGET} PUBLIC MPI::MPI_CXX Catch2::Catch2)
    add_test(NAME ${NUHL_ADD_MPI_TEST_TARGET} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NUHL_ADD_MPI_TEST_CORES} $<TARGET_FILE:${NUHL_ADD_MPI_TEST_TARGET}> WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endfunction()

nuhl_add_mpi_test(TARGET graph_datastructure_test FILES graph_datastructure_test.cpp CORES 4)
target_link_libraries(graph_datastructure_test PRIVATE cetric-lib)
