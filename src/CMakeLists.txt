file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${parallel-triangle-counter_SOURCE_DIR}/include/[a-zA-Z_0-9]+.h")

set(FILE_LIST)
if (NOT FILE_LIST)
    set(DUMMY_FILE ${CMAKE_CURRENT_BINARY_DIR}/null.cpp)
    file(WRITE "${DUMMY_FILE}" "")
    list(APPEND FILE_LIST "${DUMMY_FILE}")
endif()


add_library(cetric-lib ${FILE_LIST} ${HEADER_LIST})

target_include_directories(cetric-lib PUBLIC ${PROJECT_SOURCE_DIR}/include/)

target_link_libraries(cetric-lib PRIVATE local_spooky)
target_link_libraries(cetric-lib PUBLIC MPI::MPI_CXX)
target_link_libraries(cetric-lib PUBLIC tlx)
target_link_libraries(cetric-lib PUBLIC Sparsehash::Sparsehash)
target_link_libraries(cetric-lib PUBLIC cereal::cereal)
target_link_libraries(cetric-lib PUBLIC magic_enum::magic_enum)
target_link_libraries(cetric-lib PUBLIC fmt::fmt)
target_link_libraries(cetric-lib PUBLIC message-queue)
target_link_libraries(cetric-lib PUBLIC graph-io)
target_link_libraries(cetric-lib PUBLIC TBB::tbb)
target_link_libraries(cetric-lib PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(cetric-lib PUBLIC kassert::kassert)
target_link_libraries(cetric-lib PUBLIC utils_tm)
target_link_libraries(cetric-lib PUBLIC Boost::headers Boost::mpi)
target_include_directories(cetric-lib PUBLIC ${BACKWARD_INCLUDE_DIRS})
target_link_libraries(cetric-lib PUBLIC ${BACKWARD_LIBRARIES})
target_compile_definitions(cetric-lib PUBLIC ${BACKWARD_DEFINITIONS})

option(CETRIC_BINARY_INTERSECTION_RECURSIVE "Use recursion in binary search intersection" OFF)
if (CETRIC_BINARY_INTERSECTION_RECURSIVE)
  target_compile_definitions(cetric-lib PUBLIC -DCETRIC_BINARY_INTERSECTION_RECURSIVE)
endif()

target_compile_options(cetric-lib PUBLIC -Wall -Wextra -pedantic)
add_sanitizers(cetric-lib)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)
