file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${parallel-triangle-counter_SOURCE_DIR}/include/[a-zA-Z_0-9]+.h")

set(FILE_LIST
  shared_memory_edge_iterator.cpp
)

add_library(cetric-lib ${FILE_LIST} ${HEADER_LIST})

target_include_directories(cetric-lib PUBLIC ${PROJECT_SOURCE_DIR}/include/)

target_link_libraries(cetric-lib PRIVATE local_spooky)
target_link_libraries(cetric-lib PUBLIC MPI::MPI_CXX)
target_link_libraries(cetric-lib PRIVATE libkagen)
target_link_libraries(cetric-lib PUBLIC tlx)
target_link_libraries(cetric-lib PUBLIC sparsehash)
target_link_libraries(cetric-lib PUBLIC cereal::cereal)
target_link_libraries(cetric-lib PUBLIC debug_assert)
target_link_libraries(cetric-lib PUBLIC fmt::fmt)
target_link_libraries(cetric-lib PUBLIC message-queue)
target_link_libraries(cetric-lib PUBLIC graph-io)
target_link_libraries(cetric-lib PUBLIC TBB::tbb)
target_link_libraries(cetric-lib PUBLIC kassert::kassert)
target_link_libraries(cetric-lib PUBLIC range_adaptor)
target_link_libraries(cetric-lib PUBLIC utils_tm)
target_include_directories(cetric-lib PUBLIC ${BACKWARD_INCLUDE_DIRS})
target_link_libraries(cetric-lib PUBLIC ${BACKWARD_LIBRARIES})
target_compile_definitions(cetric-lib PUBLIC ${BACKWARD_DEFINITIONS})


target_compile_options(cetric-lib PUBLIC -Wall -Wextra -pedantic)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)
