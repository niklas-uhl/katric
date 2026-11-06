file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${parallel-triangle-counter_SOURCE_DIR}/include/[a-zA-Z_0-9]+.h")

set(FILE_LIST)
if (NOT FILE_LIST)
    set(DUMMY_FILE ${CMAKE_CURRENT_BINARY_DIR}/null.cpp)
    file(WRITE "${DUMMY_FILE}" "")
    list(APPEND FILE_LIST "${DUMMY_FILE}")
endif()


add_library(cetric-lib ${FILE_LIST} ${HEADER_LIST})

target_include_directories(cetric-lib PUBLIC ${PROJECT_SOURCE_DIR}/include/)

target_link_libraries(cetric-lib PUBLIC MPI::MPI_CXX)
target_link_libraries(cetric-lib PUBLIC tlx)
target_link_libraries(cetric-lib PUBLIC sparsehash::sparsehash)
target_link_libraries(cetric-lib PUBLIC cereal::cereal)
target_link_libraries(cetric-lib PUBLIC magic_enum::magic_enum)
target_link_libraries(cetric-lib PUBLIC fmt::fmt)
target_link_libraries(cetric-lib PUBLIC spdlog::spdlog)
# Compile-time log level: SPDLOG_DEBUG/SPDLOG_TRACE calls below this level are compiled out.
set(CETRIC_ACTIVE_LOG_LEVEL "INFO" CACHE STRING "Compile-time spdlog level (TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL, OFF)")
target_compile_definitions(cetric-lib PUBLIC SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${CETRIC_ACTIVE_LOG_LEVEL})
target_link_libraries(cetric-lib PUBLIC message-queue)
target_link_libraries(cetric-lib PUBLIC graph-io)
target_link_libraries(cetric-lib PUBLIC TBB::tbb)
if (CETRIC_USE_OPENMP)
  target_link_libraries(cetric-lib PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(cetric-lib PUBLIC CETRIC_HAS_OPENMP)
endif()
target_link_libraries(cetric-lib PUBLIC kassert::kassert)
target_link_libraries(cetric-lib PUBLIC utils_tm)
target_link_libraries(cetric-lib PUBLIC Boost::headers)
target_link_libraries(cetric-lib PUBLIC kamping::types)

option(CETRIC_BINARY_INTERSECTION_RECURSIVE "Use recursion in binary search intersection" OFF)
if (CETRIC_BINARY_INTERSECTION_RECURSIVE)
  target_compile_definitions(cetric-lib PUBLIC -DCETRIC_BINARY_INTERSECTION_RECURSIVE)
endif()

target_compile_options(cetric-lib PUBLIC -Wall -Wextra -pedantic)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)
