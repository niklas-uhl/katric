file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${parallel-triangle-counter_SOURCE_DIR}/include/*.h")

set(FILE_LIST
        distributed_graph_io.cpp
        )

add_library(cetric-lib ${FILE_LIST} ${HEADER_LIST})

target_include_directories(cetric-lib PUBLIC ${PROJECT_SOURCE_DIR}/include/)
target_include_directories(cetric-lib PUBLIC ${SPARSEHASH_INCLUDE_DIR})
target_include_directories(cetric-lib PUBLIC ${KAGEN_INCLUDE_DIRS})

target_link_libraries(cetric-lib PUBLIC local_spooky)
target_link_libraries(cetric-lib PUBLIC MPI::MPI_CXX)
target_link_libraries(cetric-lib PUBLIC ${KAGEN_LINK_LIBRARIES})
target_link_libraries(cetric-lib PUBLIC nlohmann_json::nlohmann_json) 


target_compile_options(cetric-lib PUBLIC -Wall -Wextra -pedantic)

source_group(
        TREE "${PROJECT_SOURCE_DIR}/include"
        PREFIX "Header Files"
        FILES ${HEADER_LIST})
