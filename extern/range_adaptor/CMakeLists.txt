add_library(range_adaptor INTERFACE)
target_include_directories(range_adaptor INTERFACE .)
target_link_libraries(range_adaptor INTERFACE TBB::tbb tlx)
