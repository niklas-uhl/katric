add_library(local_spooky spooky.cpp spooky.h)

target_include_directories(local_spooky PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
