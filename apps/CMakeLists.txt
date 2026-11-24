# Build settings recorded as Adiak metadata (build.*): CMAKE_CXX_FLAGS(_<CONFIG>) are variables, the rest are katric's
# effective COMPILE_OPTIONS / COMPILE_DEFINITIONS (including those of linked targets), evaluated by file(GENERATE).
# Flags CMake adds itself (-std, -arch, -fPIC) are not included.
string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type)
string(STRIP "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_build_type}}" KATRIC_BUILD_CXX_FLAGS)
configure_file(build_info.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/build_info.cpp.gen @ONLY)
# Evaluated once per enabled language; flags differ per language (e.g. MPI::MPI_CXX via kamping-v2), so only C++ writes.
file(
  GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/build_info.cpp
  INPUT ${CMAKE_CURRENT_BINARY_DIR}/build_info.cpp.gen
  CONDITION $<COMPILE_LANGUAGE:CXX>
  TARGET katric)

add_executable(katric katric.cpp parse_config.cpp algorithm.cpp verify.cpp git_hash.cpp
                      ${CMAKE_CURRENT_BINARY_DIR}/build_info.cpp)
target_include_directories(katric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(katric PRIVATE katric-algorithms katric_cxx_flags CLI11::CLI11 kamping::spdlog-adapter
                                     cmake_git_version_tracking)

# Competitors: one adapter per competitor (competitors/<name>.cpp), each in its own object library, so that the
# competitor's headers, include directories and definitions stay in that one translation unit.
if(KATRIC_COMPETITOR_TRIC)
  add_library(katric-tric OBJECT competitors/tric.cpp)
  target_include_directories(katric-tric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(katric-tric PRIVATE katric-lib katric_cxx_flags tric::tric)
  target_link_libraries(katric PRIVATE katric-tric)
  target_compile_definitions(katric PRIVATE KATRIC_HAVE_TRIC)
endif()
if(KATRIC_COMPETITOR_HAVOQGT)
  add_library(katric-havoqgt OBJECT competitors/havoqgt.cpp)
  target_include_directories(katric-havoqgt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(katric-havoqgt PRIVATE katric-lib katric_cxx_flags havoqgt::havoqgt)
  target_link_libraries(katric PRIVATE katric-havoqgt)
  target_compile_definitions(katric PRIVATE KATRIC_HAVE_HAVOQGT)
endif()

set_target_properties(katric PROPERTIES CXX_CLANG_TIDY "${KATRIC_CLANG_TIDY_COMMAND}")
