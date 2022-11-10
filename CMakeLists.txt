cmake_minimum_required(VERSION 3.16)
project(cetric LANGUAGES C CXX)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

    set(CMAKE_CXX_STANDARD 17)

    # Let's ensure -std=c++xx instead of -std=g++xx
    # set(CMAKE_CXX_EXTENSIONS OFF)

    # Let's nicely support folders in IDEs
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    # Add build type Asan for using AddressSanitizer
    #include(cmake/AddAsanBuildType.cmake)
    include(CTest)
endif()

option(CETRIC_DEBUG_CONTAINERS "Use GNU debug containers" OFF)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

find_package(sparsehash REQUIRED)
add_library(sparsehash INTERFACE)
target_include_directories(sparsehash INTERFACE ${SPARSEHASH_INCLUDE_DIR})

find_package(Boost REQUIRED COMPONENTS headers)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
#set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
if (CETRIC_DEBUG_CONTAINERS)
  add_compile_definitions(_GLIBCXX_DEBUG)
  add_compile_definitions(_GLIBCXX_DEBUG_PEDANTIC)
endif()
set(KASSERT_ASSERTION_LEVEL $<IF:$<CONFIG:Debug>,30,0> CACHE STRING "KASSERT assertion level")

add_subdirectory(extern)

# The compiled library code is here
add_subdirectory(src)

# The executable code is here
add_subdirectory(apps)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(tests)
endif()
