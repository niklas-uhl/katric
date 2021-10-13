# from https://gitlab.cern.ch/dss/map-benchmarks/-/blob/master/cmake/Findsparsehash.cmake
# include (FindPackageHandleStandardArgs)
#
# find_path (SPARSEHASH_INCLUDE_DIR google/sparse_hash_map)
#
# find_package_handle_standard_args (sparsehash DEFAULT_MSG SPARSEHASH_INCLUDE_DIR)
if(SPARSEHASH_INCLUDE_DIR)
    set(SPARSEHASH_FIND_QUIETLY TRUE)
endif(SPARSEHASH_INCLUDE_DIR)

find_path(SPARSEHASH_INCLUDE_DIR google/sparsehash/sparsehashtable.h)

# handle the QUIETLY and REQUIRED arguments and set SPARSEHASH_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(sparsehash DEFAULT_MSG SPARSEHASH_INCLUDE_DIR)

mark_as_advanced(SPARSEHASH_INCLUDE_DIR)