# Applies PATCH (a git diff) to the current directory, once: a patch that is already applied is skipped, a patch that
# applies neither forwards nor backwards fails the build. Used as FetchContent PATCH_COMMAND, which may run again on
# reconfigure.
#
# cmake -DPATCH=<file> -P apply_patch.cmake
execute_process(COMMAND git apply --reverse --check ${PATCH} RESULT_VARIABLE already_applied OUTPUT_QUIET ERROR_QUIET)
if(already_applied EQUAL 0)
  message(STATUS "Patch already applied: ${PATCH}")
  return()
endif()
execute_process(COMMAND git apply ${PATCH} RESULT_VARIABLE result)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Failed to apply patch ${PATCH}")
endif()
