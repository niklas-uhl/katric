# Runs COMMAND (a ;-separated list) and checks that it exits successfully and reports EXPECTED triangles.
# Used instead of PASS_REGULAR_EXPRESSION, which ignores the exit code.
#
# cmake -DCOMMAND=<cmd;args...> -DEXPECTED=<count> -P check_triangle_count.cmake
execute_process(
  COMMAND ${COMMAND}
  OUTPUT_VARIABLE output
  ERROR_VARIABLE output
  RESULT_VARIABLE result)
message("${output}")
if(NOT result EQUAL 0)
  message(FATAL_ERROR "command failed with exit code ${result}")
endif()
if(NOT output MATCHES "triangles: +([0-9]+)")
  message(FATAL_ERROR "no triangle count in the output")
endif()
if(NOT CMAKE_MATCH_1 EQUAL EXPECTED)
  message(FATAL_ERROR "expected ${EXPECTED} triangles, got ${CMAKE_MATCH_1}")
endif()
