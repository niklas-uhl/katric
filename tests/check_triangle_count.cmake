# Runs COMMAND (a ;-separated list) with --kagen_option_string KAGEN_OPTION_STRING and checks that it exits
# successfully and reports EXPECTED triangles. Used instead of PASS_REGULAR_EXPRESSION, which ignores the exit code.
# The option string is passed separately because it contains semicolons; quoted, it stays a single argument.
#
# cmake -DCOMMAND=<cmd;args...> -DKAGEN_OPTION_STRING=<options> -DEXPECTED=<count> -P check_triangle_count.cmake
execute_process(
  COMMAND ${COMMAND} --kagen_option_string "${KAGEN_OPTION_STRING}"
  OUTPUT_VARIABLE output
  ERROR_VARIABLE output
  RESULT_VARIABLE result)
message("${output}")
if(NOT result EQUAL 0)
  message(FATAL_ERROR "command failed with exit code ${result}")
endif()
if(NOT output MATCHES "([0-9]+) triangles")
  message(FATAL_ERROR "no triangle count in the output")
endif()
if(NOT CMAKE_MATCH_1 EQUAL EXPECTED)
  message(FATAL_ERROR "expected ${EXPECTED} triangles, got ${CMAKE_MATCH_1}")
endif()
