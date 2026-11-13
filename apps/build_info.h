#pragma once

namespace cetric::build_info {
// How katric was built, from the CMake configuration (build_info.cpp.in); recorded as Adiak metadata (build.*)
extern char const* const build_type;
extern char const* const cxx_compiler;
extern char const* const cxx_flags;
extern char const* const cxx_definitions;
} // namespace cetric::build_info
