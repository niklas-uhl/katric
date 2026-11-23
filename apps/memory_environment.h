// What decides how memory is backed, recorded as run metadata (adiak): the
// glibc version, the transparent huge page settings and the allocator's
// environment (the straggler of doc/refactoring-plan.md, 2026-11-23).
#pragma once

#include <cstdlib>
#include <fstream>
#include <string>

#include <adiak.hpp>

#if defined(__GLIBC__)
#include <gnu/libc-version.h>
#endif

/// Adiak values `memory.glibc`, `memory.thp_enabled`, `memory.thp_defrag`
/// (the bracketed choice of /sys/kernel/mm/transparent_hugepage/*), and
/// `memory.GLIBC_TUNABLES` / `memory.MALLOC_MMAP_THRESHOLD_` if set.
inline void record_memory_environment() {
#if defined(__GLIBC__)
  adiak::value("memory.glibc", std::string(gnu_get_libc_version()));
#endif
  for (char const* setting : {"enabled", "defrag"}) {
    std::ifstream file(std::string("/sys/kernel/mm/transparent_hugepage/") +
                       setting);
    std::string line;
    if (std::getline(file, line)) {
      auto const open = line.find('[');
      auto const close = line.find(']');
      if (open != std::string::npos && close != std::string::npos) {
        line = line.substr(open + 1, close - open - 1);
      }
      adiak::value(std::string("memory.thp_") + setting, line);
    }
  }
  for (char const* name : {"GLIBC_TUNABLES", "MALLOC_MMAP_THRESHOLD_"}) {
    if (char const* value = std::getenv(name)) {
      adiak::value(std::string("memory.") + name, std::string(value));
    }
  }
}
