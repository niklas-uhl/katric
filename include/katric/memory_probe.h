// Memory snapshots at phase boundaries, to find what fills a node on RMAT with
// BriefKAsten grid 16 / 64 KiB (doc/briefkasten-migration.md, step 8: the node
// runs out of memory in iteration 2, the OOM killer takes a rank, its peers'
// CMA reads fail). Opt-in: the environment variable KATRIC_MEMORY_PROBE.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <fstream>

#include <sys/resource.h>
#include <unistd.h>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "katric/caliper.h"

namespace katric::profiling {

/// Whether KATRIC_MEMORY_PROBE is set (read once).
[[nodiscard]] inline auto memory_probe_enabled() -> bool {
  static bool const enabled = std::getenv("KATRIC_MEMORY_PROBE") != nullptr;
  return enabled;
}

/// If enabled, reports this rank's memory as counters of a Caliper region
/// `where` (nested in the region open now; per rank with Caliper's recorder):
/// `mem_rss_kb` (resident now), `mem_peak_rss_kb` (the process's high-water
/// mark so far), and glibc's view of its heap (mallinfo2, all arenas, MPI's
/// allocations included): `mem_heap_used_kb` (allocated, not freed),
/// `mem_heap_free_kb` (freed but kept by malloc), `mem_heap_mmap_kb` (chunks
/// in their own mappings, returned at free). RSS far above used + free + mmap
/// is memory not from malloc (MPI's own mappings, shared memory); heap used
/// staying high after an iteration, when katric has freed everything but the
/// input, is memory someone else holds (MPI's eager / unexpected-message
/// buffers). Read-only: it never releases or discards memory. `where` must be
/// a string literal.
inline void memory_snapshot(char const* where) {
  if (!memory_probe_enabled()) {
    return;
  }
  CALI_MARK_BEGIN(where);
  long pages = 0;
  long resident = 0;
  if (std::ifstream statm("/proc/self/statm"); statm >> pages >> resident) {
    counter_add("mem_rss_kb", resident * (sysconf(_SC_PAGESIZE) / 1024));
  }
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  counter_add("mem_peak_rss_kb", usage.ru_maxrss);
#if defined(__GLIBC__) && (__GLIBC__ > 2 || __GLIBC_MINOR__ >= 33)
  auto const info = mallinfo2();
  counter_add("mem_heap_used_kb", static_cast<std::int64_t>(info.uordblks / 1024));
  counter_add("mem_heap_free_kb", static_cast<std::int64_t>(info.fordblks / 1024));
  counter_add("mem_heap_mmap_kb", static_cast<std::int64_t>(info.hblkhd / 1024));
#endif
  CALI_MARK_END(where);
}

}  // namespace katric::profiling
