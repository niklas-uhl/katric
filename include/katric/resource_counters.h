// Counters to tell why one rank is slower than the others in the same code on
// the same amount of data (the straggler of doc/refactoring-plan.md, results
// 2026-11-23): page faults, where the rank runs, its transparent huge pages,
// and, where perf events are allowed, user-space cycles, instructions and dTLB
// load misses.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>

#include <sys/resource.h>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "katric/caliper.h"

namespace katric::profiling {

/// Opt-in (`enabled()`: the environment variable KATRIC_RESOURCE_COUNTERS is
/// set): it costs ~40 ms per use at 2^22 edges per rank on SuperMUC-NG (most
/// likely reading smaps_rollup, which walks the page tables).
/// Reports, when it goes out of scope, to the Caliper region open then (so
/// construct it after the region's scope guard): `page_faults` and
/// `major_page_faults` during its lifetime; `cpu` and `numa_node` the rank runs
/// on and `anon_huge_pages_kb` of the process at the end (Linux); and
/// `cycles`, `instructions`, `dtlb_load_misses` (user space) during its
/// lifetime, each only if the perf event could be opened
/// (/proc/sys/kernel/perf_event_paranoid <= 2). Per rank through Caliper's
/// recorder (experiments/straggler.suite.yaml, eval/perrank.py); in the
/// aggregated counters only min/max/sum.
class ResourceCounters {
public:
  ResourceCounters() {
    faults_ = faults();
#ifdef __linux__
    for (std::size_t i = 0; i < num_events; ++i) {
      fds_[i] = open_event(events[i].type, events[i].config);
    }
    for (int const fd : fds_) {
      if (fd >= 0) {
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
      }
    }
#endif
  }
  /// Whether KATRIC_RESOURCE_COUNTERS is set (read once).
  [[nodiscard]] static auto enabled() -> bool {
    static bool const enabled = std::getenv("KATRIC_RESOURCE_COUNTERS") != nullptr;
    return enabled;
  }

  ResourceCounters(ResourceCounters const&) = delete;
  auto operator=(ResourceCounters const&) -> ResourceCounters& = delete;
  ResourceCounters(ResourceCounters&&) = delete;
  auto operator=(ResourceCounters&&) -> ResourceCounters& = delete;

  ~ResourceCounters() {
#ifdef __linux__
    std::int64_t values[num_events] = {};
    for (std::size_t i = 0; i < num_events; ++i) {
      int const fd = fds_[i];
      if (fd < 0) {
        continue;
      }
      ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
      std::uint64_t value = 0;
      if (read(fd, &value, sizeof(value)) == sizeof(value)) {
        values[i] = static_cast<std::int64_t>(value);
      }
      close(fd);
    }
#endif
    auto const [minor, major] = faults();
    counter_add("page_faults", minor - faults_.first);
    counter_add("major_page_faults", major - faults_.second);
#ifdef __linux__
    for (std::size_t i = 0; i < num_events; ++i) {
      if (fds_[i] >= 0) {
        counter_add(events[i].name, values[i]);
      }
    }
    unsigned cpu = 0;
    unsigned node = 0;
    if (syscall(SYS_getcpu, &cpu, &node, nullptr) == 0) {
      counter_add("cpu", cpu);
      counter_add("numa_node", node);
    }
    if (auto const kb = anon_huge_pages_kb(); kb >= 0) {
      counter_add("anon_huge_pages_kb", kb);
    }
#endif
  }

private:
  static auto faults() -> std::pair<std::int64_t, std::int64_t> {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return {usage.ru_minflt, usage.ru_majflt};
  }

#ifdef __linux__
  struct Event {
    std::uint32_t type;
    std::uint64_t config;
    char const* name;
  };
  static constexpr std::size_t num_events = 3;
  static constexpr Event events[num_events] = {
      {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cycles"},
      {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "instructions"},
      {PERF_TYPE_HW_CACHE,
       PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
           (PERF_COUNT_HW_CACHE_RESULT_MISS << 16),
       "dtlb_load_misses"},
  };

  /// A disabled user-space counter of this process on any CPU; -1 if not
  /// allowed or not supported.
  static auto open_event(std::uint32_t type, std::uint64_t config) -> int {
    perf_event_attr attr{};
    attr.size = sizeof(attr);
    attr.type = type;
    attr.config = config;
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    return static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0));
  }

  /// AnonHugePages of /proc/self/smaps_rollup in kB, -1 if not readable.
  static auto anon_huge_pages_kb() -> std::int64_t {
    std::ifstream smaps("/proc/self/smaps_rollup");
    std::string const key = "AnonHugePages:";
    for (std::string line; std::getline(smaps, line);) {
      if (line.starts_with(key)) {
        return std::stoll(line.substr(key.size()));
      }
    }
    return -1;
  }

  int fds_[num_events] = {-1, -1, -1};
#endif
  std::pair<std::int64_t, std::int64_t> faults_;
};

}  // namespace katric::profiling
