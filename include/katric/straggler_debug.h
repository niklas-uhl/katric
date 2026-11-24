// Printed diagnostics for the straggler of doc/refactoring-plan.md (results
// 2026-11-23): on some ranks (all >= ~1030 at 1536 cores) the ghost push
// pattern build runs the same instructions in 3.3x the cycles with 22x the
// dTLB load misses, and the allocator variants (hugetlb, fixed mmap threshold)
// do not change that. perf is impractical on SuperMUC-NG, so this prints per
// rank where the arrays of the build live and which access costs the time.
//
// Opt-in: the environment variable KATRIC_STRAGGLER_DEBUG is set. Runs in
// Graph::ghost_push_pattern() before the real build (outside its Caliper
// region `pattern`, inside `prepare`). Every line starts with
// `STRAGGLER r=<rank> ` and is a list of key=value pairs, so
// `grep '^STRAGGLER'` on the job log gives a table per line kind (`kind=`):
//   host    host name, CPU and NUMA node the rank runs on, n / ghosts / edges
//   array   one of the build's arrays (adjncy, owners, plus the copies below):
//           address, bytes, the VMAs of /proc/self/smaps overlapping it (count,
//           summed size / Rss / AnonHugePages, THPeligible, the first one's
//           name, e.g. [heap]) and the NUMA nodes of its pages (move_pages
//           without moving: `numa=N0:8192,absent:0`)
//   time    the build's loop in stages, min of 3 runs each, in microseconds:
//           scan (read every neighbor), is_local (+ count ghost entries),
//           owner (+ owners[ghost index]), dedupe (+ last rank sent to; the
//           real loop's body without the append), the same dedupe loop on
//           owners_copy (a fresh std::vector) and owners_thp (a fresh 2 MiB
//           aligned buffer with madvise(MADV_HUGEPAGE)), and the real build
// AnonHugePages and THPeligible are per VMA (the kernel gives no per-page
// huge-page flag without root): an array in [heap] shares its VMA with
// everything else there.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "katric/ghost_exchange.h"

namespace katric::profiling {

/// Whether KATRIC_STRAGGLER_DEBUG is set (read once).
[[nodiscard]] inline auto straggler_debug_enabled() -> bool {
  static bool const enabled = std::getenv("KATRIC_STRAGGLER_DEBUG") != nullptr;
  return enabled;
}

namespace straggler_debug_detail {

struct Vma {
  std::uintptr_t begin = 0;
  std::uintptr_t end = 0;
  std::string name;
  std::int64_t rss_kb = 0;
  std::int64_t anon_huge_kb = 0;
  int thp_eligible = -1;
};

/// The VMAs of /proc/self/smaps (empty if not readable, e.g. on macOS).
inline auto read_smaps() -> std::vector<Vma> {
  std::vector<Vma> vmas;
  std::ifstream smaps("/proc/self/smaps");
  auto const field = [](std::string const& line, char const* key,
                        std::int64_t& value) {
    std::size_t const len = std::strlen(key);
    if (line.compare(0, len, key) == 0) {
      value = std::stoll(line.substr(len));
      return true;
    }
    return false;
  };
  for (std::string line; std::getline(smaps, line);) {
    // a header line: "begin-end perms offset dev inode [name]"; field lines
    // have their colon before the first space
    auto const dash = line.find('-');
    auto const space = line.find(' ');
    if (dash != std::string::npos && space != std::string::npos &&
        dash < space && line.find(':') > space) {
      Vma vma;
      vma.begin = std::stoull(line.substr(0, dash), nullptr, 16);
      vma.end =
          std::stoull(line.substr(dash + 1, space - dash - 1), nullptr, 16);
      std::istringstream rest(line.substr(space + 1));
      std::string perms, offset, dev, inode;
      rest >> perms >> offset >> dev >> inode;
      std::getline(rest >> std::ws, vma.name);
      vmas.push_back(std::move(vma));
      continue;
    }
    if (vmas.empty()) {
      continue;
    }
    std::int64_t value = 0;
    if (field(line, "Rss:", value)) {
      vmas.back().rss_kb = value;
    } else if (field(line, "AnonHugePages:", value)) {
      vmas.back().anon_huge_kb = value;
    } else if (field(line, "THPeligible:", value)) {
      vmas.back().thp_eligible = static_cast<int>(value);
    }
  }
  return vmas;
}

/// The NUMA node of every 4 KiB page of [data, data + bytes), as a histogram
/// "N<node>:<pages>,...,absent:<pages>" (move_pages(2) with nodes = NULL only
/// queries); "-" off Linux.
inline auto numa_histogram([[maybe_unused]] void const* data,
                           [[maybe_unused]] std::size_t bytes) -> std::string {
#ifdef __linux__
  constexpr std::uintptr_t page = 4096;
  auto const first = reinterpret_cast<std::uintptr_t>(data) & ~(page - 1);
  auto const last = reinterpret_cast<std::uintptr_t>(data) + bytes;
  std::vector<void*> pages;
  for (std::uintptr_t a = first; a < last; a += page) {
    pages.push_back(reinterpret_cast<void*>(a));
  }
  if (pages.empty()) {
    return "-";
  }
  std::vector<int> status(pages.size(), 0);
  if (syscall(SYS_move_pages, 0, pages.size(), pages.data(), nullptr,
              status.data(), 0) != 0) {
    return "error";
  }
  std::map<int, std::size_t> nodes;
  std::size_t absent = 0;
  for (int const s : status) {
    if (s >= 0) {
      ++nodes[s];
    } else {
      ++absent;
    }
  }
  std::ostringstream out;
  for (auto const& [node, count] : nodes) {
    out << 'N' << node << ':' << count << ',';
  }
  out << "absent:" << absent;
  return out.str();
#else
  return "-";
#endif
}

inline void print_array(std::ostringstream& out,
                        std::string const& prefix,
                        std::vector<Vma> const& vmas,
                        char const* name,
                        void const* data,
                        std::size_t bytes) {
  auto const begin = reinterpret_cast<std::uintptr_t>(data);
  auto const end = begin + bytes;
  std::size_t count = 0;
  std::uintptr_t vma_bytes = 0;
  std::int64_t rss_kb = 0;
  std::int64_t anon_huge_kb = 0;
  int thp_eligible = -1;
  std::string first_name = "-";
  for (Vma const& vma : vmas) {
    if (vma.end <= begin || vma.begin >= end) {
      continue;
    }
    if (count == 0) {
      first_name = vma.name.empty() ? "anon" : vma.name;
      thp_eligible = vma.thp_eligible;
    } else if (vma.thp_eligible != thp_eligible) {
      thp_eligible = -2;  // mixed
    }
    ++count;
    vma_bytes += vma.end - vma.begin;
    rss_kb += vma.rss_kb;
    anon_huge_kb += vma.anon_huge_kb;
  }
  out << prefix << "kind=array name=" << name << " addr=0x" << std::hex << begin
      << std::dec << " bytes=" << bytes << " vmas=" << count
      << " vma_bytes=" << vma_bytes << " vma_rss_kb=" << rss_kb
      << " vma_anon_huge_kb=" << anon_huge_kb
      << " thp_eligible=" << thp_eligible << " vma_name=" << first_name
      << " numa=" << numa_histogram(data, bytes) << '\n';
}

/// Min over 3 runs of `run`, in microseconds; `run` returns a value that is
/// summed into `sink` so the loop is not optimized away.
template <typename Run>
auto time_min_us(Run const& run, std::uint64_t& sink) -> std::int64_t {
  auto best = std::numeric_limits<std::int64_t>::max();
  for (int i = 0; i < 3; ++i) {
    auto const start = std::chrono::steady_clock::now();
    sink += run();
    auto const stop = std::chrono::steady_clock::now();
    best = std::min<std::int64_t>(
        best, std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
                  .count());
  }
  return best;
}

}  // namespace straggler_debug_detail

/// Prints the diagnostics above for the pattern build on `G` (undirected, as
/// the build needs) to stderr, as one write per rank. Not collective.
template <typename Graph>
void print_straggler_debug(Graph const& G) {
  namespace detail = straggler_debug_detail;
  int const rank = G.comm().rank();
  std::string const prefix = "STRAGGLER r=" + std::to_string(rank) + " ";
  std::size_t const n = G.num_local_vertices();
  std::span<int const> const owners = G.numbering().ghost_owners();
  auto const size = static_cast<std::size_t>(G.comm().size());

  // The stages of build_ghost_push_pattern()'s loop. Handles as plain
  // integers: local vertices < n, ghost index = handle - n.
  auto const scan = [&] {
    std::uint64_t sum = 0;
    for (auto const u : G.local_vertices()) {
      for (auto const v : G.neighbors(u)) {
        sum += static_cast<std::uint64_t>(v);
      }
    }
    return sum;
  };
  auto const is_local = [&] {
    std::uint64_t ghosts = 0;
    for (auto const u : G.local_vertices()) {
      for (auto const v : G.neighbors(u)) {
        ghosts += static_cast<std::size_t>(v) >= n ? 1 : 0;
      }
    }
    return ghosts;
  };
  auto const owner = [&G, n](std::span<int const> owner_of) {
    return [&G, n, owner_of] {
      std::uint64_t sum = 0;
      for (auto const u : G.local_vertices()) {
        for (auto const v : G.neighbors(u)) {
          auto const h = static_cast<std::size_t>(v);
          if (h >= n) {
            sum += static_cast<std::uint64_t>(owner_of[h - n]);
          }
        }
      }
      return sum;
    };
  };
  std::vector<std::size_t> last_sent(size);
  auto const dedupe = [&G, n, &last_sent](std::span<int const> owner_of) {
    return [&G, n, owner_of, &last_sent] {
      std::ranges::fill(last_sent, std::numeric_limits<std::size_t>::max());
      std::uint64_t sent = 0;
      for (auto const u : G.local_vertices()) {
        auto const u_local = static_cast<std::size_t>(u);
        for (auto const v : G.neighbors(u)) {
          auto const h = static_cast<std::size_t>(v);
          if (h < n) {
            continue;
          }
          auto const dest = static_cast<std::size_t>(owner_of[h - n]);
          if (last_sent[dest] == u_local) {
            continue;
          }
          last_sent[dest] = u_local;
          ++sent;
        }
      }
      return sent;
    };
  };

  // fresh copies of owners: a plain vector, and a 2 MiB aligned buffer
  // madvised for transparent huge pages before the copy touches it
  std::vector<int> const owners_copy(owners.begin(), owners.end());
  constexpr std::size_t huge = std::size_t{2} << 20;
  std::size_t const thp_bytes =
      std::max(huge, (owners.size_bytes() + huge - 1) / huge * huge);
  auto* const owners_thp =
      static_cast<int*>(std::aligned_alloc(huge, thp_bytes));
#ifdef __linux__
  madvise(owners_thp, thp_bytes, MADV_HUGEPAGE);
#endif
  std::ranges::copy(owners, owners_thp);
  std::span<int const> const thp_span(owners_thp, owners.size());

  std::uint64_t sink = 0;
  std::ostringstream times;
  times << prefix << "kind=time"
        << " scan_us=" << detail::time_min_us(scan, sink)
        << " is_local_us=" << detail::time_min_us(is_local, sink)
        << " owner_us=" << detail::time_min_us(owner(owners), sink)
        << " dedupe_us=" << detail::time_min_us(dedupe(owners), sink)
        << " dedupe_copy_us=" << detail::time_min_us(dedupe(owners_copy), sink)
        << " dedupe_thp_us=" << detail::time_min_us(dedupe(thp_span), sink)
        << " build_us="
        << detail::time_min_us(
               [&G] { return build_ghost_push_pattern(G).send_vertex.size(); },
               sink)
        << " sink=" << sink % 1000 << '\n';

  std::ostringstream out;
  char host[256] = "-";
  unsigned cpu = 0;
  unsigned node = 0;
#ifdef __linux__
  gethostname(host, sizeof(host));
  host[sizeof(host) - 1] = '\0';
  syscall(SYS_getcpu, &cpu, &node, nullptr);
#endif
  out << prefix << "kind=host host=" << host << " cpu=" << cpu
      << " numa_node=" << node << " n=" << n << " ghosts=" << owners.size()
      << " edges=" << G.num_local_edges() << '\n';
  // after the timing: the copies are touched, so smaps shows their pages
  auto const vmas = detail::read_smaps();
  if (n > 0) {
    auto const first = G.neighbors(*G.local_vertices().begin());
    detail::print_array(out, prefix, vmas, "adjncy", first.data(),
                        G.num_local_edges() * sizeof(first[0]));
  }
  detail::print_array(out, prefix, vmas, "owners", owners.data(),
                      owners.size_bytes());
  detail::print_array(out, prefix, vmas, "owners_copy", owners_copy.data(),
                      owners_copy.size() * sizeof(int));
  detail::print_array(out, prefix, vmas, "owners_thp", owners_thp, thp_bytes);
  out << times.str();
  std::free(owners_thp);

  std::string const text = out.str();
  std::fwrite(text.data(), 1, text.size(), stderr);
  std::fflush(stderr);
}

}  // namespace katric::profiling
