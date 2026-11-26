# Competitors behind the benchmark harness

Working checklist (planned 2026-11-21). Tick items (`[x]`) as they land and add the commit hash.

## Context

The dissertation version needs competitor runs through KaTric's own pipeline: KaGen input, Caliper/Adiak output,
`--verify`, kaval suites and `eval/katric_eval.py`. For IPDPS'23, TriC, HavoqGT and Tom2D ran as external binaries
(sibling checkouts with KaGen patched in), and the old runner parsed their log lines
(`git show f8a2d47^:experiments/runners.py`). That runner is gone, and `make_algorithm` (`apps/algorithm.cpp`) only
returns KaTric.

The model is KaCCv2 (`benchmark/detail/*_algorithm.cpp`, `algorithm_factory.cpp`): one binary, one pimpl adapter per
competitor, and a factory that switches on `--algorithm`. KaCCv2's weak spots, and what we do instead:

| KaCCv2 | Here |
| --- | --- |
| Competitors are always built, pinned to branches | Opt-in, pinned to SHAs |
| Adapters differ in what counts as run time | One rule for all (decision 2) |
| Some adapters have no `reset()`, so later iterations may reuse state | Each adapter documents why a re-run is clean |

## Decisions

1. **One binary, opt-in.** `--algorithm katric|tric|havoqgt|tripoll|tom2d|combblas|lcc|grape`, each competitor behind
   `AbstractAlgorithm`.
   - Each competitor has a CMake option `KATRIC_COMPETITOR_<NAME>`, default OFF. The `experiments` preset turns on the
     ones that build on SuperMUC.
   - Each fork comes in via FetchContent, pinned by SHA, with `SYSTEM EXCLUDE_FROM_ALL` and a comment naming the
     branch.
   - Selecting a competitor that was not compiled in is a parse error that names the CMake option.
2. **Timing scope: native input → count.**
   - `ingest` only converts the `kagen::Graph` into what the competitor's own file reader would produce. This is not
     timed.
   - `total` covers everything after that: the competitor's preprocessing (graph construction, ordering,
     distribution) and its counting. Where the code allows, it is split into Caliper sub-regions `preprocessing` and
     `counting`.
   - `counting` alone gives the paper's numbers again. HavoqGT's `total` will be much worse than in the paper, which
     left out its preprocessing.
3. **Competitors:** TriC, HavoqGT, TriPoll/YGM, Tom2D, CombBLAS (masked SpGEMM), LCC (Strausz et al.) and GRAPE
   (libgrape-lite's LCC) as the general-framework baseline. Bader's code is out because it is shared memory only
   (OpenMP, 32-bit ids). Other frameworks are out too (checked 2026-11-23): Blogel and Pregel+ ship no usable TC,
   PowerGraph is unmaintained and already beaten by TriC/Tom2D, GraphX and Giraph are JVM, DistTC is GPU, and
   GraphBLAS/LAGraph, GBBS and Ligra are shared memory.
4. **Forks:** a `katric` branch in each fork under niklas-uhl. The branch has a library target with a "construct from
   CSR or edge list + communicator" entry point, no `main`, and no KaGen submodule.
   - Each `katric` branch sits on the competitor's **latest upstream** version (checked 2026-11-24, see "Upstreams").
   - Dependencies the counting does not need are dropped from the fork's build (Metall, CGAL, UMap, HDF5, ...).
   - C++ competitors live in a namespace of their own (`tric`, `xxtric`, `havoqgt`), so that all of them link into
     one binary. Macros are contained by the one-TU-per-adapter rule. C competitors (Tom2D, LCC): check `nm` for
     clashes. Known clash to come: HavoqGT bundles an old YGM in `namespace ygm`, TriPoll uses the real one.
5. **LCC is GPL-3.0** (`fvella/lcc` + `spcl/CLaMPI`). It is linked only with `KATRIC_COMPETITOR_LCC=ON`. Binaries
   built with it are GPL and not distributed, and the README says so.
6. **CombBLAS builds on KaCCv2's adapter and fork** `niklas-uhl/CombBLAS@46ebfc31`, which carries the fixes.
   - Reused from `KaCCv2/benchmark/detail/combblas_algorithm.cpp`: `ImplBase`'s edge list → `ReadFromLocalEdgeList`
     ingest, and the `par::Mpi_datatype<unsigned long long>` specialization.
   - Written ourselves, the Azad/Buluç/Gilbert (IPDPS-W'15) count: L and U from A via `Prune`, the masked `PSpGEMM`
     L·U, `EWiseMult` with A, then `Reduce`.
   - Needs square grids, as KaCCv2's LACC suites do.

## Shared pieces

- [x] `make_algorithm` switches on a `Config::algorithm` enum. The deprecated `patric`/`cetric-x` spellings map to
      `katric`, so eval's `variant` column keeps working. A competitor that is not compiled in is rejected after
      parsing, naming its CMake option (`is_compiled_in()`)
- [x] One adapter per competitor in `apps/competitors/<name>.{h,cpp}`: the header declares only a factory
      (`make_tric()`), the `.cpp` is an object library `katric-<name>` that alone links the competitor's target, so its
      headers, include directories and definitions stay in that TU. The option ON defines `KATRIC_HAVE_<NAME>`
- [x] Competitor-specific options (`--tric-variant`, `--tric-buffer-size`, ...), under a help group per competitor
      (`->group()`, not `add_option_group()`: `option_values()` would not see the options of a named option group).
      They land in `conf.options`, so Adiak records them as `config.*`
- [ ] (started: vertex distribution, sorted CSR) `apps/competitors/convert.h` converts the `kagen::Graph` into a global edge list, a CSR with 64-bit global ids,
      or the vertex distribution (`kagen::BuildVertexDistribution`). It uses KaGen's own builders where they fit and
      never assumes sorted neighbourhoods
- [ ] Each adapter's constructor checks its constraints and aborts with a clear message:
  - square p (Tom2D, CombBLAS)
  - equal ranks per node (HavoqGT)
  - the MPI thread level YGM and GRAPE (`MPI_THREAD_MULTIPLE`) need. `main` asks for a higher level only when one
    of them is selected
- [ ] `reset()` rule: if the competitor mutates its input or state, `reset()` rebuilds it from the converted native
      input (region `reset`, untimed). Otherwise it is a no-op, with a comment saying why
- [ ] Communicators: pass `comm` where the fork takes one. Where a fork hard-codes `MPI_COMM_WORLD` (TriC, Tom2D, LCC),
      accept that: katric runs on `MPI_COMM_WORLD` anyway (legacy message queue)

## Per competitor, in order

The order follows what the dissertation chapter needs:
1. **TriC** first: it was a paper competitor, and it is the cheapest port (header-only, the fork already builds from
   KaGen), so it also tests the shared design.
2. **HavoqGT:** the other paper competitor; together with TriC it redoes the IPDPS'23 comparison.
3. **LCC:** the strongest new claim (up to 100x over TriC on scale-free graphs). It is the asynchronous design closest
   to "send N(v) once per rank", and reviewers would ask for it.
4. **TriPoll:** buffered asynchronous messaging with routing, the closest relative of DiTriC and its grid
   indirection.
5. **Tom2D** and 6. **CombBLAS:** the 2D / linear-algebra family. They give breadth but are not urgent: Tom2D was not
   in the paper, and the CombBLAS counter is our own code.
7. **GRAPE:** the answer to "why not a general graph framework?". Its LCC is DiTriC's base scheme (degree
   orientation, each ghost's out-edges sent once per rank) without contraction or indirection, well engineered, and
   the winner of the LDBC Graphalytics 2023 competition. So it isolates what contraction, indirection and rebalancing
   buy against an independent implementation. No build risk (CMake, MPI, Threads; Apache-2.0).

LCC and TriPoll carry the build risk (autotools + CLaMPI, YGM's MPI thread level), so a blocker shows up early
enough to decide whether to drop one.

| Step | Competitor | Fork / pin | Adapter notes |
| --- | --- | --- | --- |
| 1 | TriC (Ghosh & Halappanavar, HPEC'20) | `niklas-uhl/tric` (`katric` on `sg0/tric` master `6939dbe`) | See below |
| 1b | xxtric (Ghosh, arXiv 2610.29988, 2026), TriC's successor, CPU path | fork of `sg0/xxtric` (`main`, `eb797a32`) | `--algorithm xxtric`; same `Graph` as TriC, single counting |
| 2 | HavoqGT (Pearce et al.) | `niklas-uhl/havoqgt` (`master`, `fa09fbe3`) | See below. Equal ranks per node (32 of 48 on SuperMUC in the paper) |
| 3 | LCC (Strausz et al., IPDPS'22) | new forks of `fvella/lcc` and `spcl/CLaMPI` | See below |
| 4 | TriPoll / YGM (Steil et al., SC'21) | `niklas-uhl/tripoll` (`katric` on `LLNL/tripoll` main `ccfe8a0`), YGM master `0072d87` | See "TriPoll: findings" |
| 5 | Tom2D (Tom & Karypis 2019) | `niklas-uhl/2DTriangleCounting` (`katric` on `KarypisLab/2DTriangleCounting` master `f704a53`) | See "Tom2D: findings". Square p only |
| 6 | CombBLAS masked SpGEMM | `niklas-uhl/CombBLAS@46ebfc31` | Decision 6. `preprocessing` builds L and U; `counting` is the SpGEMM plus the reduce |
| 7 | GRAPE / libgrape-lite LCC (Fan et al., SIGMOD'17) | new fork of `alibaba/libgrape-lite` (`master`, `beaa6e8a`) | See below |

- **TriC.** Header-only. `run_kagen` (`graph.hpp:786`) already builds TriC's `Graph` from KaGen's CSR, so moving it
  into `ingest` is the whole conversion. The variants are distinct classes (`TriangulateAggrBuffered`, `...Hash2`,
  ...), so `--tric-variant plain|buffered|hash2` picks one at run time; check that their headers coexist in one TU.
  `preprocessing` constructs the `Triangulate*` object, `counting` is `count()`.
- **Tom2D.** C with GKlib and global state. On the fork branch, split `main.c` into `mpitc_count(csr, comm)`, check
  that the global state is re-initialized per call, and check for GKlib symbol clashes. The native input is its
  binary CSR (`.first_out`/`.head`), in memory.
- **HavoqGT.** The native input is an edge list. `preprocessing` is building the delegate-partitioned graph plus the
  degree ordering, which took over 900 s in the paper. `counting` is the triangle count. Metall and Boost are
  required only when the option is on.
- **TriPoll / YGM.** The native input is an edge list fed to its undirected graph. `preprocessing` builds the
  `ordered_directed_graph`; `counting` is `tc_push_only`, with `tc_push_pull` as an option. YGM comes via FetchContent
  at a pinned SHA.
- **LCC.** Autotools C, so the fork branch adds CMake: a library target and CLaMPI via FetchContent. The native input
  is adjacency lists. Build with CLaMPI and SIMD, as their README prescribes for the paper results. The public CLaMPI
  lacks the paper's custom eviction scores.
- **GRAPE.** Run `LCCOpt` (`examples/analytical_apps/lcc/lcc_opt.h`, what `--opt` runs for undirected graphs).
  What it does, from the source at `beaa6e8a`, in BSP supersteps on an edge-cut fragment:
  1. Degree exchange, then orientation from low to high degree, ties broken by a hash of the gid.
  2. Each local `v` sends N⁺(v) once to each rank holding an out-neighbour (`SendMsgThroughOEdges` →
     `frag.OEDests(v)`). The receiver keeps only ids it knows (`Gid2Vertex`).
  3. Sorted neighbourhoods, SIMD merge intersection, a per-thread dense bitmap for hubs
     (|N⁺| > 0.4 · max |N⁺|), threads (`ParallelEngine`) on top of MPI.
  4. LCC only: an `atomic_add` on all three vertices per triangle, then a superstep that sends the ghosts' counts to
     their owners.

  To do on the fork branch:
  - A switch `--grape-global-count` that drops step 4's atomics and sync superstep. KaTric computes no LCC, so the
    default (as published, LCC) carries extra work; report both.
  - Pin `count_t` to 64 bits: `LCC32`/`LCC64` is picked by `--edge_num`, not by triangle count, and RMAT hubs could
    overflow 32 bits.
  - Native input is an edge list fed through `AddVertex`/`AddEdge` + `ConstructFragment`. `BasicLocalFragmentLoader`
    (the in-memory one) forces hash partitioning and no rebalancing; the opt driver uses the segment partitioner +
    edge rebalancing, i.e. a balanced 1D split. Find a way to keep KaGen's 1D distribution (segment partitioner
    given KaGen's ranges, or the regular loader fed per rank). This is the main integration work.
  - `preprocessing` is fragment construction (vertex map/idxer, CSR, ghosts) plus steps 1–2; `counting` is step 3
    (+ 4). Its own benchmarks time only the algorithm.
  - Ranks per node: one per node (or per socket) plus threads, its intended configuration. MPI-only (48 ranks per
    node) does not work as published, see the message pool below.

  Communication (`ParallelMessageManagerOpt`, `thread_local_message_buffer_opt.h`):
  - BSP rounds. Compute threads serialize `(gid, msg)` into per-thread channels, one buffer per destination rank
    (2 MiB block). A full buffer goes to a send queue, and a dedicated **send thread** `MPI_Isend`s it (tag =
    round), so sending overlaps the computation of the round. That early sending only kicks in when one thread
    sends more than one block (LCC uses the default, ~2 MiB) to one destination in a round, roughly
    V / (p · t) > 2 MiB. At high p in weak scaling almost everything goes out in `FlushMessages()` at the end of
    the round. The receiver stores early messages but processes nothing before the next round.
  - At the end of a round, every rank sends a zero-length end marker to **all** p − 1 others, then `MPI_Waitall`:
    O(p) messages per rank and round, even with sparse communication.
  - A **receive thread** loops on blocking `MPI_Probe(ANY_SOURCE)` and receives into pool buffers; the end markers
    count down the producers. Messages are processed only in the **next** round (`ParallelProcess`, deserialized by
    the compute threads), so all ghost neighbourhoods are buffered at once. No routing or indirection.
  - Termination: `MPI_Allreduce` over "sent something or `ForceContinue()`".
  - **Buffer memory is unbounded**, like TriC plain's `sbuf_`; the 2 MiB block only sets message granularity. The
    pool allocates a new chunk when empty (`MessageBufferPool::take()` never blocks). Sent chunks are parked in
    `this_round_to_others_` until the next round's `Prepare()`, and received chunks in `recv_bufs_stash_` until the
    next round has processed them. LCC's ghost exchange is one round, so a rank's peak holds all N⁺(v) copies it
    sent, all it received (raw), and then the ghosts' `complete_neighbor`: up to three copies of
    O(Σ_v |N⁺(v)| · |OEDests(v)|) volume. TriC plain holds O(remote wedges) instead.
  - Needs `MPI_THREAD_MULTIPLE` (`sync_comm.h` requests it). `main` asks for it only when GRAPE is selected.
  - Memory: `Worker::initPool` preallocates and zeroes a message pool of at least 16 MiB · (p + 1) · threads,
    capped at 60 % of the node's `MemAvailable` **per rank**. Each rank also has `hardware_concurrency()` channels
    with one buffer per destination. With many ranks per node this runs out of memory already at moderate p (e.g.
    p = 192, 1 thread: ≥ 3 GiB per rank, 48 ranks per node). Patching this on the fork would change GRAPE; use its
    intended rank layout instead and say so.

  Expect: it holds all ghost neighbourhoods at once (no buffered/async processing), so memory and skew at high p are
  where KaTric should differ. Its own numbers (LCC, `GraphalyticsReport.md`): graph500-28/29/30 in 60/137/318 s on
  8 cloud nodes; 9–36× faster than PowerGraph (`Performance.md`).

For each competitor:
- [x] TriC (see "TriC: findings" below)
- [x] HavoqGT
- [x] LCC
- [x] TriPoll / YGM
- [x] Tom2D
- [ ] CombBLAS
- [ ] GRAPE

Record for each, once it is done:
- fork SHA
- native input format
- what `preprocessing` and `counting` contain
- constraints
- validation results

### Review of the fork work (open, user)

Niklas still has to review all fork changes made with Claude (2026-11-21 to 2026-11-26). Every `katric` branch is pushed
and pinned in `extern/CMakeLists.txt` (LCC pins CLaMPI); a clean build of all five competitors from these SHAs works
(2026-11-26). Review with `git log -p <base>..<head>` in the sibling checkout; fixes are marked `// katric:`.

- [ ] `niklas-uhl/tric` `katric`: `6939dbe..f1dec08` (5 commits; library target, `namespace tric`, inrecv, bug fixes,
      `vertex_ranges.hpp` for `--tric-sparse-ranges`)
- [ ] `niklas-uhl/xxtric` `katric`: `eb797a3..4521532` (3 commits; CPU path as library, TriC's fixes)
- [ ] `niklas-uhl/havoqgt` `katric`: `fa09fbe..a06e177` (6 commits; library target, count split, `--havoqgt-stats`,
      MPI buffer and empty-rank fixes, bundled YGM → `havoqgt::ygm`)
- [ ] `niklas-uhl/lcc` `katric`: `42c30b0..cf0f9f1` (4 commits; CMake, `lcc_api.h`, window and degree-reduction fixes)
- [ ] `niklas-uhl/CLaMPI` `katric`: `bb55272..82a4e6c` (3 commits; CMake, macOS `aligned_alloc`, window-mode fix)
- [ ] `niklas-uhl/2DTriangleCounting` `katric`: `f704a53..db9fbd9` (2 commits; cherry-pick of `650d720` from
      `nuhl-integration`, static library + API split, counter and communicator fixes)
- [ ] `niklas-uhl/tripoll` `katric`: `ccfe8a0..437572c` (1 commit; port to YGM master, own `tripoll::ygm` target,
      drivers removed)
- [ ] The adapters in `apps/competitors/` and their findings sections below

### Upstreams (checked 2026-11-24)

| Fork | Upstream | State |
| --- | --- | --- |
| `niklas-uhl/tric` | `sg0/tric`, **archived** 2026-10-29 (final `6939dbe`), successor `sg0/xxtric` | was 72 commits behind; `katric` rebased onto `6939dbe` (old branch: `katric-2022`) |
| `niklas-uhl/havoqgt` | `llnl/havoqgt` master `2e8b2a8` (2021) | up to date; `develop` only adds HDF5/CC I/O |
| `niklas-uhl/2DTriangleCounting` | `KarypisLab/2DTriangleCounting` (frozen 2019) | up to date |
| `niklas-uhl/CombBLAS@46ebfc31` | `PASSIONLab/CombBLAS` master | contains all of master |

Upstream TriC since 2022: new variants `ibufastric`/`inbufastric` (single counting, the README's recommendation,
`-DAGGR_BUFR_INRECV`), self-loops skipped in `cfastric`, C++20. The three variants we use are unchanged apart from
that (`bufastric`, `hashfastric2` identical).

### Tom2D: findings (2026-11-26)

- **Fork:** `niklas-uhl/2DTriangleCounting`, branch `katric` on upstream master `f704a53` (2019, frozen) + a
  cherry-pick of `650d720` from `nuhl-integration` (heap request arrays instead of the hard `NPROCS = 1024`; without
  it Tom2D cannot run above 1024 ranks, which the TriPoll paper ran into). The rest of `nuhl-integration` (KaGen
  submodule and generator path, binary reader) is not needed. Static C library `tom2d::tom2d`, interface
  `include/tom2d.h`, built from `mpi_tc.c`, `io.c`'s exchange helpers and GKlib with GKlib's flags PRIVATE
  (`-O3 -march=native` outside Debug, as upstream's build; no `-Werror`). `GKlibSystem.cmake` (global
  `CMAKE_C_FLAGS`), `main`, `cmdline`, the readers, the Grappa RMAT generator and the committed build directory are
  gone. All other functions are `static`; exported are `tom2d_*`, `ancy_all2allv*` and GKlib's symbols (`gk_*`,
  `errexit`, `HTable_*`, ...), none of which anything else in katric defines.
- **Native input:** its readers' output, a 1D-distributed CSR (`vtxdist`, local `xadj`, 64-bit global ids in `adjncy`,
  both directions). KaGen's distribution is kept. Tom2D sorts the neighbourhoods itself (in preprocessing).
- **Regions:** `mpitc_MapJIK2D` split at its own timer boundary. `preprocessing` = `tom2d_preprocess` (its `timer_1`:
  cyclic redistribution, degree ordering by a distributed counting sort, upper/lower split, transposes); `counting` =
  `tom2d_count` (its `timer_2`: row/column communicators, Cannon's √p shifts with the kernel, global sum). It takes
  ownership of the CSR and frees it, so `reset()` hands it fresh `malloc`'d copies (untimed).
- **Constraints:** `MPI_COMM_WORLD`, perfect square p (both checked in the constructor), < 2^31 local vertices per rank
  (checked). Pure MPI (no OpenMP in the counter). Whether a global n ≥ 2^31 works is untested.
- **Fixed (marked `// katric:`, user decision):** the kernel's per-vertex counter `ltriangles` was `int32_t` (the
  authors' comment asks whether it should be `int64_t`); the row and column communicators were never freed (two per
  call); the count is `MPI_Allreduce`d instead of reduced to rank 0. Upstream also declares
  `mpitc_blobBytesBlockCyclic` as `int64_t` in `proto.h` but defines it `ssize_t` (only an error on macOS); on the
  fork the function is static with a matching prototype.
- **Validation:** `--verify --iterations 2` at 1, 4, 9 ranks: RGG2D, RHG, GNM, RMAT (n=2^14, m=2^17); ctests
  `katric.<graph>.tom2d.<1,4>cores` on the example set. The harness got a per-p expected count
  (`expected_<graph>_<p>`): KaGen's rgg2d is a different graph on 1 rank (416390 triangles = its reference count).
  Non-square p is rejected.
- **Against the original:** upstream `f704a53` + `650d720` (its own METIS reader) vs. the adapter (KaGen's METIS
  reader, `balance-vertices`), com-amazon, 4 ranks, Release, 5 runs each: preprocessing 0.044–0.049 s vs.
  0.041–0.050 s, counting 0.009 s vs. 0.008–0.010 s, same count. Within noise.

### TriPoll: findings (2026-11-26)

- **YGM version:** TriPoll's CMake fetched YGM's `develop` branch, which no longer exists, so upstream TriPoll does not
  build today. Its code was last updated 2023-07 "to current YGM develop", which YGM squash-merged into v0.6. Both v0.6
  (one missing `<algorithm>` include) and master were easy to build; the user chose **YGM master** (`0072d87`, v0.9 +
  hubcast, 2026-08) as the fairer comparison (2.5 more years of YGM's communication layer), no A/B against v0.6.
- **Fork:** `niklas-uhl/tripoll`, branch `katric` on `LLNL/tripoll` main `ccfe8a0` (MIT): INTERFACE `tripoll::tripoll`
  over `tripoll::ygm`, an INTERFACE target the fork declares over YGM's `include/` (YGM's own CMake is not run: it
  downloads Boost from source and builds Arrow/Parquet and Boost.JSON, none used here). Dependencies: MPI, Boost
  headers, spdlog (header-only; katric's), cereal v1.3.2 headers. Drivers, RMAT generator and `hash.hpp` removed.
  Port to YGM master, no algorithmic change: `ygm::utility::timer`, `YGM_ASSERT_RELEASE`, `partitioner.owner`,
  `gather_topk`, missing std includes, `std::max<uint64_t>` (macOS `size_t` ≠ `uint64_t`).
- **Native input:** an edge list fed to `undirected_graph::async_add_edge`, which stores both directions on the ranks
  YGM's hash partitioner picks. `ingest` keeps only u < v (KaGen has both directions), so the drivers'
  `uniquify_edges` (sort + dedup for RMAT's duplicates) is skipped, as LCC's `mirror()`.
- **Regions:** `preprocessing` = `async_add_edge` of all edges + `make_dodgr` (degree order, `uint32_t` as its drivers;
  the drivers call this preprocessing too); `counting` = `tc_push_only` / `tc_push_pull` + a sum. Both graphs live only
  in `run()`, `reset()` is a no-op. The `ygm::comm` is created once in the constructor (untimed), from katric's
  communicator (no `MPI_COMM_WORLD` restriction, no MPI thread level: YGM has no threads).
- **Options:** `--tripoll-variant push-pull|push-only` (default push-pull), `--tripoll-routing none|nr|nlnr` (sets
  `YGM_COMM_ROUTING` before the communicator is created; default `none`, YGM's default).
- **The paper's setup** (Steil et al., SC'21, `~/Documents/References/Steil2021`; the repo has no experiment scripts):
  - Push-Pull everywhere (strong and weak scaling, the competitor table); push-only only as the baseline of its §5.10.
  - YGM of 2021 aggregated per rank only, i.e. today's `none`. The authors blame their slowdown from 128 to 256 nodes
    on that and name node-level aggregation as the remedy (§5.4). The later YGM paper (Steil et al. 2023) shows the
    effect of routing on an embedding benchmark from 64 to 512 nodes: parallel efficiency 26 % without routing, 61 %
    NR, 98 % NLNR.
  - 24 ranks per node (all cores of Catalyst, MPI only) for the scaling runs; 1024 cores on 64 nodes (16 per node) for
    the comparison with HavoqGT (Pearce et al.), Tom2D and TriC. YGM's logger is off and its
  welcome message is not printed by default.
- **As published, not changed:** the triangle callback is called through an uninitialized pointer to its (stateless)
  type, so it counts into a global; `tc_push_*` print a few `cout0` lines; their `static num_edges_sent` is never reset,
  so the "Sent a total of N edges" line accumulates over iterations (log only).
- **Validation:** `--verify --iterations 2` at 1, 2, 4 ranks, both variants: RGG2D, RHG, GNM, RMAT (n=2^14, m=2^17); ctests
  `katric.<graph>.tripoll-{push-only,push-pull,nlnr}.<2,4>cores` on the full example set (24 tests). The comparison
  with the original driver is not possible: upstream does not build (see YGM version).

### HavoqGT: findings (2026-11-24)

- **2026-11-26:** HavoqGT's bundled old YGM (mailboxes, `comm_exchanger`, `mpi.hpp`) moved to `include/havoqgt/ygm/`,
  namespace `havoqgt::ygm`, so that it cannot clash (ODR) with the real YGM that TriPoll links into the same binary.


- **Fork:** `niklas-uhl/havoqgt`, branch `katric` on upstream master (via `fa09fbe3`): INTERFACE target
  `havoqgt::havoqgt` (MPI, Boost headers, Threads); Metall, KaGen, CGAL, UMap, HDF5, executables and tests dropped.
  Boost.Interprocess stays: `bip::vector`/`offset_ptr` let the graph live in a mapped heap (Metall); with
  `std::allocator` they act as `std::vector` / raw pointer, and Boost is needed anyway (unordered, container, random).
- **Native input:** an edge list (`tuple<u64, u64, double>`, both directions), distributed by HavoqGT itself
  (vertices cyclically, delegates above `--havoqgt-delegate-threshold`, default 2^20 as the old driver).
- **Regions:** `preprocessing` = `delegate_partitioned_graph` construction + `triangle_count_orient()` (directed
  2-core); `counting` = `triangle_count_wedges()`. The fork splits `new_triangle_count()` into these two; its
  diagnostics (allreduces, degree histograms) run only with `--havoqgt-stats` (default off). Everything is rebuilt
  per run, `reset()` is a no-op.
- **Constraints:** `MPI_COMM_WORLD` only; equal ranks per node in contiguous blocks (checked, YGM's nl/nr mailbox).
  `havoqgt::init` (node-local communicators, O(p) rank-by-rank) runs once in the constructor, untimed.
- **Fixed (marked `// katric:`):** missing `offset_ptr.hpp` include (Boost 1.92); `&v[0]` → `v.data()` on MPI
  buffers (empty vectors abort with GCC ≥ 15's default `-O0` assertions); `vertices_end()` stepped below index 0 on
  a rank without vertices (more ranks than vertices).
- **Validation:** `--verify --iterations 2` at 1, 2, 4 ranks: triangle, hiv, com-amazon, RGG2D, RHG, GNM, RMAT; 16
  ctests incl. `--havoqgt-stats`.

### xxtric: findings (2026-11-24)

- **Fork:** `niklas-uhl/xxtric`, branch `katric` on `sg0/xxtric` main `eb797a3`: INTERFACE target `xxtric::xxtric`
  (MPI, C++20), the CPU path (`deftypes`, `utils`, `graph`, `xxtric.hpp`) inline-safe, self-contained and in
  `namespace xxtric`. CUDA variants, `main.cpp` and Makefiles untouched and unbuilt.
- **Native input, regions, `reset()`:** as TriC's buffered variants (same `Graph`, `edge_stat_` rebuilt).
  `--xxtric-buffer-size` defaults to its `-1` (2 × ghost entries, no cap).
- **Fixed as on TriC** (marked `// katric:`): degree-0 vertices read `edge e1-1` out of range (RMAT aborted); the
  buffer shrank to 2 with one remote wedge and `count()` never ended (`triangle.metis` on 4 ranks);
  `Graph::operator delete` called `free()` on memory from `operator new` (removed).
- **Validation:** `--verify --iterations 2` at 1, 2, 4 ranks, default buffer and `--xxtric-buffer-size 8000`:
  triangle, hiv, com-amazon, RGG2D, RHG, GNM, RMAT. ctests `katric.<graph>.xxtric[-buffer-size].<p>cores`.

### LCC: findings (2026-11-24)

- **Forks:** `niklas-uhl/lcc` `katric` on `fvella/lcc` master `42c30b0` (2022, inactive), `niklas-uhl/CLaMPI` `katric` on
  `spcl/CLaMPI` master `bb55272` (2021). Both GPL-3.0 (decision 5, README note). CMake instead of autotools:
  `lcc::lcc` (static C, built as `--with-simd --with-clampi`, the paper's configuration, OpenMP) fetches
  `clampi::clampi` (static C, adaptive scheme on as its README prescribes for x86).
- **Library calls** (`include/lcc_api.h`) replace `main()`: `lcc_setup` (1 x p mesh as in the paper, N rounded up to a
  multiple of p with isolated vertices, communicators, `cl_init` once), `lcc_preprocess` (2D partitioning, which with
  R = 1 is a block of N/p consecutive vertices per rank; `norm_graph` sort/dedup; CSC), `lcc_count` (the kernel
  `lcc_func_bin_simd` plus a global sum of its per-vertex triangle counts / 3; LCC itself only computes LCC values),
  `lcc_free`. Every other symbol of `lcc.c` is `static`; the library exports only these four.
- **Native input:** LCC's reader output, a chunk of `(u, v)` pairs per rank. KaGen's edges come with both directions,
  so LCC's `mirror()` is skipped (it would only add duplicates that `norm_graph` removes).
- **Regions:** `preprocessing` = setup + preprocess; `counting` = `lcc_count` (window creation, CLaMPI caches, kernel,
  allreduce), as LCC's own timer around `lcc_func_bin_simd`. The edge list is copied in `reset()` (untimed), since LCC
  takes ownership and reallocates it. Everything else is rebuilt per run.
- **Options:** `--lcc-cache-size` (bytes per rank for CLaMPI's caches, allocated up front; LCC hard-coded 8 GiB, which
  rules out many ranks per node: 1 rank per node + OpenMP threads is its natural layout), `--lcc-degree-reduction`
  (its `-H 1`, off by default). OpenMP threads via `OMP_NUM_THREADS` (ctests: 1).
- **Constraints:** `MPI_COMM_WORLD`; N < 2^32 (local and remote ids are 32-bit, `MPI_UINT32_T` in the `Get`s), so the
  largest weak-scaling inputs cannot run; at most 1024 ranks (`MAX_PROC_J`); < 2^31 edges per rank (`int ned`); O(N)
  arrays per rank (`row_pp = N`: receive buffers, `reach`, degrees; CLaMPI index 3.2 N bytes).
- **Fixed (marked `// katric:`):**
  - LCC: the offsets window had `col_bl` entries, the kernel reads `col[v + 1]` (MPI_ERR_RMA_RANGE for the last local
    vertex with Open MPI); `nglobal_ed` from an `int` passed as `MPI_UINT64_T`.
  - CLaMPI: the window mode was read into one `char` (1 byte overflow) and compared with / assigned string constants,
    so every window was effectively *always-cache*; fixed, and LCC now requests always-cache explicitly (same
    behaviour as the published runs; transparent mode would invalidate the cache on every flush). The global
    `cl_conf_t clampi` defined in a header (fails with `-fno-common`).
  - Portability only (Linux/x86 unchanged): SSE `memcpy` only on x86_64, `aligned_alloc` size rounded on macOS,
    `cpu_set_t` code only on Linux.
  - LCC, `--lcc-degree-reduction`: a rank without edges after its 1D split scanned with `nedge - 1` wrapped around
    and read `r_ed[-2]` (`triangle.metis` on 4 ranks); removing the edges of removed vertices read `o_ed[2 * pnod]`
    before checking `pnod < nod` (up to two entries past the list).
- **Validation:** `--verify --iterations 2` at 1, 2, 4 ranks (`OMP_NUM_THREADS=1`, 64 MiB cache), with and without
  `--lcc-degree-reduction`: triangle, hiv, com-amazon, RGG2D, RHG, GNM, RMAT. 16 ctests.

### TriC: findings (2026-11-21, rebased 2026-11-24)

- **Rebase (2026-11-24):** `katric` = upstream `6939dbe` + library target + fixes + `namespace tric` (all headers;
  std includes hoisted in `utils.hpp`). New `--tric-variant inrecv` (`TriangulateAggrBufferedInrecv`): counts each
  triangle once (precedence by vertex id). It had the degree-0 read and the buffer-of-2 livelock (`triangle.metis`
  on 4 ranks) too; both fixed as in the other variants. `--tric-sparse-ranges` is only implemented for
  buffered/hash2 and rejected otherwise.

- **Fork:** `niklas-uhl/tric`, branch `katric` @ `4fb1e5d1` (on top of `main` `6049cb15`): library target `tric::tric`,
  no `main`, no KaGen, header globals `inline`, `AGGR_BUFR` always defined (`Graph` then has `edge_stat_`, so all
  variants coexist in one TU), and the bug fixes below marked `// katric:`.
- **Native input:** TriC's `Graph` as its binary reader (`BinaryEdgeList::read`) builds it: 1D vertex distribution
  (KaGen's), local CSR with 64-bit global ids, **sorted** neighbourhoods. Sorting happens in `ingest`.
- **Regions:** `preprocessing` constructs the variant's `Triangulate*` object (buffer sizing; for the buffered variants
  also the `erange_` allreduce and the local counting). `counting` is `count()`. `count()` accumulates into its
  object, so every run constructs a new one; `clear()` (TriC's own free) runs at the end of `total`.
- **reset():** the buffered variants mark edges done in `Graph::edge_stat_`, so `reset()` rebuilds it. The plain
  variant does not modify the graph.
- **Variants:** `plain` = `TriangulateAggrFatCompressed` (the paper's `tric`), `buffered` = `TriangulateAggrBuffered`
  (the paper's `tric-buffered`: its CMake target set `AGGR_BUFR` and, by a typo meant for `tric-new`, `AGGR_HASH2`;
  `main.cpp` checks `AGGR_BUFR` first), `hash2` = `TriangulateAggrBufferedHash2`.
- **Constraints:** none on p. TriC counts every triangle three times (no orientation) and divides by 3, so it needs a
  simple graph: parallel edges or self-loops give wrong counts.

Bugs in TriC (fixed on the fork branch, marked `// katric:`), all correctness or UB, none changes the work done:
- Degree-0 vertices: the buffered constructors skip only `(e0 + 1) == e1`, so for an isolated vertex they read
  `edge_list_[e1 - 1]`, out of range at the ends of the local edge array. RMAT aborted with `_GLIBCXX_ASSERTIONS`.
- `bufastric.hpp` zeroes only `erange_[0, nv)` of its `2 nv` entries before the `MPI_SUM` allreduce. Fresh `mmap`
  pages hide it; reused memory (a second iteration) gives garbage ranges, and `edge_within_max()` then prunes or
  keeps wedges wrongly.
- `free()` on memory from `new[]` (buffered and hash2).
- hash2 shrinks its buffer to `min(2 nghosts, -s)` (max over ranks). With one remote wedge that is 2 entries, too few
  for vertex + neighbour + `-1` marker: `lookup_edges()` sends `[v, -1]` forever and `count()` never ends
  (`triangle.metis` on 2 ranks). Fixed by a floor of 3; inputs with more than one remote wedge per rank are unaffected.

Bugs in the old fork's KaGen path (`run_kagen`, removed on the branch), relevant to the IPDPS'23 TriC numbers on
generated graphs:
- **No sorting:** `check_edgelist()` stops at the first neighbour larger than the query, so unsorted neighbourhoods
  miss triangles. With today's KaGen and the sort disabled: RGG2D (n=2^14, m=2^17) 87,573 instead of 416,390
  triangles, GNM 123 instead of 631; RHG and RMAT happened to be right. Fewer found triangles also means less work. To
  check: were the paper's generator runs verified, and was the old KaGen's CSR sorted?
- `Graph(xadj.size(), ...)` makes `lnv_` one too large, so the loops visit a vertex `n_local` whose
  `edge_indices_[n_local + 1]` is read out of range (UB; usually an empty range).
- `edge_stat_` is never filled, so the buffered variant cannot have run on generated graphs.

Behind a switch, `--tric-sparse-ranges` (default off = TriC as published):
- Buffered and hash2 keep `erange_`, the first and last neighbour of **every** global vertex, on every rank, built by
  an `MPI_Allreduce` over `2 n` entries: O(n) memory and communication per rank, which grows linearly in weak scaling
  (2^20 vertices per rank on 2^14 ranks: 256 GiB per rank). They only look up neighbours of local vertices, so
  `vertex_ranges.hpp` on the fork holds the ranges of the local vertices and the ghosts, the latter fetched from their
  owners (alltoallv of requests and answers): O(local + ghosts) memory, O(ghosts) communication. The values equal
  `erange_` (checked on every lookup with a temporary abort-on-mismatch build), so pruning, messages and counts are
  unchanged. Cost: a ghost's range is a hash lookup instead of an array access.

Performance issues, **not** changed (they are TriC as published):
- Buffered: the constructor finds each remote edge's owner in `targets` with `std::find` (O(p) per remote edge);
  `count()` polls with `MPI_Testsome` over all p requests and rescans all local vertices in every `lookup_edges()`
  call, once per buffer refill. On com-amazon with 2 ranks (debug build) this is 9.5 s against 0.4 s for plain.
- Plain: the constructor counts each vertex's wedges with an O(deg^2) loop instead of `e1 - m - 1`, and `sbuf_`
  holds all remote wedges at once (O(wedges) memory, the documented design).
- `Edge` carries a `double` weight: 16 bytes per edge.

Validation: `--verify --iterations 3` at 1, 2, 4 ranks, all three variants: examples (triangle, hiv, com-amazon),
RGG2D, RHG, GNM, RMAT (n=2^14, m=2^17). ctests `katric.<graph>.tric-<variant>.<p>cores`. Still open: the
adapter-vs-original-`main` timing comparison and the SuperMUC comparison with the IPDPS'23 logs.

## Experiments and eval

- [ ] `experiments/competitors-real-world.suite.yaml` and `competitors-weak-scaling.suite.yaml`. They `import` the
      existing instance sets, so the inputs are the same as KaTric's. The old log2 generator-flag problem is gone,
      because the input comes from `--kagen_option_string`
- [ ] Separate suites for square-p competitors (Tom2D, CombBLAS) with kaval's `--cores sqr` / `sqr-pow2`, as in
      KaCCv2's `paper-combblas`
- [ ] TriPoll runs both `--tripoll-routing none` (the paper's setup) and `nlnr` (its best on current YGM at high p);
      user decision 2026-11-26
- [ ] A HavoqGT suite with 32 tasks per node (check that kaval supports `tasks_per_node` per suite)
- [ ] Core ranges go only in the suite header comments (no `max_cores`)
- [x] (TriC) `eval/katric_eval.py`: the `variant` column takes the `algorithm` value (plus `tric-variant` and similar).
      Nothing else changes, since the output format is the same
- [ ] `experiments/local/smoke.suite.yaml`: each enabled competitor with `--verify` (core counts 1 and 4 for the 2D
      ones)

## Verification, per competitor

- **Build:** `cmake --preset dev -DKATRIC_COMPETITOR_<NAME>=ON`. With every option OFF, the build is unchanged.
- **Correct counts:**
  - `--verify` on `examples/` and on small KaGen graphs (RGG, RHG, GNM, RMAT), at 1, 2 and 4 ranks (1 and 4 for
    square p).
  - A ctest per enabled competitor, like `tests/check_triangle_count.cmake`.
- **The adapter does not slow it down:**
  - On one mid-size input, compare the adapter's `counting` time with the fork's original `main` at its old commit.
    They should be within noise.
  - This catches conversions that hand over a different data layout, such as unsorted neighbourhoods where the
    competitor expects sorted ones. In that case, the competitor's own sort belongs in `preprocessing`.
- **Independent iterations:** `--iterations 3` gives the same count every time, and the second and third
  iterations are not systematically faster than the first.
- **Cluster:** a SuperMUC run of the TriC suite at small scale, compared against the IPDPS'23 TriC numbers from the old
  logs.
