# katric modernization plan

Working checklist for the `chore/modernize` branch. Tick items (`[x]`) as they land and add the commit hash.

## Goal

Bring katric's build, measurement, input and experiment setup in line with kascade and KaCCv2. That means
FetchContent for every dependency, CMake 3.25 presets, KaGen input and kaval as the experiment runner. Measurement
deliberately differs: Caliper/Adiak with JSON output instead of kamping timers (see step 3). Where those projects have weak spots, do it better here instead of copying them
(see "Where katric should do better").

The algorithms stay unchanged. Every step is checked by comparing triangle counts on fixed inputs and seeds.

## Done so far

| Commit    | Change                                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------------------ |
| `2814716` | Debug and progress output via spdlog v1.16.0 + kamping-spdlog-adapter; compile-time level `CETRIC_ACTIVE_LOG_LEVEL` |
| `ee5719b` | Derived MPI datatypes via kamping-types, `mpi_traits` removed                                                |
| `af6f9f0` | Boost.MPI replaced by kamping-types                                                                          |
| `25be39d` | Empty `.gitmodules` and `load_submodules.sh` removed, README updated                                         |
| `ff681d8` | All dependencies via FetchContent; backward-cpp, sanitizers-cmake and others dropped                         |

The Caliper/Adiak measurement (step 3) was developed on `explore/caliper-adiak` and merged.

## Work items

### 1. CMake structure

- [x] Add `CMakePresets.json` (v6 for CMake 3.25; `dev` with warnings and `experiments`, both in `build/`; dev test +
      workflow presets) and a `CMakeUserPresets.json.template` (compiler, local checkouts); gitignore
      `CMakeUserPresets.json`
- [x] `KASSERT_ASSERTION_LEVEL` actually reaches the code (kassert reads a target property, not the variable, so
      katric's assertions were never at level 30 in Debug)
- [x] KaMPIng: `kamping::types` did not define its assertion level, so every TU including kamping-types printed a
      `#warning`. Fixed upstream (kassert v1.1.0 `KASSERT_DEFAULT_ASSERTION_LEVEL`, kamping v0.3.2 with a kassert
      instance of its own for kamping-types, kamping-v2 main); katric pins these, and `KASSERT_ASSERTION_LEVEL` is now
      the default of katric's kassert instance instead of a definition on `cetric-lib`
- [x] Replace the `PUBLIC -Wall -Wextra -pedantic` on `cetric-lib` with a `<project>_cxx_flags` interface target
      driven by a `*_CXX_FLAGS` cache variable
- [x] Make `cetric-lib` an `INTERFACE` library (it only compiles an empty `null.cpp`); list headers in a
      `FILE_SET HEADERS` (the current glob uses the stale `${parallel-triangle-counter_SOURCE_DIR}` and matches nothing)
- [x] Use `PROJECT_IS_TOP_LEVEL`; add `CETRIC_BUILD_TESTS` / `CETRIC_BUILD_APPS` options; set the C++ standard per
      target via `target_compile_features` (C++20 is needed for kamping-v2 later)
- [x] Replace `cmake/CheckGit.cmake`, `FileEmbed.cmake`, `.degit-sync` handling, `buildfile_embed/` and
      `apps/git_hash.*` with cmake-git-version-tracking
- [x] Replace `scripts/get_clang_format.sh` and the other format scripts with Format.cmake (opt-in option
      `CETRIC_ENABLE_FORMAT_CMAKE`). ccache and debug containers are not project options; set
      `CMAKE_CXX_COMPILER_LAUNCHER` / `-D_GLIBCXX_DEBUG` locally if needed
- [x] Pin graph-io (`fix-modern-toolchain`) and message-queue (`katric-legacy`) to commit SHAs instead of branch names
- [x] Move all `FetchContent_Declare` calls into `extern/CMakeLists.txt`, including app/test-only deps
      (cmake-git-version-tracking, KaTestrophe, Format.cmake)
- [x] (Boost was removed later) Boost and oneTBB: `find_package` first, FetchContent (Boost 1.92.0 `iterator` + `range`, oneTBB v2023.2.0) as
      fallback; system packages linked last so `/opt/homebrew/include` cannot shadow the pinned fmt. All other
      dependencies are always fetched (CLI11 `FIND_PACKAGE_ARGS` dropped, bumped to v2.5.0)
- [x] MPI tests: register only rank counts `<= MPIEXEC_MAX_NUMPROCS`
- [x] KaTestrophe v1.2.0 (declared before kamping-v2, which pins an older one): `KATESTROPHE_MAX_CORES` (cache,
      `MPIEXEC_MAX_NUMPROCS`) replaces katric's own core filter; tests call KaTestrophe directly, no wrapper functions
- [ ] Remaining configure warning (old `cmake_minimum_required`): KaGen's bundled xxHash (check a newer KaGen release)
- [x] C++20 (`cxx_std_20` on `cetric-lib`). katric had always been compiled as C++17: legacy-message-queue forced
      `INTERFACE -std=c++17` (fixed there and in graph-io via `target_compile_features`); `GridPosition::operator==` made
      `const` (C++20 reversed-operator ambiguity)
- [x] End-to-end tests: katric on `examples/` (triangle, hiv, com-amazon) x cetric/ditric/ditric-grid/hybrid x 2/4
      ranks, checking exit code and triangle count (`tests/check_triangle_count.cmake`)
- [x] Replace Boost ranges/iterators with `std::ranges` and drop Boost (concat of local nodes + ghosts is an index
      loop now; `AuxiliaryNodeData::range()` is materialized, debug output only)
- [ ] When TBB is fetched, it is built as a static library (oneTBB warns against that); decide whether to build it shared
- [x] Decide on the cetric → katric rename: keep cetric for now

### 2. Hash tables: Abseil

- [x] Option `CETRIC_USE_ABSEIL` with a sparsehash/Abseil switch (`4847c53`), validated: all tests pass, all
      non-timing counters identical
- [x] Abseil only: sparsehash, the switch, `cetric::hash` (MurmurHash64A) and utils_tm dropped; `absl::Hash` with an
      `AbslHashValue` hook in `RankEncodedNodeId`. This also fixes `default_map<NodeId, ...>` hashing 64-bit ids
      through an `int` overload. Counters again identical
- [x] macOS: abseil's `_XOPEN_SOURCE` patch applied via `cmake/apply_patch.cmake` (skips if applied, fails loudly
      otherwise); abseil linked before system packages (Homebrew's abseil headers shadowed it)
- [x] Dead `compact_graph.h` / `simple_graph.h` removed instead of ported
- [ ] Measure run time against the sparsehash version (`4847c53` without the option) on the cluster before merging

### 3. Measurement and JSON output: Caliper/Adiak

- [x] Phases as scoped Caliper regions under a top-level `total` per iteration; counters added on the fly
      (`counter_add`) in the edge iterator and every communication step; cross-rank min/max/avg (sum for counters)
      written as JSON via `--caliper-output <prefix>` → `<prefix>.timers.json` / `<prefix>.counters.json`
      (`74defff`, `c1037d6`, `b9b5028`, `0941c41`). `spot` and Caliper's other configs work via `--caliper-config`
- [x] Run metadata via Adiak: the CLI11 option values (enum options named via magic_enum) and Adiak's standard
      collectors; cereal, `Statistics` (with the broken `LocalStatistics` copy) and `--json-output` removed
      (`9db1c7f`, `03fb46f`)
- [x] Configure with Python < 3.15 for Caliper's MPI wrapper generator (`321a55a`)
- [ ] Measure the profiling overhead and check the Linux-only Adiak collectors (command line, libraries) on the
      cluster
- [ ] Make kaval read the JSON (`scripts/caliper_to_json.py` merges both files into a per-iteration document);
      until then `experiments/runners.py` does not work (it needs `--json-output`), accepted until the kaval switch

### 4. Graph input: KaGen

- [x] `--kagen_option_string` (required) replaces `input`, `--input-format`, `--gen*`, `--read-edge-partitioned` and
      `--cache-input`; KaGen's CSR output is converted to the `LocalGraphView` (`apps/input.h`). Validated
      against the previous commit: triangle counts identical on examples + rgg2d/rhg/gnm (fixed seeds, 2 and 4 ranks);
      all counters identical except the (already nondeterministic) grid policy (`870128a`)
- [x] `read-edge-partitioned = true` in `configs/cetric.toml`/`ditric.toml` is now `distribution=balance-edges` in the
      option string (counters identical with it; KaGen's default `balance-vertices` gives a different split). Experiment
      inputs must pass it to reproduce the paper setup. Unknown keys in `--config` files are errors now (CLI11 silently
      ignored the removed key)
- [x] KaGen does not read the binary `first_out`/`head` format; the examples all have METIS versions, the binary
      copies are removed. Convert large inputs to ParHIP
- [x] Pre-partitioned input (`--partitioning`) dropped: KaGen's distributions (`balance-edges`, `explicit` vertex
      ranges) are enough. `examples/partitions` removed; the `partitioned: True` suites become obsolete (step 5)
- [x] ctest: file inputs via KaGen (`balance-edges`) plus an rgg2d generator input
- [x] graph-io removed. katric has its own `LocalGraphView` (`datastructures/local_graph_view.h`, the input and
      load-balancing exchange format; `first_out()` replaces graph-io's hash-map indexer) and `BasicEdge`;
      `MPI_NODE` is plain `MPI_UINT64_T` (the typo'd `#ifdef MPI_UNINT64_T` always fell back to
      `MPI_UNSIGNED_LONG_LONG`). The KaGen conversion moved to `cetric/input.h`; the unit tests read the examples
      via KaGen (`tests/test_graphs.h`). No input post-processing is left: KaGen's output is only converted.
      Counters identical to the previous commit. Not done: building `DistributedGraph` from `kagen::Graph`
      directly; kept `LocalGraphView` because the load balancing exchanges it and its message volume is counted

### 4b. Benchmark harness: abstract algorithm (like KaCCv2's `benchmark/`)

KaCCv2's runner generates the input once, hands it to an `AbstractAlgorithm` (`ingest`, `reset`, `run`, result
getters) built by a factory (`get_algorithm(config, comm)`), and verifies against a reference. katric's `main`
instead re-reads or regenerates the input in every iteration (`--cache-input none` is the default), rebuilds the
`DistributedGraph` per iteration and dispatches algorithm x communication policy in a 6-way if-chain.

- [x] `generate_input(option string, comm) -> kagen::Graph`, called once before the iteration loop; `InputCache` and
      `--cache-input` removed. Each iteration copies the unmodified `LocalGraphView` and builds the `DistributedGraph`
      from it (as `--cache-input in-memory` did) (`870128a`). Generation time is now the Caliper region `input`
      (the `input:` log line and Adiak `input_time` are gone)
- [x] `AbstractAlgorithm` with `ingest(kagen::Graph&&)`, `reset()`, `run()`, `triangle_count()`
      (`apps/algorithm.h`); `make_algorithm(conf)` replaces the if-chain (cetric / patric / cetric-x x message-queue /
      grid policy)
- [x] katric's algorithms modify the graph (orientation, `remove_internal_edges`, `compact`, ghost expansion), so
      `ingest` keeps a pristine `DistributedGraph` and `reset()` copies it into the working graph. Same memory as
      the old `--cache-input in-memory`, but the conversion runs once. Caliper regions: `input`, `verify`, `ingest`
      once, `reset` per iteration, `total` unchanged (so run times stay comparable). The `wedges` counter is computed
      once in `ingest`. Validated against `870128a`: triangle counts and all counters identical (com-amazon, rgg2d,
      rhg x cetric/ditric/hybrid/patric/cetric-x, 4 ranks)
- [ ] Optional `--reread-input`: call `generate_input` + `ingest` before each iteration instead of keeping the
      unmodified copy, for inputs where two copies do not fit (fast with KaGen's binary ParHIP format; convert the
      large inputs to it)
- [x] `--verify`: gather the graph on rank 0 and count sequentially (small inputs only, `apps/verify.h`), like
      KaCCv2's `BFSCC` reference; checks every iteration, exits with 1 on a mismatch. The end-to-end ctests pass it
- [ ] Competitors behind the same interface (later, not part of this step): TriC, HavoqGT, Tom2D and Bader's code
      run today as separate binaries via `runners.py`. Each needs a converter from `kagen::Graph` to its own input
      format and a pinned fork via FetchContent (`EXCLUDE_FROM_ALL`), like KaCCv2's ParConnect/CombBLAS/KaCCv1.
      Decide per competitor whether hooking it in is worth it or kaval runs its binary

### 4c. Static analysis: clang-tidy

- [x] `.clang-tidy` modelled on KaCCv2's, `HeaderFilterRegex: 'include/cetric/'`; `bugprone-chained-comparison`
      disabled (fires on every `KASSERT(a <op> b)`); `.clangd` adds `-std=c++20` (`08524a7`)
- [x] `CETRIC_ENABLE_CLANG_TIDY` sets `CXX_CLANG_TIDY` on katric's executables and tests only, with
      `--extra-arg=-std=c++20` (g++-16 defaults to C++20, so CMake passes no `-std` flag). Homebrew's clang-tidy
      analyzes against libc++, not libstdc++ 16; good enough for finding bugs (`08524a7`)
- [x] fmt 11.0.2 -> 12.2.1: the "does not compile with fmt 12" note was stale, and fmt 11.0.2's consteval format
      strings broke clang-tidy in spdlog's error handler (`6d81118`)
- [x] Bug-class findings fixed: forwarding of the comparator, `throw` of a string literal, identical branches,
      moved-from `node_range` (`61fb64a`); `compact()` set the edge count from the moved-from `head_`, so the
      compacted graph reported 0 local edges; counts and counters unchanged on com-amazon (`db45af4`)
- [ ] Remaining findings, none a confirmed bug: ~98 `bugprone-narrowing-conversions` (`size_t` -> `int` MPI counts
      and displacements in `graph_communicator.h`, `cetric.h`; overflow above 2^31 elements per PE, same as
      plain MPI), uninitialized members of default-constructed graphs/configs (`cppcoreguidelines-pro-type-member-init`;
      always assigned before use today), `InputCache` (goes away with `--cache-input`, step 4b)
- [ ] No mass `modernize-*`/`readability-*` rewrite of the algorithm headers (noisy diffs); apply those only to code
      that is being touched anyway (apps, new harness)
- [ ] Optional: a CI job on the bug-finding checks only

### 5. Experiment runner: kaval

- [ ] Fetch kaval into `./kaval` like kascade/KaCCv2, pinned to a SHA and gitignored; add `justfile` and
      `pyproject.toml` / uv setup
- [ ] kaval passes `--caliper-output <run prefix>` and reads the JSON back. Decide then whether `--caliper-output`
      alone should also write spot's `.cali` (for Thicket), i.e. default to `spot,katric-timers,katric-counters`
- [ ] Convert `experiments/paper-suites/*.suite.yaml` and `all-suites`; check `threads_per_rank` and the graph specs
      against kaval's format
- [ ] Remove `expcore.py`, `runners.py`, `run-experiments.py`, `simple_runner.py`, `slugify.py`, the templates,
      `*.env` and `cetric-from-config.py` (keep `configs/*.toml`: they are CLI11 algorithm presets for `--config`)

### 6. Cleanup

- [x] `shmetric` removed, with what only it used: `shared_memory_cetric.h`, `AdjacencyGraph`
      (`datastructures/graph.h`), the `DistributedGraph(graphio::Graph&&)` constructor, `shmetric-benchmark.cpp`
      and `experiments/run_shmem.sh` (`a5328cf`)
- [ ] Add a GitHub Actions job that builds and runs the tests
- [ ] Rewrite the README: presets, kaval usage, `--kagen_option_string`

## Where katric should do better than the reference projects

| Issue in kascade / KaCCv2 | Where | Do in katric instead |
| --- | --- | --- |
| Dependencies on moving branches: kamping and kamping-v2 on `main`, several `main`/`master`/`fix/...` pins in `benchmark/CMakeLists.txt`, kaval on `feat/horeka2` | KaCCv2, kascade | Tags or SHAs only, with a comment naming the branch (KaCCv2's own BriefKAsten comment explains why branch pins break) |
| The abseil patch is applied with `\|\| true`, so a failed patch goes unnoticed | both | Apply it only on Apple, or check with `--dry-run` and fail loudly |
| The same macOS patch file is copied into each repo (`cmake/patches/` vs. repo root) | both | Keep it in `cmake/patches/`; long term, upstream it or share a small CMake helper repo |
| Log and assertion levels passed as `-D` flags inside `CMAKE_CXX_FLAGS` strings in presets | both | Real cache options (`CETRIC_ACTIVE_LOG_LEVEL` exists; add one for KASSERT); presets set options, not raw flags |
| `dev` and `experiments` presets share `binaryDir: build`, so switching presets overwrites the cache | kascade | Separate build dirs per preset (`build/dev`, `build/experiments-$env{SLURMD_NODENAME}`) |
| Generator differs between presets and projects (Makefiles vs. Ninja) | both | Ninja for `dev`; Makefiles only where a cluster lacks Ninja |
| Bench-only dependencies declared in a subdirectory, separate from the rest | both | All `FetchContent_Declare` calls in one place |
| Generated results committed: 41 PDFs, `.out` logs and `~` backup files in `eval/` | kascade | Keep plots and raw output out of git |
| Working notes and handover files at the repo root (`TAKEOVER_*.md`, `*_pitch.md`) | KaCCv2 | Notes go in `doc/` |
| No CI | both | GitHub Actions build + test job |

## Open decisions

- [x] Measurement: Caliper/Adiak instead of kamping timers (step 3)
- [x] Keep or drop `shmetric`? Dropped
- [ ] Abseil: opt-in until the counts match (current plan), or the default right away?
- [x] Drop `--cache-input` entirely. If two in-memory copies do not fit, an optional `--reread-input` can run
      `generate_input` again before each iteration instead of keeping the unmodified copy; with KaGen's binary
      ParHIP format, rereading is about as fast as the old cache
- [x] Reference for `--verify`: sequential count on a gathered graph (small inputs only)

## Deferred (no action now)

- kamping-v2 communication: pass a communicator through `CommunicationUtility` (~40 hard-coded `MPI_COMM_WORLD`),
  `kamping::v2::alltoallv`, `dstl::sparse_alltoall`. Needs C++20.
- BriefKAsten instead of the legacy message-queue: needs C++23, flow control off, and a redesign of
  `ConcurrentBufferedMessageQueue`.
- Drop KaMPIng from kamping-spdlog-adapter (plain MPI); afterwards katric needs only kamping-types from KaMPIng v1.

## Order and validation

1. CMake structure and presets (including the git-version swap and the SHA pins) — done
2. Abseil option — done (cluster timing open)
3. Caliper/Adiak measurement and JSON output, replacing tlx timers and cereal — done (cluster overhead open)
4. Next, in this order:
   1. clang-tidy config and a first run on the algorithm headers (4c) — done
   2. KaGen input generated once (4 + first item of 4b) — done
   3. Abstract algorithm, factory, `reset`, `--verify` (4b), then remove graph-io — done
5. kaval switch and removal of the old runner scripts
6. Cluster validation through kaval: Abseil vs. sparsehash run time, Caliper overhead, Linux Adiak collectors
7. CI (build, tests, clang-tidy bug checks) and README; competitors behind `AbstractAlgorithm` as needed

CMake came first because every later step adds options and dependencies. KaGen and the harness come before kaval
because kaval needs `--kagen_option_string`, and with generation done once per run the measured iterations no longer
include input I/O. The open cluster measurements wait for kaval so they run through the new pipeline once instead
of the broken `runners.py`.

After each step: build with the `dev` preset, run the tests, and compare triangle counts on `examples/` plus a few
KaGen graphs with fixed seeds against the previous commit. For the Abseil step, also compare message counts; small
differences are expected because iteration order changes.
