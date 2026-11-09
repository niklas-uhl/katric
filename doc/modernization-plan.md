# katric modernization plan

Working checklist for the `chore/modernize` branch. Tick items (`[x]`) as they land and add the commit hash.

## Goal

Bring katric's build, measurement, input and experiment setup in line with kascade and KaCCv2. That means
FetchContent for every dependency, CMake 3.25 presets, kamping timers with nlohmann JSON output, KaGen input and
kaval as the experiment runner. Where those projects have weak spots, do it better here instead of copying them
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

A Caliper/Adiak measurement prototype is on `explore/caliper-adiak` (`34eb2b6`). It is not merged.

## Work items

### 1. CMake structure

- [x] Add `CMakePresets.json` (v6 for CMake 3.25; `dev` with warnings and `experiments`, both in `build/`; dev test +
      workflow presets) and a `CMakeUserPresets.json.template` (compiler, local checkouts); gitignore
      `CMakeUserPresets.json`
- [x] `KASSERT_ASSERTION_LEVEL` actually reaches the code (kassert reads a target property, not the variable, so
      katric's assertions were never at level 30 in Debug)
- [ ] KaMPIng: `kamping::types` does not define `KAMPING_ASSERT_ASSERTION_LEVEL` (only `kamping::kamping` does), so
      every TU including kamping-types prints a `#warning`; fix upstream in KaMPIng
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

### 3. Measurement and JSON output

- [ ] Replace `tlx::MultiTimer` with `kamping::measurements::timer` (plus counters where useful)
- [ ] Replace cereal with nlohmann_json v3.12.0 + kamping-nlohmann-json-adaptor v1.0.0 for config, statistics and
      `--json-output` (~73 cereal/MultiTimer uses, mainly `statistics.h`, `config.h`, `message_statistics.h`,
      `apps/katric.cpp`, `apps/shmetric.cpp`)
- [ ] Fix the `LocalStatistics` copy constructor and assignment, which drop `intersection_size_local/global` and `wedges`
- [ ] Make the JSON layout match what kaval and the kascade/KaCCv2 eval scripts expect

### 4. Graph input: KaGen

- [ ] Add `--kagen_option_string` and build the distributed graph from KaGen's output (generators and file input)
- [ ] Check whether KaGen reads the binary `first_out`/`head` format (`examples/hiv.*`); if not, convert the
      examples to METIS or KaGen's binary format
- [ ] Check KaGen support for pre-partitioned input (`examples/partitions`, `partitioned: True` in suites)
- [ ] Remove graph-io (including its `MPI_NODE` macro with the typo'd `#ifdef MPI_UNINT64_T`)

### 5. Experiment runner: kaval

- [ ] Fetch kaval into `./kaval` like kascade/KaCCv2, pinned to a SHA and gitignored; add `justfile` and
      `pyproject.toml` / uv setup
- [ ] Convert `experiments/paper-suites/*.suite.yaml` and `all-suites`; check `threads_per_rank` and the graph specs
      against kaval's format
- [ ] Remove `expcore.py`, `runners.py`, `run-experiments.py`, `simple_runner.py`, `slugify.py`, the templates,
      `*.env` and `cetric-from-config.py` (keep `configs/*.toml`: they are CLI11 algorithm presets for `--config`)

### 6. Cleanup

- [ ] Decide whether `shmetric` stays; if so, port it to the new input and JSON code. `apps/shmetric-benchmark.cpp`
      is not built by any target (dead file)
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

- [ ] Measurement: kamping timers (matches kascade/KaCCv2) or the Caliper prototype on `explore/caliper-adiak`?
      If kamping wins, keep that branch only for reference.
- [ ] Keep or drop `shmetric`?
- [ ] Abseil: opt-in until the counts match (current plan), or the default right away?

## Deferred (no action now)

- kamping-v2 communication: pass a communicator through `CommunicationUtility` (~40 hard-coded `MPI_COMM_WORLD`),
  `kamping::v2::alltoallv`, `dstl::sparse_alltoall`. Needs C++20.
- BriefKAsten instead of the legacy message-queue: needs C++23, flow control off, and a redesign of
  `ConcurrentBufferedMessageQueue`.
- Drop KaMPIng from kamping-spdlog-adapter (plain MPI); afterwards katric needs only kamping-types from KaMPIng v1.

## Order and validation

1. CMake structure and presets (including the git-version swap and the SHA pins)
2. Abseil option
3. kamping timers + nlohmann JSON, replacing cereal
4. KaGen option string, then remove graph-io
5. kaval switch and removal of the old runner scripts
6. CI and README

CMake comes first because every later step adds options and dependencies. kaval comes last because it needs both
the new JSON output and `--kagen_option_string`.

After each step: build with the `dev` preset, run the tests, and compare triangle counts on `examples/` plus a few
KaGen graphs with fixed seeds against the previous commit. For the Abseil step, also compare message counts; small
differences are expected because iteration order changes.
