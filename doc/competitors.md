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

1. **One binary, opt-in.** `--algorithm katric|tric|havoqgt|tripoll|tom2d|combblas|lcc`, each competitor behind
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
3. **Competitors:** TriC, HavoqGT, TriPoll/YGM, Tom2D, CombBLAS (masked SpGEMM) and LCC (Strausz et al.). Bader's
   code is out because it is shared memory only (OpenMP, 32-bit ids).
4. **Forks:** a `katric` branch in each fork under niklas-uhl. The branch has a library target with a "construct from
   CSR or edge list + communicator" entry point, no `main`, and no KaGen submodule.
5. **LCC is GPL-3.0** (`fvella/lcc` + `spcl/CLaMPI`). It is linked only with `KATRIC_COMPETITOR_LCC=ON`. Binaries
   built with it are GPL and not distributed, and the README says so.
6. **CombBLAS builds on KaCCv2's adapter and fork** `niklas-uhl/CombBLAS@46ebfc31`, which carries the fixes.
   - Reused from `KaCCv2/benchmark/detail/combblas_algorithm.cpp`: `ImplBase`'s edge list → `ReadFromLocalEdgeList`
     ingest, and the `par::Mpi_datatype<unsigned long long>` specialization.
   - Written ourselves, the Azad/Buluç/Gilbert (IPDPS-W'15) count: L and U from A via `Prune`, the masked `PSpGEMM`
     L·U, `EWiseMult` with A, then `Reduce`.
   - Needs square grids, as KaCCv2's LACC suites do.

## Shared pieces

- [ ] `make_algorithm` switches on a `Config::algorithm` enum. The deprecated `patric`/`cetric-x` spellings map to
      `katric`, so eval's `variant` column keeps working
- [ ] One adapter per competitor in `apps/competitors/<name>.{h,cpp}`, using pimpl so the competitor's headers stay
      in one TU. Each `.cpp` is compiled only when its option is ON, which defines `KATRIC_HAVE_<NAME>`
- [ ] Competitor-specific options in a CLI11 option group per competitor (`--tric-variant`, `--tric-buffer-size`, ...).
      They land in `conf.options`, so Adiak records them as `config.*`
- [ ] `apps/competitors/convert.h` converts the `kagen::Graph` into a global edge list, a CSR with 64-bit global ids,
      or the vertex distribution (`kagen::BuildVertexDistribution`). It uses KaGen's own builders where they fit and
      never assumes sorted neighbourhoods
- [ ] Each adapter's constructor checks its constraints and aborts with a clear message:
  - square p (Tom2D, CombBLAS)
  - equal ranks per node (HavoqGT)
  - the MPI thread level YGM needs. `main` asks for a higher level only when TriPoll is selected
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

LCC and TriPoll carry the build risk (autotools + CLaMPI, YGM's MPI thread level), so a blocker shows up early
enough to decide whether to drop one.

| Step | Competitor | Fork / pin | Adapter notes |
| --- | --- | --- | --- |
| 1 | TriC (Ghosh & Halappanavar, HPEC'20) | `niklas-uhl/tric` (`main`, `6049cb15`) | See below |
| 2 | HavoqGT (Pearce et al.) | `niklas-uhl/havoqgt` (`master`, `fa09fbe3`) | See below. Equal ranks per node (32 of 48 on SuperMUC in the paper) |
| 3 | LCC (Strausz et al., IPDPS'22) | new forks of `fvella/lcc` and `spcl/CLaMPI` | See below |
| 4 | TriPoll / YGM (Steil et al., SC'21) | new fork of `LLNL/tripoll` | See below |
| 5 | Tom2D (Tom & Karypis 2019) | `niklas-uhl/2DTriangleCounting` (`nuhl-integration`, `650d7203`) | See below. Square p only |
| 6 | CombBLAS masked SpGEMM | `niklas-uhl/CombBLAS@46ebfc31` | Decision 6. `preprocessing` builds L and U; `counting` is the SpGEMM plus the reduce |

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

For each competitor:
- [ ] TriC
- [ ] HavoqGT
- [ ] LCC
- [ ] TriPoll / YGM
- [ ] Tom2D
- [ ] CombBLAS

Record for each, once it is done:
- fork SHA
- native input format
- what `preprocessing` and `counting` contain
- constraints
- validation results

## Experiments and eval

- [ ] `experiments/competitors-real-world.suite.yaml` and `competitors-weak-scaling.suite.yaml`. They `import` the
      existing instance sets, so the inputs are the same as KaTric's. The old log2 generator-flag problem is gone,
      because the input comes from `--kagen_option_string`
- [ ] Separate suites for square-p competitors (Tom2D, CombBLAS) with kaval's `--cores sqr` / `sqr-pow2`, as in
      KaCCv2's `paper-combblas`
- [ ] A HavoqGT suite with 32 tasks per node (check that kaval supports `tasks_per_node` per suite)
- [ ] Core ranges go only in the suite header comments (no `max_cores`)
- [ ] `eval/katric_eval.py`: the `variant` column takes the `algorithm` value (plus `tric-variant` and similar).
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
