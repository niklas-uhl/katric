# katric refactoring plan

Working checklist for restructuring the algorithm code before the algorithmic changes for the dissertation version
(compute at the tail + rebalancing, message trimming, compact-forward ordering). Tick items (`[x]`) as they land and
add the commit hash. The build/tooling modernization is done and tracked in [modernization-plan.md](modernization-plan.md).

## Why now

The first cluster runs of the `phases` suite (`1962d4e`; HoreKa2 at 192/384 cores, SuperMUC-NG at 768 cores; weak
scaling, 2^22 edges per core) show where the time goes, and the main costs are in code the algorithm ideas would touch
anyway:

- With locality (RGG/RHG, 0.2–0.7% cut edges) an iteration takes under a second and stays flat from 192 to 384
  cores; the local phase is about 60% of it, sorting 12–14%.
- Without locality (permuted RGG, 99.5% cut edges) it is 8–13× slower, and preprocessing costs as much as the global
  phase: the degree exchange alone takes 2.6–5.1 s, orientation 0.9–1.4 s (the degree comparator looks up every
  ghost's degree in `AuxiliaryNodeData`'s `absl::flat_hash_map`). The global phase is intersection work, not
  waiting: `termination` is 5–10% (the 30% of a laptop run came from oversubscription).
- cetric-x's ghost expansion is pure overhead without locality: +50–60% over patric.
- On RHG the imbalance grows with p (barrier share 17% at 192 cores, 37% at 768), from the local phase and from
  preprocessing around the hubs.
- Sorting neighborhoods by degree (`--id-node-ordering false`) is 1.6–2.5× slower than `id_outward` and very
  imbalanced on RHG.
- `cetric_edge_iterator.h` has 1960 lines with about six near-copies of the intersection loop
  (sequential/parallel × with/without ghosts × edge-partitioned). The counter fix in `7b483cd` had to be applied
  three times.

## Rules

- **Validation.** Every step keeps the triangle counts (ctest `--verify` and `local/smoke`) and the non-timing
  counters identical on fixed inputs and seeds. The grid policy's message counts are nondeterministic; ignore them.
  When a step changes a counter on purpose (e.g. a smaller degree-exchange format changes `send_volume`), the commit
  says so.
- **Baseline first.** Run `phases` on SuperMUC-NG (48 / 1536 / 3072 cores) at the commit before step 1 and tag it
  `pre-refactor`, so every later step can be compared through the same suite and `eval/explore.py`.
- **What stays.** The paper configurations (`configs/ditric.toml`, `configs/cetric.toml`, the hybrid suite) must
  keep running and produce the same counts. Everything else is up for removal.
- No algorithmic change in this plan. The ideas come afterwards, on the cleaned-up code.

## Steps

Step 1 validation (2026-11-12): 70 reference runs (RGG/RHG, each also permuted, com-amazon x DiTriC/CETRIC/pre-CETRIC
with direct and grid communication and DiTriC with DPD, on 2 and 4 ranks) give identical triangle counts and counters
after every step-1 commit (grid message counters excluded). The counters are deterministic between two runs of the same
binary.
Scripts: `scripts/counters_run.sh build-release/apps/katric <dir>` (about 20 s on the laptop), then
`scripts/counters_compare.py <ref dir> <new dir>`; take the reference from the commit before a step.

### TU split (before step 1)

`apps/katric.cpp` is one translation unit with every algorithm × communication policy × node ordering × local-phase
variant (38 s Debug, 49 s Release on the laptop for any edit). Split as in KaCCv2 (`src/kacc/components`: one `.cpp`
per algorithm in a static library, a header with only non-template overloads, the templates in headers included only
by those `.cpp` files):

- [x] Static library `cetric-algorithms` in `src/`: `run_patric` / `run_cetric` / `run_cetric_x` as non-template
      overloads per communication policy (`cetric/algorithms.h`), one `.cpp` per algorithm × policy. The driver is
      split as well: `apps/parse_config.cpp` (CLI11), `apps/algorithm.cpp` (KatricAlgorithm, factory),
      `apps/verify.cpp`; `katric.cpp` no longer sees the graph or the edge iterator. Laptop (10 cores, g++-16):
      editing `katric.cpp` 18 s -> 3 s (Debug); editing one algorithm 12 s (Release); everything 35 s (Release,
      `-j10`; the `experiments` preset uses Makefiles, so build with `-j`). Three header functions were not `inline`
      (ODR, only visible with several TUs)
- [ ] Later steps: per-phase TUs once step 2 has phase functions. `-g1` (KaCCv2 uses it for multistep) saves only
      ~20% on an algorithm TU (10.2 s -> 8.3 s), not adopted

### 0. Baseline

- [ ] `phases` on SuperMUC-NG and HoreKa2 at `1962d4e` (configurations 0–2: patric, cetric-x, cetric-x with
      `--id-node-ordering false`); tag `pre-refactor` on `1962d4e` (the algorithm code is unchanged since).
      Status 2026-11-13: SuperMUC-NG 48–3072 complete except permuted RGG at 192 (cetric-x and degree ordering killed,
      out of memory) and RHG/permuted RHG at 3072 with degree ordering (300 s `I_MPI_JOB_TIMEOUT`). HoreKa2 only 192
      for the permuted inputs and 192–768 for the others: every permuted run at 384/768 dies in its first iteration.
      Root error (in the -log.txt, not the -err.txt): UCX `ibv_reg_mr(..., length=2480..7488) failed: Cannot allocate
      memory` on mlx5_0, ~100 times per run, i.e. memory registration for many concurrent small zero-copy sends runs
      out; the `MPI_ERR_INTERN` in a blocking `MPI_Recv`, the IB remote access errors and the CMA `process_vm_readv: No
      such process` are follow-ups of ranks aborting. The `MPI_Recv` is the message queue's (`queue.h:112`) in the global
      phase: `addr2line` on the HoreKa2 binary resolves the caller to `CetricEdgeIterator<..., MessageQueuePolicy>::
      run_distributed_sequential`, called from `run_patric<MessageQueuePolicy>`.
      Expected to go away with BriefKAsten, so HoreKa2 waits until then (see open decisions); a cheap test before
      that: `UCX_RNDV_THRESH=inf` (small messages eager, no registration). 1536/3072 not run.
      Not run yet: configurations 3+ (`cetric`, grid).
- [ ] Rerun the whole `phases` suite (patric, cetric-x, cetric, each direct and grid) on SuperMUC-NG from `3466ccf`
      (decided 2026-11-13), 48–1536 cores: KaMPIng assertions off (the runs above had level 10), triangle count and
      build metadata fixed. Algorithm code as `1962d4e`. This is the baseline for the refactoring steps; tag it
      instead of `pre-refactor` on `1962d4e`. 3072 left out (RHG's 19M-degree hub, see below)
- Findings from the 2026-11-13 data (time per iteration, max over ranks, SuperMUC-NG unless noted):
  - Locality inputs weak-scale flat from 48 to 1536 cores (RGG 2.2–2.6 s, RHG 1.8–2.7 s); the local phase is 60–70%.
    HoreKa2 is 2.5× faster per core on the same runs (local phase 0.62 vs 1.6 s on RGG). The SuperMUC build is fine
    (checked 2026-11-13: Release, -O3 -march=native = skylake-avx512, KASSERT off, gcc 15.1), so this is the
    hardware (Skylake-SP from 2018 vs HoreKa2's newer CPU); absolute times are not comparable across the clusters.
    The 192-core job is 25% slower than 96 and 384 in every configuration (one job, probably a slow node).
  - RHG at 3072 cores breaks: one vertex of degree 19M, its rank has 27M local edges (3.2× the mean). patric 2.7 s ->
    24 s, the degree exchange alone 10.7 s, preprocessing barrier 8.4 s; permuted RHG 18 -> 36 s. An input effect
    (KaGen RHG, γ = 2.8, 2^30 vertices), not the algorithm: keep it out of weak-scaling series (or cap the degree) and
    use it as the hub test case for idea 1.
  - Permuted inputs do not weak-scale: patric 10.7 s (48) -> 21 s (3072), global phase 5.4 -> 11.7 s, degree exchange
    and orientation grow with p. cetric-x is +35–60% over patric (ghost expansion in preprocessing, a 3 s local phase
    that finds almost nothing).
  - With locality cetric-x is within ±5% of patric on RGG and 15–25% slower on RHG from 384 on (preprocessing 1.0–1.4
    vs 0.5–0.6 s). Degree ordering: 1.6–2.6× slower everywhere (already removed in step 1).
  - The `triangles` Adiak value wrapped at 2^31 (Caliper imports `adiak_long` as `int`); the counts in these profiles
    are wrong above 2^31, the logs have the right ones. Fixed by recording `long long` (same commit as this note)
- Findings from the rerun from `3466ccf` (SuperMUC-NG, `phases_26_11_13`, 48 and 1536 cores so far, all six
  configurations; time per iteration, max over ranks, mean of 3 iterations; script in the session scratchpad):
  - KaMPIng assertions (level 10 before) cost nothing measurable: patric / cetric-x within ±2% of `phases_26_11_12`
    on the permuted inputs, ±12% on the locality inputs at 3 iterations (noise of the max over ranks)
  - Without locality, preprocessing is as large as the global phase: patric 18.2 s at 1536 (10.6 at 48), of which
    degree exchange 5.7 s (3.0), orientation 2.0 s (1.5), global phase 9.4 s (5.3). cetric-x adds ghost expansion
    3.5 s and a 2.9 s local phase: 24.8 s. pre-CETRIC's degree exchange on the contracted graph costs the same as
    patric's (5.8 s at 1536: without locality nothing is contracted away): 19.7 s
  - Grid indirection in the global phase (legacy message queue; BriefKAsten's messaging differs enough that this
    may change completely) does not pay without locality: +5–20% over direct, `termination` 1.6 ->
    5–6 s at 1536. With locality it is neutral to slightly better (patric RGG 2.13 vs 2.58 s at 1536, but the
    local phase differs as much, so partly node noise)
  - With locality, the barrier share grows from ~5% (48) to 25–29% (1536) on RGG for every configuration; on RHG
    pre-CETRIC's local phase is 4.6 s at 1536 (cetric-x 1.8, patric 1.5) with a 67% barrier share: outward
    orientation loses the out-degree bound at the hubs, as expected. cetric vs cetric-x: cetric is better on RGG
    (2.56 vs 2.93 s) and without locality (19.7 vs 24.8 s), much worse on RHG (5.6 vs 3.4 s)
  - The dense degree exchange (alltoallv) is the single largest preprocessing item without locality; measure
    `--degree-exchange sparse|grid` (step 3) against it in the next run
- [x] `cetric` (pre-CETRIC) appended to `phases` as a fourth configuration, so a rerun only adds its runs (the
      indices of the running configurations stay the same)
- [x] `communication-policy: grid` appended to `phases` (patric, cetric-x, cetric; decided 2026-11-12). Run the
      remaining baseline from this commit: the algorithm code is unchanged since `1962d4e`, step 1 changes the
      suite (`id-node-ordering` goes away)

- [ ] Targeted suites on SuperMUC-NG (48 / 192 / 768 cores; added 2026-11-18, run while the refactoring continues;
      SuperMUC-NG shuts down at the end of 2026): `lb-cost` (idea 9: DiTriC on RHG, primary cost function none / DPD /
      IDPD), `no-sync` (upper bound for idea 5: DiTriC and CETRIC on RGG / RHG, `global-synchronization` on / off),
      `degree-exchange` (step 3's transports dense / sparse / grid, DiTriC on permuted RGG and RHG). Every
      configuration verified on the laptop (4 ranks, `--verify`). First results (`*_26_11_18`, 48-768 cores):
      - `degree-exchange`: dense / sparse / grid within 5-10% of each other on permuted RGG (2.7-5.3 s) and RHG
      - `no-sync`: without barriers patric is unchanged, cetric-x on RGG 7-11% faster (48/192 cores; 768 not run)
      - `lb-cost`: DPD / IDPD make RHG ~2x slower (48/192): the repartitioning costs 0.7-0.8 s and the global phase
        grows from 0.1-0.15 to 0.6-1.05 s, the local phase gains only 5-10%. At 768 every DPD/IDPD run died: the
        global edge array (> 2^31 entries) overflowed `dstl::redistribute`'s int displacements (fixed in PR #126)
      Extended (2026-11-18): `degree-exchange` and `no-sync` by RMAT and GNM (the paper's 2^16 / 2^20 per core) and
      1536 / 3072 cores (RHG at 3072 has the 19M-degree hub)

### 1. Prune

Options that are dead or only used in old exploration suites (`experiments/legacy`), never in the paper suites.

- [x] Remove options read nowhere (and `-v`): `--orient-locally`, `--omp-schedule`, `--omp-chunksize`, and the config fields
      `synchronized`, `buffer_threshold`, `max_degree_threshold_alpha`, `empty_pending_buffers_on_overflow`
- [x] Remove the barely used ones (decided 2026-11-12, OpenMP variants including `CETRIC_USE_OPENMP` and the
      OpenMP dependency): `--degree-filtering`, `--dense-load-balancing`,
      `--edge-partitioning-static`, `--high-degree-threshold-scale`, `--parallelization-method` (OpenMP variants),
      `--tbb-partitioner`
- [x] `--flag-intersection`: dead today. `pre_intersection` / `post_intersection` / `intersect` are never called,
      `is_v_neighbor_` is never sized and would be indexed by global ID. Remove it; it comes back on compact IDs
      (idea 4, after step 4)
- Keep `--algorithm cetric` (pre-CETRIC, `run_cetric`). Not in the paper suites, but it orients cut edges outward
  in the local phase: no degree exchange before it (only on the contracted graph) and no ghost expansion, while
  triangles with two local vertices are still found locally and the contraction stays correct. The price: the
  √(2m) bound on out-degrees does not hold for vertices with many cut edges. Old real-world data: same
  local/global split as cetric-x, as fast or faster single-threaded. Decide between `cetric` and `cetric-x` from
  `phases` (step 0)
- [x] Remove (decided 2026-11-12): `--skip-local-neighborhood`, `--compact-degree-exchange` (step 3 replaces the
      degree exchange anyway)
- Keep for now: `--intersection-method binary|hybrid` (galloping for very different sizes, as at the hubs on RHG;
  measure before removing, idea 8) and `--pseudo2core` (real-world graphs have many degree-1 vertices, idea 10)
- [x] Remove (decided 2026-11-12): `--id-node-ordering false` (pure degree ordering): 1.6–2.5× slower on the
      clusters, sorting up to 13×; compact-forward ordering (idea 3) gets the skip with a cheaper key
- [x] Keep only N, DPD, IDPD (decided 2026-11-12) of the cost functions N/D/DH/DDH/DH2/DPD/IDPD/D* and primary load balancing.
      Unused in the paper, but DPD/IDPD are the starting point of idea 1
- [x] Remove commented-out code blocks in the touched files (`preprocessing`, `cetric_edge_iterator.h`)

### 2. One kernel per phase

- [x] One intersection kernel for the local phase: `intersect_local_edge` (one edge v -> u, ghost adjacencies and
      the local-prefix clipping included) is the only intersection call of the local phase, and
      `run_local_sequential<with_ghosts>` replaces the two sequential copies. The four hybrid variants stay (frozen)
      but call `intersect_local_edge`. The `NodeOrdering == EdgeOrientationOrdering` offset was dead since
      `--id-node-ordering false` went, so `run_local` no longer takes the orientation ordering. Identical counters on
      the 70 reference runs, ctest (with hybrid) passes (`70aeb31`)
- [x] Counters (`wedge_checks`, `intersection_size_*`, triangles) collected in one place per kernel:
      `LocalPhaseCounters` and `GlobalPhaseCounters` (plain counts on the sequential path, one instance per thread
      on the hybrid path instead of the iterator's three atomics). The global phase has one message handler
      (`process_message` -> `intersect_with_message`) for both paths; the receiver-parallel loop
      (`--global-degree-of-parallelism` > 2, set by no config) only runs on the hybrid path now. The hybrid path also
      reports `interface_nodes`. Identical counters on the 70 reference runs (`5f5a54d`)
- [x] `run_patric` / `run_cetric` / `run_cetric_x` (was `run_cetric_new`) call their phases directly: each phase is a
      function (one Caliper region, ending in its barrier) with ordinary parameters and return values (the threshold,
      the contracted graph, whether it was rebalanced; `TriangleCounter` collects the count). A phase is swapped by
      calling another function (e.g. the tail-compute global phase of idea 1). Headers: `cetric/preprocessing.h`
      (ghost ranks, primary load balancing, the three orientations), `cetric/contraction.h` (contraction, secondary
      load balancing, the contracted graph's orientation), `cetric/execution.h`, `cetric/triangle_counter.h`;
      `counters/cetric.h` keeps the local and global phases and the drivers. A first version ran a list of function
      pointers over a shared `PhaseState` (`2a6348d`); replaced as overengineered (decided 2026-11-13): the ghost
      degrees and ghost set never left preprocessing. Region changes against the baseline: patric's ghost-rank pass is
      the region `ghost_ranks` and its `reduction` is `reduce`; cetric-x has no empty `preprocessing_global` /
      `secondary_load_balancing` regions; the ghost degrees are freed at the end of preprocessing (were: end of the
      run, outside any region, or in `global_phase`). `--secondary-cost-function` (pre-CETRIC only, set by no
      config) kept, a starting point for idea 1. Identical counters on the 78 reference runs (`df09647`)
- [x] `with_execution_policy(conf, [parallel,] f)` and `for_each_local_node(G, conf, parallel, f)` replace the
      `if (num_threads > 1) ... else ...` / `tbb::task_arena` duplication in the drivers (`2a6348d`). The secondary
      load balancing's edge removal had a `parallel_for` without an arena; it now runs in one

### 3. Communication layer

The deferred kamping-v2 plan from modernization-plan.md, plus the degree exchange.

- [x] Pass a communicator instead of the hard-coded `MPI_COMM_WORLD`: `katric.cpp` passes `MPI_COMM_WORLD` (not
      duplicated, decided 2026-11-13: nothing can collide) to input, verification and the algorithm, `DistributedGraph` keeps
      it (`G.comm()`, constructor takes it instead of rank and size), `DegreeCommunicator` takes it from the graph,
      `CommunicationUtility`'s exchanges, `GraphCommunicator::relocate` and `LoadBalancer::run` take it as a
      parameter, the phases' barriers and reduction use `G.comm()`. Left on `MPI_COMM_WORLD`: the legacy message
      queue (hard-coded in the library) and the grid adaptor laid over its ranks, until BriefKAsten; rank lookups for
      log messages. The unused `MPITimer` is gone. Identical counters on the 78 reference runs (`b11a9d1`)
- [x] Degree exchange on `cetric::push_to_ghosts` (`cetric/ghost_exchange.h`), adapted from KaCCv2's
      `kacc::push_to_ghosts` (copied, decided 2026-11-13; a candidate for a dSTL function later). Transport
      `--degree-exchange sparse|dense|grid` (replaces `--dense-degree-exchange`; configs and suites say `dense`):
      `dstl::sparse_alltoall` (NBX), `kamping::v2::alltoallv` after an alltoall of the counts, `dstl::grid_alltoallv`.
      Used by the orientation's degree exchange and the cost functions' ghost data. Caliper sub-regions `prepare` and
      `exchange`. The grid mode reports no `received_messages` (its unordered receive does not tell sources apart). Its
      `grid_comm` (a communicator split) is built once per communicator in a `cetric::grid_comm_cache` (a
      `kamping::v2::cache`, as dSTL's `rsx_window_cache`), owned by `main` and reached through `Config::grid_comms`;
      without a cache it is built per call (region `grid_comm`).
      CommunicationUtility's three exchanges that only the old degree exchange used are gone. Identical counters on the
      78 reference runs (dense, as in the configs); the orientation test checks that all three transports deliver the
      same ghost degrees; ctests for DiTriC with `dense` and CETRIC with `grid` (`aec0ab1`)
- [x] Deduplication: per-rank last-source stamp (from KaCCv2) instead of the per-vertex `absl::flat_hash_set<PEID>`.
      Format: still two `RankEncodedNodeId`s per entry. Packing (vertex, degree) into one word ruled out (decided
      2026-11-13): KaCCv2 found it does not help much, the exchange is limited by irregularity and latency, not volume
- [x] Buffers: `(message, destination)` pairs appended in one pass over the adjacency, then a counting sort into one
      flat buffer with counts and displacements (option 2 of three, decided 2026-11-13). The alternative without the
      copy (count, then fill: two passes over the adjacency) is worth measuring only if `prepare` matters in `phases`

- [ ] Primary load balancing moves the vertices with `dstl::redistribute` (decided 2026-11-18): the cost split is
      monotone in global-ID order, so the node and edge arrays are one redistribute each, with the receive counts
      from one `MPI_Reduce_scatter_block` (replaces `GraphCommunicator::relocate` and the hand-written NBX
      `sparse_all_to_all` with its size messages). Identical triangle counts on the 70 reference runs and 54 extra
      DPD / IDPD / N runs on 3 / 5 / 8 ranks; only the `redistribution` region's message counters change (no size
      messages, `received_messages` not reported). Needs the dstl fix for empty windows inside a range (a rank
      receiving nothing between two that do, e.g. after a hub whose cost exceeds C/p: stray zero-length message;
      a rank sending nothing: hang), not merged in kamping-v2 yet. The DPD hang on graphs with fewer vertices than
      ranks is elsewhere (unchanged)

KaCCv2's communication primitives (read 2026-11-13, branch `feat/kout-symmetric`; katric already fetches kamping-v2
with dstl but links neither):

- `kacc::push_to_ghosts` (`src/kacc/graph/ghost_exchange.hpp`) is the degree exchange: for every source vertex with
  a remote neighbor it sends `vertex_to_message(u)` once per owner rank (dedup by a per-rank last-source stamp, the
  "last-rank marker" above), through `kamping::v2::alltoallv` or `dstl::grid_alltoallv` chosen by the type of the
  communicator argument (`comm_view` vs `dstl::grid_comm`), and returns the received messages. Covers the first three
  items above in one call. Needs a graph modelling KaCCv2's `graph::DistributedGraph` concept (`vertices()`,
  `neighbors(v)`, `to_local`, `neighbor_is_local` / `neighbor_owner`). Differences from what we want: per-rank buffers
  are still an `absl::flat_hash_map<int, vector>` (item 4), it times with `kamping::measurements::timer()` instead
  of Caliper, and the grid communicator should be built once per run (a collective split), not per exchange.
  Also fits the cost functions' ghost exchange (`cost_function.h`, `get_ghost_data`)
- `PackedWireEncoding` (`wire_encoding.hpp`): vertex and payload in one word. (global id, degree) fits 64 bits at
  the `phases` sizes (2^36 ids at 3072 cores, the 19M-degree hub needs 25 bits): 8 instead of 16 bytes per entry
- `build_ghost_reverse_index` (`components/ghost_reverse_index.hpp`): CSR from each ghost to the local vertices
  adjacent to it, built in two passes. Idea 12 needs exactly this (a ghost's local neighbors on r); on compact IDs
  (step 4) the hash-map slot becomes an array index
- `measure_locality` (`graph/locality.hpp`) and `resolve_auto_reachability_mode` (`components/auto_comm_mode.hpp`):
  PE fan-out, internal-edge ratio, average degree in one pass, and a collective vote (thresholded per rank, summed
  with one allreduce) that picks a communication mode identically on every rank. Candidate mechanism for "without
  locality CETRIC should not pay for ghost expansion" and for choosing direct vs grid automatically
- `dedup_alltoallv` / `scatter_reduce_{flat,grid}`: route values (or key/value pairs with a fold) to their owners,
  deduplicating / folding between grid hops. The first half of idea 6 (each rank tells owner(w) that it holds an
  in-neighbor of w) has this shape
- dstl itself: `sparse_alltoall` (NBX/PCX/RSX), `request_reply`, `redistribute` (idea 9), `distribution` (step 4)

### 4. Order-preserving compact IDs instead of rank encoding

Today every adjacency entry is a `RankEncodedNodeId`: the global ID in the low 48 bits, the owner rank in the top
16. The rank is a pure function of the ID, because the vertex ranges are contiguous (KaGen's distribution, and
the primary load balancing splits a cost prefix sum into contiguous ranges). So the encoding only caches
`get_rank(v)`, and its uses split into:

- local tests (`u.rank() == rank_`, `id_outward`, `local_prefix_end`): a range test on the ID is as cheap
- order: sorting by (rank, id) is the same as sorting by id
- routing in the global phase (owner of each cut out-edge): the only real use; without the cache it is a binary
  search over the partition table (about 12 steps at 3k ranks)

Its costs: at most 65,536 ranks and 2^48 vertices, the `find_ghost_ranks` pass (repeated after load balancing),
`id()` masking and two notions of equality throughout. Ghost data (degrees for the orientation comparator) sits
in `AuxiliaryNodeData`'s `absl::flat_hash_map`, which is the orientation cost the laptop run showed.

Replacement: number the sorted union of local vertices and ghosts 0..n+g-1. Local vertices are one contiguous
block (local index = offset + `v - first`, still arithmetic); per-ghost arrays hold global ID, rank and degree.

Order of the compact IDs (decided 2026-11-13): **local vertices first, then ghosts, each sorted by global ID** (option B
of three). It is today's `id_outward` made numeric: local test `c < n`, local index `c`, ghost-array index `c - n`; the
local phase stops at the first ghost (DiTriC too, which today skips interleaved ghosts) and CETRIC's clipping stays one
partition point. Global phase: messages carry global IDs in global order; the receiver finds the entries local to it
(its contiguous ID range) with one binary search per message and intersects locals with locals and ghosts with ghosts;
the sender emits `[ghosts below its range][locals][ghosts above]`, linear, without sorting (after CETRIC's contraction
N+(v) holds only ghosts and is already in global order). No counter changes (same edges, same neighborhood sizes).
Rejected for now:
- global-ID order (local block at an offset; simplest global phase, but CETRIC's local part becomes a middle run)
- orientation order (degree, id): the compact-forward skip (idea 3) and integer comparisons for trimming (idea 2),
  but the global phase could not merge a global-order message against N+(u) without a second sorted copy or keys on
  the wire. Flag intersection (idea 4) makes the local phase order-free, and trimming needs only (degree, id) of the
  sender's own vertices, two array lookups with per-ghost degrees. A later renumbering, if needed
Compact IDs fit 32 bits per rank (n + g < 2^32), which would halve the adjacency; measure separately.

- [x] Partition table: `cetric::graph::VertexPartition` (`datastructures/distributed/vertex_partition.h`), copied
      from KaCCv2 (decided 2026-11-13: copy what we need) without the split-vertex parts, and with empty ranks placed at
      their predecessor's end (katric gives them the range (max, max)). Built collectively in the graph's
      constructor; `find_ghost_ranks` uses its `get_rank` (local fast path, else binary search). `gather_PE_ranges`,
      `get_PE_from_node_ranges` and `--binary-rank-search` are gone (configs and suites updated). Identical counters on
      the 78 reference runs (`dc74612`)
- [ ] Local indexers: KaCCv2's `DenseIndexer` / `SparseIndexer` on plain `uint64` global IDs, with the compact IDs below
- [x] Stage 1: the edge iterator, the ghost exchange and the cost functions ask the graph (`is_owned(v)`, `owner(v)`,
      `global_id(v)`) instead of reading `RankEncodedNodeId`'s rank and ID, so the representation swap below stays
      inside the graph (`e8adcf8`). Also removed `CostFunction::rank()`, which called a graph method that does not
      exist (never instantiated). Noticed: the parallel cost evaluation increments a shared `node_index` from all
      threads (a race; hybrid path, frozen)
- [x] Before stage 2 (decided 2026-11-13): `cetric_edge_iterator.h` (948 lines) split into `counters/local_phase.h`
      (`local_phase::intersect_edge` + `local_phase::run<with_ghosts>`, free functions over the graph, returning the
      interface vertices) and `counters/global_phase.h` (`global_phase::run`: sending N+(v), the message handler, queue
      setup; `last_proc` is local to the call), about 370 lines together, no class holding state across phases. The
      frozen hybrid code is gone (four parallel local variants, the parallel global phase with its task pools, the
      async send path, `concurrent_buffered_queue.h`, `task_pool.h`, `thread_pool.h`); it stays in git history and is
      re-ported later. `--num-threads > 1` now exits with an error; the hybrid options stay accepted so the paper
      configs parse, `--grainsize` is gone. Hybrid ctest and the 2-thread smoke runs disabled, so is
      `paper-hybrid-real-world` (run it from the paper commit). `intersection()` moved below the kernels it calls (it
      found them only through ADL). Identical counters and triangle counts on the 70 reference runs and the 54 extra
      load-balancing runs
- [x] Dropped `--secondary-cost-function` and the "rebalanced" branch of the contracted global phase (decided
      2026-11-18): it never rebalanced (the cost was computed, then the contracted graph rebuilt in place, so the
      local pass after it found nothing). Idea 1 gets a real rebalancing step on the new graph. Identical counters
- Stage 2 carried out as a new pipeline next to the old graph (decided 2026-11-18): the new graph types and phase
  code first, then the algorithms move over one at a time (DiTriC, CETRIC, pre-CETRIC), each checked by identical
  counters; the old graph code goes at the end
- [x] Stage 2, DiTriC on the new pipeline (2026-11-18): `graph/graph.h` (`VertexLayer`: range, ghosts' global ids and
      owners sorted by global id, partition table; `CsrGraph`: rows over handles, optional ghost rows and rows in
      global ids), `graph/build.h` (input graph, `exchange_ghost_values` on push_to_ghosts, `orient`), `phases/`
      (preprocessing with the cost functions, local and global kernels, drivers). Identical counters on the 124
      reference runs. Notes:
      - the input graph is built inside the run (region `input_graph`, replacing `ghost_ranks`): one sort of the cut
        edges by (head, position) gives the ghosts and the handles. The old graph was built in the untimed `reset`
      - the global phase works on a copy of each out-row in global ids (`global_rows` in preprocessing): merging a
        message against handles needed a ghost-id lookup per step (1.65x slower message handling on permuted RGG).
        Oriented graph = m/2 handles + m/2 global ids, the old graph held m entries
      - laptop, 8 ranks, 2^18 vertices / 2^22 edges: permuted RGG 0.49 -> 0.41 s per iteration (preprocessing 0.16 ->
        0.06 s, global phase 0.29 -> 0.25 s, building 0.085 s), RHG 0.094 -> 0.082 s
      - the DPD hang on graphs with fewer vertices than ranks is gone for DiTriC (the old cost function exchanged the
        degrees a second time on ranks without ghosts, a collective only those ranks entered)
- [ ] Stage 2, the swap, as a pipeline of immutable graphs built from KaCCv2's building blocks (decided 2026-11-13;
      adopt KaCCv2's interface and layering, not `SplitDistributedCSRGraph` or `FilteredGraph`):
      - input graph: what KaGen and the load balancer produce (`LocalGraphView`, global IDs), unchanged
      - oriented graph: `VertexPartition` + `DenseIndexer` for the local vertices; a ghost table (sorted ghost global
        IDs + owner per ghost, from `get_rank` once per ghost; global ID -> slot by binary search, no hash map: only
        needed while building and when depositing exchanged data); an out-CSR over compact IDs (katric's own builder:
        each undirected edge kept at its smaller endpoint in (degree, global ID)); per-ghost degrees from
        `push_to_ghosts` deposited into a `std::vector` indexed by `c - n`; for CETRIC the ghost rows (each ghost's
        out-edges into local vertices) from KaCCv2's `build_ghost_reverse_index`, adjusted to index by `c - n`
        instead of its hash slot map and to take an edge filter (the ghost -> local orientation); its targets come out
        sorted by local index, the order the clipping needs. No in-edges kept (only the ghost rows needed them)
      - contracted graph: `SparseIndexer` over the vertices with cut edges (inserted in global-ID order) + their out-CSR,
        sharing the oriented graph's ghost table and compact IDs
      - the compact ID is KaCCv2's neighbor handle: overload `neighbor_is_local` (`c < n`), `neighbor_owner`,
        `neighbor_local_index` for it, and the stage-1 accessors become these; plain CSR with `std::span` neighbors
        replaces `EdgeRange` / `RankEncodedEdge` / `EdgeLocator`. KaCCv2's generic code (`push_to_ghosts`,
        `measure_locality`, `build_ghost_reverse_index` for idea 12) then copies almost unchanged
      - adjustments to KaCCv2's blocks: the ghost table takes `SparseIndexer`'s storage idea without its
        `VertexPartition` base (whose `is_local` means "in my range", the opposite of a ghost); the reverse index as
        above. Katric's own: the oriented CSR build and the message split in the global phase
      Removes `RankEncodedNodeId`, `find_ghost_ranks`, `AuxiliaryNodeData`, the mutating `DistributedGraph`
      (`first_out_offset_`, `remove_internal_edges`, `remove_in_edges_and_expand_ghosts`) and the old node indexers.
      Validated by identical counters
- [ ] Stage 3: 32-bit compact IDs (n + g < 2^32 per rank) — measure the local phase
- Known issue (pre-existing, also in the binary from before step 1): DPD load balancing on a graph with fewer vertices
  than ranks (`examples/triangle.metis` on 4 ranks) hangs
- [ ] Compact IDs in the adjacency, built once after `find_ghosts`; per-ghost arrays for global ID, rank and degree.
      In KaCCv2's concepts this is a neighbor handle (`neighbor_t` != `vertex_id_t`, projected by
      `neighbor_target`), so the indexers never see compact IDs
- [ ] Routing, local tests, `id_outward` and `local_prefix_end` on compact IDs (array lookup / range test);
      `RankEncodedNodeId`, `find_ghost_ranks` (with `gather_PE_ranges` / `get_PE_from_node_ranges`),
      `--binary-rank-search` and `AuxiliaryNodeData` removed
- [ ] Messages carry global IDs. The numbering preserves the global order, so a receiver merges the received list
      against `N+(u)` mapped through `to_global` (monotone) without translating the received IDs
- [ ] Contraction (`compact()`, today a `SparseNodeIndexer`) and the cetric-x ghost CSR on the new IDs
- [x] Remove the `std::set<RankEncodedNodeId> ghosts` filter in `run_distributed`: every caller passed an empty set
      (removed in step 2, `5f5a54d`)
- [ ] Compare against `pre-refactor` in `phases`: preprocessing (orientation, degree exchange) and `send_ns`
- [ ] A communication context next to the graph's communicator (`G.comm()`) for per-communicator state, e.g. the
      degree exchange's `grid_comm_cache`, which today reaches the callers through a non-owning pointer in `Config`
      (`Config::grid_comms`; a `Config` without it silently splits the communicator on every grid exchange). Decided
      2026-11-13 to keep the caller-owned cache over a static keyval and move it here when the graph's ID and
      partition layer is reworked

KaCCv2's graph class itself (`SplitDistributedCSRGraph`, CSR plus split-vertex replicas) is not adopted: katric
needs the out/in split per vertex after orientation, the ghost CSR and contraction, so it keeps its own oriented
graph on top of the shared partition/indexer layer.

### 5. Afterwards (not part of this plan)

The algorithm ideas (models in [model/](model/)):

1. Compute at the tail + one rebalancing step (cost-based, DPD/IDPD): the growing imbalance on skewed inputs
2. Forward-style message trimming: send volume of the global phase
3. Compact-forward skip in the local phase: for an edge v→u every common out-neighbor w comes after u in
   orientation order, so with N+(v) in orientation order the intersection can start after u. Today's degree order
   is slow because of its comparator (degree lookups per comparison, hash map for ghosts), not the order. Ways to
   get the skip cheaply:
   - a precomputed integer rank per vertex (one sort of local vertices and ghosts by (degree, id)), or compact IDs
     numbered in orientation order (step 4), so neighborhoods sort by plain integers
   - the forward algorithm (Schank & Wagner): process vertices in orientation order and append each v to A(u) of
     its out-neighbors; the A lists are in orientation order by construction and intersect with the skip built
     in, without sorting neighborhoods (one vertex sort, m extra entries of memory)
   - flag intersection (idea 4) makes the skip unnecessary: it never scans N+(v) per edge
   In the global phase, message trimming (idea 2) is the same skip on the sender's side; it needs the orientation
   order as a cheap key there
4. Flag intersection in the local phase on compact IDs: mark N+(v) in an array of n+g flags, then scan each N+(u).
   Saves the d+(v) term per edge (the hubs' quadratic share) and the sorting of local neighborhoods. The global
   phase keeps merging (received lists hold global IDs) and needs only the contracted graph sorted. Compare with
   idea 3, both target the local phase

5. Interleave the local and global phase: send the global phase's messages first (they depend only on the
   oriented graph; for CETRIC the remote part of N+(v), contraction is structural) and handle received messages
   by polling within the local phase. A rank then needs its own local + global work instead of the slowest rank's
   local phase plus the slowest rank's global phase, and waiting overlaps with computation. Pre-CETRIC could also
   run the contracted graph's degree exchange during the local phase (its local phase needs no ghost degrees).
   Upper bound from the first runs (barriers between the phases + `termination`): 2–5% of an iteration without
   locality or skew, 13% on RHG at 192 cores and 20–25% at 768, growing with p; complements idea 1 (hides the
   imbalance that idea 1 removes). Costs: no separate phase regions (measure with accumulated counters, keep it
   switchable), polling granularity, peak memory (no contraction before messages are buffered). A first
   approximation needs no code: `global-synchronization` off removes the barriers. After step 2 (a driver of
   swappable phases)

6. Owner-side relevance filtering of messages. An entry w of N+(v) sent to rank r can only close a triangle there if
   r holds an in-neighbor u -> w; call these ranks P(w). w's owner knows P(w) (it knows w's neighbors, their ranks
   and, after the degree exchange, the orientation), so it sends P(w) to the ranks in P(w), and senders drop
   entries w with r not in P(w). Exchange cost Σ_w |P(w)|², independent of p; sets above a cap are not sent (hub
   entries are always kept). Model on a random graph without locality ([model/owner_filter.py](model/owner_filter.py),
   exact rank lists, 16 bits per rank ID): entries kept 74% / 34% / 16% / 8% at p = 16 / 64 / 192 / 768, exchange
   ~0.3 of the message volume, total volume 0.88× / 0.58× / 0.45× / 0.39×; the receivers' merge work shrinks with
   the kept entries. Order: degree exchange -> orientation -> P(w) exchange -> global phase. Refinements: a small
   Bloom filter over rank IDs instead of exact lists (exchange ~0.08 at 4 bits per ID, some false positives); not
   keeping entries just because w lies on the receiver. Variants ruled out:
   - R(w) (all ranks with a neighbor of w) piggybacked on the degree exchange: no extra round, but too coarse,
     total volume 0.91–1.17×
   - receiver-side Bloom filters of W_r = ∪ N+(u), u on r ([model/bloom.py](model/bloom.py),
     [model/bloom_adaptive.py](model/bloom_adaptive.py)): every sender needs every receiver's filter; with a fixed
     number of bits per element the volume grows to 2.2× at 768 cores, with sizes adapted per pair it stays ≤ 1×
     but gains little beyond a few hundred cores (traffic per pair shrinks as 1/p, the filter content does not).
     Filtering at grid relays scales better but costs unpacking and repacking every message there, and needs grid
   - one global Bloom filter of all edges: the sender could send only candidate wedges, but it needs O(m) memory
     per rank or node (6.4 GB per rank at 3072 cores with the `phases` size and 4 bits per edge), and building it
     is an all-reduce over that array
   Still to model: RHG (hubs) and inputs with locality; [model/relevance.py](model/relevance.py) gives the upper
   bound (share of entries usable at all)
7. Reuse a received message for several receivers: N+(v) arrives once per rank (`last_proc_`), but
   `process_neighborhood` merges the whole message against N+(u) for every local receiver u. Marking the message
   once and scanning each N+(u) (idea 4 applied to the global phase) saves |message| per additional receiver; a
   counter for receivers per message shows how much
8. Faster intersection kernels: the local phase is about 60% of the time with locality. SIMD merge (AVX-512 on
   SuperMUC-NG, AVX2) and galloping for very different sizes (hubs; `--intersection-method binary|hybrid` exists)
9. Cost-based load balancing: measure the existing `--primary-cost-function` D / DPD / IDPD on `phases` before
   implementing idea 1 (how much one repartitioning fixes the RHG imbalance, and at what redistribution cost). The
   redistribution itself fits `dstl::redistribute` (kamping-v2): a cost prefix sum split into contiguous vertex
   ranges gives each rank a send and a receive window, one call for the per-vertex data and one for the edges
10. Degree-1 / 2-core pruning: vertices of degree < 2 are in no triangle. `--pseudo2core` only skips their sending
    today; removing them (iteratively, for the 2-core) matters for real-world graphs, not for the synthetic inputs
11. Message compression (delta + varint over sorted lists): low priority, the global phase is computation-bound on
    the clusters; possibly worth it at large p with grid indirection, where every message travels twice

12. Skip entries the receiver already knows (the removed `--skip-local-neighborhood`, which segfaulted for all
    three algorithms at removal): the sender drops the entries w of N+(v) that live on the receiving rank r; r
    rebuilds them from its own cut edges (v, w) and orients them with the exchanged degrees. Saves the share of
    entries owned by the receiver: about 1/p without locality, more with partial locality (real-world graphs with
    a good partition). Complements idea 6 (which drops entries useless on r). Needs the ghost's local neighbors on
    r (an in-edge index per ghost), cheap with compact IDs. Measure first: a counter for message entries owned by
    the receiver

Plus: without locality, CETRIC should not pay for ghost expansion (fall back to patric's path, or pre-CETRIC).

## Open decisions

- [x] Hybrid (MPI + TBB) path: frozen during the refactoring (decided 2026-11-12). Step 2 writes the kernels
      sequential first, the hybrid ctests are disabled meanwhile, and the threads are ported afterwards
- [ ] BriefKAsten instead of the legacy message queue: still deferred (needs C++23, flow control off, a redesign of
      `ConcurrentBufferedMessageQueue`); revisit after step 3. Priority (2026-11-13): the algorithm and the
      synchronous communication first, tuning the asynchronous communication comes last
- [x] KaMPIng assertions off in the `experiments` preset (`KAMPING_ASSERTION_LEVEL=none`, decided 2026-11-13; the
      `phases` baseline ran with level 10 = exceptions)
- [x] Step-1 "decide" items: all removed except `--intersection-method` and `--pseudo2core` (decided 2026-11-12)
