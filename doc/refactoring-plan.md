# katric refactoring plan

Working checklist for restructuring the algorithm code before the algorithmic changes for the dissertation version
(compute at the tail + rebalancing, message trimming, compact-forward ordering). Tick items (`[x]`) as they land and
add the commit hash. The build/tooling modernization is done and tracked in [modernization-plan.md](modernization-plan.md).

## Why now

The laptop run of the `phases` suite (`experiments/local/phases.suite.yaml`, 1–8 cores, weak scaling) shows where
the time goes, and the main costs are in code that the algorithm ideas would touch anyway:

- Without locality (permuted RGG/RHG), preprocessing dominates next to the global phase. Orientation and the degree
  exchange take about 0.2 s at 8 cores, against 0.01 s on 1 core with the same per-core size. The degree
  comparator looks up every ghost's degree in an `absl::flat_hash_map` (`AuxiliaryNodeData`).
- The global phase spends about 30% in `termination`, even on Poisson graphs: most received messages are only
  handled when the buffers are flushed at the end.
- On RMAT, barrier waits and `termination` dominate: imbalance, not work.
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

### 0. Baseline

- [ ] SuperMUC-NG run of `phases` (`just run-supermuc-experiment phases ...`), tag `pre-refactor`
- [ ] Decide whether `phases` should also run `communication-policy: grid` (the paper's choice at large p)

### 1. Prune

Options that are dead or only used in old exploration suites (`experiments/legacy`), never in the paper suites.

- [ ] Remove options read nowhere: `--orient-locally`, `--omp-schedule`, `--omp-chunksize`, and the config fields
      `synchronized`, `buffer_threshold`, `max_degree_threshold_alpha`, `empty_pending_buffers_on_overflow`
- [ ] Check and remove the barely used ones: `--degree-filtering`, `--dense-load-balancing`,
      `--edge-partitioning-static`, `--high-degree-threshold-scale`, `--parallelization-method` (OpenMP variants),
      `--tbb-partitioner`, `--flag-intersection` (and `is_v_neighbor_`)
- [ ] `--algorithm cetric` (the pre-CETRIC variant without ghost expansion, `run_cetric`): used in 25 legacy suites,
      in none of the paper ones. Removing it also removes the only (stub) secondary load balancing path
- [ ] Decide: `--skip-local-neighborhood`, `--pseudo2core`, `--compact-degree-exchange`,
      `--intersection-method binary|hybrid` (tried in old suites, all off in the paper configs)
- [ ] Decide: `--id-node-ordering false` (pure degree ordering). It is 1.4–3.5× slower on the laptop (sorting up to
      12×), and compact-forward ordering (idea 3) would replace it with a cheaper key
- [ ] Decide: cost functions N/D/DH/DDH/DH2/DPD/IDPD/D* and primary load balancing. Unused in the paper, but
      DPD/IDPD are the starting point of idea 1, so probably keep the framework and drop the variants nobody needs
- [ ] Remove commented-out code blocks in the touched files (`preprocessing`, `cetric_edge_iterator.h`)

### 2. One kernel per phase

- [ ] One intersection kernel for the local phase, parameterized by "with ghosts" and the execution policy, instead
      of `run_local_sequential`, `run_local_sequential_with_ghosts`, `run_local_parallel`,
      `run_local_parallel_with_ghosts`, `run_local_parallel_edge_partitioned(_with_ghosts)`
- [ ] Counters (`wedge_checks`, `intersection_size_*`, triangles) collected in one place per kernel
- [ ] `run_patric` / `run_cetric_new` as a sequence of phase functions with the same signature (graph, config,
      state), so a phase can be swapped (e.g. the tail-compute global phase of idea 1) without copying the driver
- [ ] The `if (num_threads > 1) ... else ...` duplication around `tbb::task_arena` in a single helper

### 3. Communication layer

The deferred kamping-v2 plan from modernization-plan.md, plus the degree exchange.

- [ ] Pass a communicator through `CommunicationUtility`, the graph communicators and the edge iterator instead of
      the ~40 hard-coded `MPI_COMM_WORLD`; duplicate it at startup
- [ ] Degree exchange (`graph_communicator.h`, `get_ghost_data`): `kamping::v2::alltoallv` for `--dense`,
      `dstl::grid_alltoallv` as the 2D variant (O(√p) messages instead of O(p) at 3k ranks), `dstl::sparse_alltoall`
      instead of the NBX in `CommunicationUtility`
- [ ] Degree-exchange format: today each entry is two `RankEncodedNodeId`s (16 bytes, the degree in an id slot) and
      per-vertex deduplication uses an `absl::flat_hash_set<PEID>`; send vertex + degree in a packed struct and
      deduplicate with a last-rank marker as in the global phase
- [ ] Buffers per rank as a flat vector with displacements instead of `absl::flat_hash_map<PEID, vector>`

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

- [ ] Adopt KaCCv2's `VertexPartition` and `DenseIndexer` / `SparseIndexer` (`KaCCv2/src/kacc/graph/`, without the
      split-vertex `ReplicaIndexer`) for the partition table and the local indexers, on plain `uint64` global IDs.
      Copy or share them (a small common library) — decide when starting
- [ ] Compact IDs in the adjacency, built once after `find_ghosts`; per-ghost arrays for global ID, rank and degree.
      In KaCCv2's concepts this is a neighbor handle (`neighbor_t` != `vertex_id_t`, projected by
      `neighbor_target`), so the indexers never see compact IDs
- [ ] Routing, local tests, `id_outward` and `local_prefix_end` on compact IDs (array lookup / range test);
      `RankEncodedNodeId`, `find_ghost_ranks`, `--binary-rank-search` and `AuxiliaryNodeData` removed
- [ ] Messages carry global IDs. The numbering preserves the global order, so a receiver merges the received list
      against `N+(u)` mapped through `to_global` (monotone) without translating the received IDs
- [ ] Contraction (`compact()`, today a `SparseNodeIndexer`) and the cetric-x ghost CSR on the new IDs
- [ ] Replace the `std::set<RankEncodedNodeId> ghosts` filter in `run_distributed` (unused when empty)
- [ ] Compare against `pre-refactor` in `phases`: preprocessing (orientation, degree exchange) and `send_ns`

KaCCv2's graph class itself (`SplitDistributedCSRGraph`, CSR plus split-vertex replicas) is not adopted: katric
needs the out/in split per vertex after orientation, the ghost CSR and contraction, so it keeps its own oriented
graph on top of the shared partition/indexer layer.

### 5. Afterwards (not part of this plan)

The algorithm ideas (models in [model/](model/)): compact-forward ordering in the local phase (idea 3),
forward-style message trimming (idea 2), compute at the tail + one rebalancing step (idea 1). Also the
`termination` share of the global phase (flush threshold), which the baseline shows as a target of its own.

## Open decisions

- [ ] Keep the hybrid (MPI + TBB) path during the refactoring, or freeze it and port it after step 2?
- [ ] BriefKAsten instead of the legacy message queue: still deferred (needs C++23, flow control off, a redesign of
      `ConcurrentBufferedMessageQueue`); revisit after step 3
- [ ] Which of the step-1 "decide" items stay as options for the dissertation's ablations
