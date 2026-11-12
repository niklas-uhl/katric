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

### 0. Baseline

- [ ] `phases` on SuperMUC-NG and HoreKa2 at `1962d4e` (first results in, the remaining core counts and permuted RHG
      pending); tag `pre-refactor` on `1962d4e` (the algorithm code is unchanged since)
- [ ] `cetric` (pre-CETRIC) appended to `phases` as a fourth configuration, so a rerun only adds its runs (the
      indices of the running configurations stay the same)
- [ ] Decide whether `phases` should also run `communication-policy: grid` (the paper's choice at large p)

### 1. Prune

Options that are dead or only used in old exploration suites (`experiments/legacy`), never in the paper suites.

- [ ] Remove options read nowhere: `--orient-locally`, `--omp-schedule`, `--omp-chunksize`, and the config fields
      `synchronized`, `buffer_threshold`, `max_degree_threshold_alpha`, `empty_pending_buffers_on_overflow`
- [ ] Check and remove the barely used ones: `--degree-filtering`, `--dense-load-balancing`,
      `--edge-partitioning-static`, `--high-degree-threshold-scale`, `--parallelization-method` (OpenMP variants),
      `--tbb-partitioner`
- [ ] `--flag-intersection`: dead today. `pre_intersection` / `post_intersection` / `intersect` are never called,
      `is_v_neighbor_` is never sized and would be indexed by global ID. Remove it; it comes back on compact IDs
      (idea 4, after step 4)
- Keep `--algorithm cetric` (pre-CETRIC, `run_cetric`). Not in the paper suites, but it orients cut edges outward
  in the local phase: no degree exchange before it (only on the contracted graph) and no ghost expansion, while
  triangles with two local vertices are still found locally and the contraction stays correct. The price: the
  √(2m) bound on out-degrees does not hold for vertices with many cut edges. Old real-world data: same
  local/global split as cetric-x, as fast or faster single-threaded. Decide between `cetric` and `cetric-x` from
  `phases` (step 0)
- [ ] Decide (proposed: remove): `--skip-local-neighborhood`, `--pseudo2core`, `--compact-degree-exchange` (step 3
      replaces the degree exchange anyway), `--intersection-method binary|hybrid` (tried in old suites, all off in
      the paper configs)
- [ ] Decide (proposed: remove): `--id-node-ordering false` (pure degree ordering): 1.6–2.5× slower on the
      clusters, sorting up to 13×; compact-forward ordering (idea 3) gets the skip with a cheaper key
- [ ] Decide (proposed: keep N, DPD, IDPD): cost functions N/D/DH/DDH/DH2/DPD/IDPD/D* and primary load balancing.
      Unused in the paper, but DPD/IDPD are the starting point of idea 1
- [ ] Remove commented-out code blocks in the touched files (`preprocessing`, `cetric_edge_iterator.h`)

### 2. One kernel per phase

- [ ] One intersection kernel for the local phase, parameterized by "with ghosts" and the execution policy, instead
      of `run_local_sequential`, `run_local_sequential_with_ghosts`, `run_local_parallel`,
      `run_local_parallel_with_ghosts`, `run_local_parallel_edge_partitioned(_with_ghosts)`
- [ ] Counters (`wedge_checks`, `intersection_size_*`, triangles) collected in one place per kernel
- [ ] `run_patric` / `run_cetric` / `run_cetric_new` as a sequence of phase functions with the same signature (graph, config,
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

Open: which order the compact IDs follow.

- Global ID order (as above): the global phase merges received global-ID lists against `to_global(N+(u))`
  directly. The local phase gets no skip (see idea 3).
- Orientation order (degree, id), computed once after the degree exchange: sorting by ID is then sorting in
  orientation order, so the local phase gets the compact-forward skip (idea 3) and the sender can trim messages
  (idea 2) with plain integer comparisons. Local vertices are then no longer one contiguous block (local index via
  an array), and the global phase sorts the contracted graph by global ID (small with locality) or merges by key.
- Decide together with ideas 2–4: with flag intersection (idea 4) the local phase needs no order at all.

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

Plus: without locality, CETRIC should not pay for ghost expansion (fall back to patric's path, or pre-CETRIC).

## Open decisions

- [ ] Keep the hybrid (MPI + TBB) path during the refactoring, or freeze it and port it after step 2?
- [ ] BriefKAsten instead of the legacy message queue: still deferred (needs C++23, flow control off, a redesign of
      `ConcurrentBufferedMessageQueue`); revisit after step 3
- [ ] Which of the step-1 "decide" items stay as options for the dissertation's ablations
