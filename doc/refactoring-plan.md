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

## Variants

Decided 2026-11-19: one algorithm, KaTric, with one option. The old `--algorithm` names were misleading (`patric` is
not Arifuzzaman's PATRIC but his surrogate approach, Arifuzzaman 2015; `cetric` was not the paper's CETRIC and was the
default).

| | DiTriC (default) | CETRIC (`--contraction`) |
|---|---|---|
| old name | `--algorithm patric` (still accepted) | `--algorithm cetric-x` (still accepted) |
| preprocessing | degree exchange, orientation by (degree, global id) | the same, plus the ghosts' out-edges to local vertices |
| stored | the out-neighbors N+(v) of the local vertices, including the cut out-edges | the same plus the ghosts' out-edges; every cut edge is stored on both of its ranks (as in the input) |
| local phase | intersections over the out-neighbors of the local vertices | also through the ghosts' out-edges: every triangle with >= 2 local vertices |
| global phase | `global_ids`: whole graph in place; sends the full N+(v) of the interface vertices | `contraction`: only the cut out-edges; sends the contracted N+(v) |

The code namespace, include directory, CMake targets and options are `katric` too (`katric::`, `include/katric/`,
`KATRIC_*`; before: `cetric`). The ghosts' out-edges and contraction come only together: contraction without them loses the triangles g < v < w (g remote,
v->w local); the ghosts' out-edges without contraction count them twice. Both variants push (surrogate); computing at the tail
(idea 1) would be a different algorithm.

pre-CETRIC (`--algorithm cetric`, local orientation with cut edges outward, reorientation by the contracted degree
after the contraction) is removed: it loses the out-degree bound at hubs (RHG at 1536: local phase 3.8-4.6 s, wedge
checks 508 G vs DiTriC's 130 G, max/avg 246) and never beats DiTriC. `--algorithm cetric` now fails; the code is at the
tag `pre-refactor`. Its one useful idea, reorienting the contracted graph, carries over to CETRIC (below).

Load balancing per variant (open, after the review):

- DiTriC: exchange the degrees once, orient, compute the cost (IDPD needs the in-neighbors, so before dropping them),
  then move only N+(v) (at most sqrt(2m) entries per vertex instead of d(v)); no second degree exchange. Today the
  primary load balancing moves the undirected neighborhoods before orientation (historical)
- CETRIC: moving N+(v) alone does not deliver v's in-edges from remote tails (the ghosts' out-edges to v), so the primary load
  balancing stays on the undirected input. The full-graph IDPD prices the wrong work (DiTriC's receive work); the
  global phase's cost is the contracted IDPD (sum over cut in-edges w->v of d+_cut(w) + d+_cut(v)), known only after
  the first partition. Option: at the contraction, exchange the cut degree, reorient the cut-edge graph by (cut degree,
  global id) taking each edge from whichever side keeps it (a local vertex's out-edge or a ghost's out-edge; bounds d+_cut by sqrt(2 m_cut)),
  exchange the cut out-degree, compute the contracted IDPD, and rebalance the contracted graph if its max/avg exceeds a
  threshold. After such a move the global phase must also intersect rank-local pairs (`global.h` skips them today).
  First step: the contracted IDPD's max/avg as a counter

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
      Results of the extension (`*_rerun_26_11_18`, binary `466103e`, i.e. still the old graph):
      - `degree-exchange`: dense and grid are equal up to 3072 cores; sparse (NBX) loses from 1536 on without locality
        (GNM exchange 1.58 vs 0.51 s at 3072). Keep dense. The transport is not the lever: 2-3 s of the region at
        768-3072 cores on permuted RGG were local work (exclusive time: finding the ghosts and filling
        AuxiliaryNodeData's hash map), the exchange itself 0.2-1.6 s. Stage 2 removed that local work
      - `no-sync`: dropping the barriers changes nothing beyond noise (±10%) for DiTriC and CETRIC on RGG, RHG and GNM,
        up to 3072 cores. Idea 5 (interleaving the phases) has little to gain on these inputs
      - RMAT (KaGen defaults, 2^16 / 2^20 per core) does not run at scale: 58 s per iteration at 48 cores (GNM: 2.3 s),
        230 s at 192 (job timeout), killed (memory) at 1536 / 3072. 40x GNM's wedge checks, and the busiest rank does
        6.4x the average global-phase work (local phase 11x); at 48 cores 45 of the 52 s global phase are other
        ranks waiting in `termination`. The case for idea 1 / cost-based balancing; small `lb-cost` runs on RMAT next
      - the `lb-cost` rerun at 768 (64-bit redistribute) has not run yet
      Decisions (2026-11-18): `--degree-exchange sparse` is not feasible at scale without locality, kept as an option
      (the code is small and local); the configs say `dense`. The barriers between the phases and
      `--global-synchronization` are removed: they only served the legacy phase timing and cost nothing measurable
      either way; imbalance is read from the work counters (max / avg over ranks) and `termination`. The `no-sync`
      suite is gone with the option; `lb-cost-rmat` added (DiTriC, none / DPD / IDPD, 48 / 192 cores)

- Results 2026-11-18 on the new graph (build `8d1eca2`, stage 2 + no barriers; `phases_26_11_18` 48 / 192 cores,
  `lb-cost-rmat_26_11_18`, `lb-cost_rerun2_26_11_18` 48 cores; RHG at 192 and lb-cost 192 / 768 still outstanding).
  Where to pick up: the three next steps at the end of this list
  - `phases`, new vs old graph (time per iteration): faster everywhere, x0.66-0.96. CETRIC without locality gains
    most (permuted RGG 192: 20.6 -> 13.5 s; preprocessing 9.6 -> 4.0 s and local phase 2.8 -> 0.9 s from the ghost
    rows); DiTriC preprocessing 6.4 -> 3.7 s, global phase -10-15% (rows in global ids); with locality x0.85-0.96.
    With locality the global phase now looks slower (RGG 0.03 -> 0.15 s): without barriers it absorbs the local
    phase's imbalance; the total still drops
  - new costs without locality: `input_graph` 2.1-2.3 s (one sort of ~8M (head, position) pairs per rank; includes
    building the graph, untimed before) and `degree_exchange` 1.9-2.7 s, mostly depositing the received values
    (binary search over ~6M ghosts per entry)
  - CETRIC vs DiTriC (direct communication): with locality all three algorithms are equal (RGG 1.78-1.85 s, RHG
    1.53-1.71 s at 48): the local phase is 75% of the time and the global phase is tiny, so CETRIC's 2.5-4.7x smaller
    send volume (RGG 2.0 vs 5.2 M entries at 48, RHG 1.7 vs 7.9 M) does not show. Without locality DiTriC is best
    (permuted RGG 9.6 / 11.5 s vs CETRIC 11.8 / 13.5 s, pre-CETRIC 10.9 / 12.6 s; permuted RHG 9.4 vs 11.6 / 13.7 s):
    contraction removes nothing (send volume and global work within 2% of DiTriC) and CETRIC pays for the ghosts' out-edges,
    pre-CETRIC for the second degree exchange. Grid indirection is slower than direct in every case up to 192 cores.
    CETRIC can only win where the global phase matters and locality removes much of it: larger p, smaller n/p, or
    real-world graphs with a good partition
  - `lb-cost-rmat`: IDPD fixes RMAT, DPD only partly. Time per iteration none / DPD / IDPD: 46 / 20.6 / 11.2 s at
    48 cores (x4.1 with IDPD), 187 / 49 / 23 s at 192 (x8.1). IDPD balances the global-phase work almost perfectly
    (wedge checks max/avg 6.4 -> 1.05 at 48, 16.4 -> 1.06 at 192); what remains is send-volume imbalance (x2.5-3.05,
    the hubs' messages). IDPD counts in-neighbors, where DiTriC does the work (at the receiver); DPD models computing
    at the tail (idea 1), which DiTriC does not do: Arifuzzaman's surrogate argument, measured. Balancing costs
    1.3-3.5 s. RMAT does not weak-scale even balanced: global work 22 -> 149 * 10^9 wedge checks from 48 to 192 cores
  - `lb-cost` RHG at 48 on the new graph: none 1.58, DPD 2.99, IDPD 2.63 s. RHG's work is already balanced at this
    scale (local max/avg 1.08); repartitioning moves range boundaries into the hub region and turns local work into
    global work (global wedge checks 0.01 -> 1.27 * 10^9)
  - Next steps (in this order):
    1. Rebalance only when needed: compute the IDPD cost, and redistribute only if its max/avg over ranks exceeds a
       threshold (one allreduce); covers RMAT (needs it) and RHG (does not)
    2. The two new costs: build the input graph by bucketing the cut edges by owner rank first, then sorting per
       bucket; deposit the exchanged ghost values by a linear merge (dense mode receives in rank order, each source's
       part ascending, i.e. in global-id order already; sparse and grid need a sort)
    3. Idea 1 (compute at the tail, balanced by DPD) against DiTriC + IDPD on RMAT, the bar to beat
  - Message trimming (idea 2) is on branch `feat/message-trimming` (see idea 2 below); suite `trimming` submitted

- Results 2026-11-19 (`trimming_26_11_18` 48 / 192 / 1536, binary `8f66911-dirty`; `lb-cost_rerun2_26_11_18` 192 /
  768; `phases_26_11_18` 1536; `degree-exchange_rerun_26_11_18` GNM 768; trimming 768 still outstanding)
  - Next step 1 (threshold on the IDPD max/avg) confirmed and stays first: on RHG, DPD / IDPD are ~2x slower at every
    scale (none / DPD / IDPD: 1.73 / 3.26 / 3.35 s at 192, 2.04 / 3.73 / 3.66 s at 768). Repartitioning alone costs
    1.0-1.3 s, more than the whole global phase without it, and global wedge checks grow ~100x (210 M -> 22-26 G at
    768). The 768 runs no longer die (64-bit displacements of PR #126 work)
  - Global phase without locality, per rank (avg), split with the counters: intersection work hardly grows with p
    (DiTriC permuted RGG handle 2.7 -> 2.9 s from 48 to 1536), the rest (phase - send - handle: polling, MPI progress,
    buffers) does: 1.1 -> 3.6 s, 44-64% of the phase at 1536 (GNM 1.8 of 2.8 s). On RHG handling is ~1 ms, so its
    `termination` region really is waiting (for the local phase's imbalance). Since `5618dc5` the counters `poll_ns`
    (in `poll`, outside the handler) and `termination_handle_ns` (handling inside `terminate`; ~40% of handling on the
    laptop) split this exactly
  - Message trimming: send volume x0.56-0.60, messages -37%; global phase of DiTriC on permuted RGG -8 / -9 / -18% at
    48 / 192 / 1536 (GNM -3 / -10 / -15%), CETRIC up to -24% (RGG) / -36% (GNM) at 1536; total -3 to -11%. Sending
    costs +27% (48) to +5% (1536), handling -15-18%; at 1536 most of the gain is the rest (-1 s), not handling
    (-0.5 s). Wedge checks -1-2%. On RHG volume -86-93% but no time effect
  - `phases` at 1536: the new graph is faster than the old everywhere (x0.56-0.99). Without locality DiTriC stays best
    (permuted RGG 15.3 vs CETRIC 17.7, pre-CETRIC 16.9 s); about 7 s of it is before counting (input graph 2.5 s,
    preprocessing 4.5 s of which degree exchange 3.5 s): next step 2. Pre-CETRIC on RHG breaks: local phase 3.8 s,
    508 G wedge checks (DiTriC 130 G) with max/avg 246 (one rank 81.6 G); likely its local orientation before the
    global degrees are known (not checked in the code)
  - Degree exchange at 3072: grid beats dense for the first time on GNM (exchange 0.44 vs 0.50 s), not yet on permuted
    RGG (1.73 vs 1.53 s); RHG is dominated by the hub (6.8 s either way). Expected to pay off beyond ~3k cores; needs
    runs at >= 6144
  - Communication overhead (the rest above, trimming's gain, grid in the global phase) is deferred until the BriefKAsten
    migration, which may change it; measure again then rather than tuning the legacy queue now

- Results 2026-11-21 (`baseline_26_11_20` 192-3072, `baseline-rmat_26_11_20` 48-768, binary `30461a68-dirty`;
  analysis scripts not kept, `eval/katric_eval.py` loads both). Where to pick up: the next steps at the end
  - Best configuration nearly everywhere: DiTriC, direct, message trimming, no load balancing. CETRIC wins only on RGG /
    RHG with locality, by 2-3%. Grid beats direct on GNM from 768 on (2.54 vs 3.07 s at 1536, 2.80 vs 4.10 s at 3072:
    polling 0.6 vs 1.9 s) and on permuted RGG at 3072 (with trimming); on RMAT it is 1.5-3x slower (polling max/avg
    4.8). GNM with direct is noisy (up to 34% spread across iterations at 768-1536)
  - IDPD never pays off outside RMAT: +0.4-0.7 s (RGG), ~1.3 s (RHG, GNM), 5.5-7.2 s (permuted inputs), and the work
    was balanced before (wedge checks max/avg 1.01 -> 1.03-1.05). On permuted RGG at 1536 `primary_load_balancing`
    is 6.8 s: cost-function communication 3.4 s (two push_to_ghosts), evaluation 0.6 s, redistribution 0.24 s, and
    ~2.5 s rebuilding the graph (`to_kagen_graph` + `from_kagen`)
  - Of the cost-function communication, the transfer is ~0.7 s (the minimum of `exchange`, as alltoallv waits for the
    last rank); the rest is `prepare` (bucketing, avg 1.17 s, ~70 ns per adjacency entry) and waiting for one rank
    whose `prepare` takes ~2x the average (max 2.46 s) although its edge count is not larger. The same in the degree
    exchange. On the laptop bucketing costs ~8 ns per entry (8 ranks, and a single-process benchmark at p = 1536) and
    shows no fixed straggler: the cost and the straggler come from the cluster node. On RMAT it is the transfer itself
    (`exchange` max 5.2 s, `prepare` 0.06 s avg): the unbalanced input's hub ranks
  - RHG at 3072 (1.2 -> 5-10 s) is the input: a hub of degree 19.1 M (0.76 M at 1536), its rank holds 3.3x the average
    edges and spends 3.7 s in `input_graph` (the cut-edge sort of `from_kagen`; the other ranks 0.09 s) while
    everyone waits in the degree exchange; KaGen's generation 3.6 -> 66 s. Read RHG at 3072 with that in mind
  - RMAT with IDPD: work balanced (wedge checks max/avg 1.05-1.16), send volume not (max/avg 2.3-3.0); at 768 the
    slowest rank polls 35 s against 10.8 s of handling, global phase 52 s, nearly all `termination`. Per-rank wedge
    checks triple from 48 to 768 (483 M -> 1.42 G). At 768 only direct + trimming fit: the six other configurations
    (no trimming, every grid run) were killed (SIGKILL after 40-53 s, presumably memory)
  - On permuted inputs ~4-4.5 s per iteration are before counting: `input_graph` ~2.2 s, degree exchange 1.3-2.0 s
  - Done 2026-11-21:
    - `fadbd12`: push_to_ghosts builds its send pattern once per graph (`GhostPushPattern`, region `prepare/pattern`),
      later exchanges only gather; one pass into reserved (u32, u32) pairs (a two-pass build was 1.6x slower in the
      benchmark, one pass ~15% faster than before). `exchange_ghost_degrees` returns at once when the graph has its
      degrees. Locally the cost function's `prepare` 0.22-0.28 -> 0.08-0.12 s (8 ranks, 3 iterations)
    - `e13a2de`: `--rebalance-threshold` (redistribute only if the cost's max/avg exceeds it; otherwise the graph is kept
      with its degrees and pattern, so preprocessing has no degree exchange; counters `rank_cost`, `rebalanced`) and
      `--primary-cost-function IDD` (IDPD with a ghost's out-degree replaced by its degree: no second exchange). Model
      `doc/model/cost_proxy.py`: IDD balances RMAT's push work to 1.18-1.70 (IDPD 1.03-1.53, edges 1.48-2.30) and its
      max/avg on the input tracks IDPD's (RMAT 4-40, RGG 1.0-1.3), so a threshold of ~1.5 separates them. Locally:
      permuted RGG is kept, 1.54 s vs none 1.44 vs IDPD unthresholded 2.50 s; RMAT IDD 2.33 vs IDPD 2.29 vs none 5.05 s
    - Suites (submitted 2026-11-21): `straggler` (per-rank times of the ghost exchanges at 1536 through Caliper's
      recorder, read with `eval/perrank.py`), `rebalance` (none / IDPD at 1 / IDPD and IDD at 1.5), `rebalance-rmat`
      (IDPD vs IDD at 1.5). Suites now use `ncores: node-size-pow2`, the range is set at job creation
  - Next steps (in this order; replace the list of 2026-11-18):
    1. Evaluate `straggler`, `rebalance`, `rebalance-rmat`: is the slow rank fixed (same rank / node), does the
       threshold keep the graph everywhere but RMAT and cost little, IDD vs IDPD on RMAT; then pick the defaults
       (threshold, IDD or IDPD) and decide on making message trimming the default (wins or ties everywhere)
    2. The cost of `from_kagen` (was step 2): bucket the cut edges by owner rank first, then sort per bucket; check with
       a benchmark first, as for the bucketing. Matters twice per iteration with redistribution, and on hub ranks. (The
       other half of the old step 2, depositing the ghost values, is done: push_to_ghosts receives into place)
    3. A peak-memory counter per phase: where RMAT runs out of memory at 768 (message buffers or the redistribution's
       copies); needed for the BriefKAsten comparison anyway
    4. RMAT's send-volume imbalance after balancing: a volume term in the cost (as DPD+reply in the model), or idea 1
       (compute at the tail, balanced by DPD) against DiTriC + IDPD / IDD on RMAT (was step 3)
    5. Cluster checks: RHG at 3072 with another seed (is the 19 M hub an outlier), more iterations or the minimum for
       GNM with direct

- Results 2026-11-23 (`straggler_26_11_21` 1536, `rebalance_26_11_21` 192-3072, `rebalance-rmat_26_11_21` 48-768; all
  runs complete, no errors). Where to pick up: the next steps at the end
  - `straggler` (per-rank recorder, `eval/perrank.py`): not one rank but 33-62 of 1536 (2-4%), all on nodes 21-31
    (spread over the cores), 2.5-2.9x slower in `prepare`; the same ranks in every iteration and in both
    configurations, also after IDPD's redistribution. Only the sub-region `prepare/pattern`
    (`build_ghost_push_pattern`) is slow there (0.99 vs 0.34 s, exclusive): `input_graph`, `orientation`, `sorting`,
    `global_ids` and the global phase run at the others' speed, so neither a slow core nor more data. The pattern
    build is the one phase with a random `owners_[ghost]` lookup per adjacency entry and a freshly reserved 8 B x
    adjacency-entries buffer: hypothesis memory (pages / TLB, e.g. no transparent huge pages on those ranks), not
    tested. Costs ~0.65 s per pattern build at 1536 (every rank waits in `exchange`), ~6% of a permuted run
  - `rebalance`, cost max/avg on the input and whether it rebalanced at 1.5:
    GNM / RGG / permuted RGG: IDPD 1.03-1.06, IDD 1.04-1.06, kept (both);
    RHG <= 1536: IDPD 1.08-1.18 kept, IDD 2.0 rebalanced; permuted RHG <= 1536: IDPD 1.04-1.14 kept, IDD 1.65-1.75
    rebalanced; RHG (both) at 3072: IDPD 3.35-3.40, IDD 10-15, rebalanced (both)
    - IDD overestimates RHG (the hubs' degree instead of their out-degree) and rebalances it for nothing: RHG
      1.3 -> 2.5 s, permuted RHG 8.9 -> 12.6 s. The model's "IDD's max/avg tracks IDPD's" does not hold on RHG
    - RHG at 3072: the 19 M hub sets off every cost function, and rebalancing hurts (none 5.3 s, IDPD 9.0 s; even the
      global phase 0.78 -> 1.7 s)
    - The check when the graph is kept, vs none: IDPD +0.15-0.25 s on RGG (~15%), +0.2-0.5 s GNM, +0.9-1.0 s (~11%)
      permuted; IDD about half (its one exchange replaces the degree exchange; left is `cost_function_evaluation`,
      avg 0.47 s on the permuted inputs, 0.2 s on RGG)
    - fadbd12: IDPD unthresholded, permuted RGG 1536 17.4 -> 16.2 s (load balancing 6.8 -> 5.8 s); none unchanged
    - The input's edge max/avg (ingest `local_edges`) tracks IDPD's cost max/avg on every input: GNM / RGG / permuted
      1.00-1.05 (cost 1.03-1.06), RHG 1.05-1.16 (1.04-1.18), RHG 3072 3.33 (3.35-3.40), RMAT 5.1-26.9 (6.9-44)
  - `rebalance-rmat`: IDPD beats IDD at every scale (9.7 / 20.3 / 68.4 s vs 10.6 / 22.3 / 85.1 s at 48 / 192 / 768):
    IDD balances worse (longer global phase) and saves little, as RMAT's cost-function communication is the hubs'
    transfer (exchange 4.4 vs 4.8 s at 192). The global phase still grows superlinearly (6.8 s at 48 -> 52 s at 768)
    as in `baseline-rmat`: send-volume imbalance (next step 4 of 2026-11-21)
  - Decisions (2026-11-23):
    - Defaults: `--primary-cost-function IDPD`, `--rebalance-threshold 1.5`, message trimming on
      (`--no-message-trimming` turns it off; kaval omits false flags). IDD removed as a cost function
    - `--rebalance-precheck degree-bound` (default; user, 2026-11-23): IDD lives on as the pre-check. The cost with
      each ghost's out-degree bounded by its degree needs only the degree exchange, which preprocessing (and the
      orientation, if the graph is kept) needs anyway, and bounds IDPD per vertex; at or below the threshold the graph
      is kept without the out-degree exchange (counter `rank_cost_bound`), above it IDPD decides (`rank_cost`), so
      IDD's overestimate on RHG costs one exchange, not a redistribution. Left when kept: the evaluation (0.2-0.6 s
      above). Not a bound on IDPD's max/avg in theory (the average grows too), so a case with the bound below and IDPD
      above the threshold would be missed. `--rebalance-precheck none` computes IDPD always. (An edge-count
      pre-check was considered: free and it tracked IDPD here, but blind on inputs balanced by edges by
      construction, e.g. `distribution=balance-edges`)
    - Suites that relied on the old defaults now say so (`rebalance-threshold: 1`, `no-message-trimming`), so that
      they rerun as before
  - Next steps (in this order; replace the list of 2026-11-21, its steps 2-5 stay):
    1. Straggler: rerun `straggler` (the instrumentation of 2026-11-23, `profiling::ResourceCounters` in the pattern
       build: per-rank page faults, CPU and NUMA node, huge pages, and cycles / instructions / dTLB load misses if perf
       events are allowed; `eval/perrank.py --counters`); do the slow ranks stay with the nodes, and is it the TLB /
       pages. The Linux part (perf events, getcpu, smaps_rollup) is untested locally (macOS): check the counters of a
       small cluster run first. Also run `rebalance` once with the new defaults (degree-bound pre-check) to measure
       what the check costs when it keeps the graph
    2. The cost of `from_kagen` (2026-11-21 step 2)
    3. A peak-memory counter per phase (2026-11-21 step 3)
    4. RMAT's send-volume imbalance (2026-11-21 step 4)
    5. Cluster checks: RHG at 3072 with another seed (as rebalancing hurts there, decides whether 1.5 needs more
       than the max/avg), GNM with direct with more iterations

- Results 2026-11-23, later (`straggler_26_11_23` 1536 on other nodes, `rebalance_26_11_23` with the new defaults;
  `rebalance-rmat_26_11_23` not in yet; the IDD configuration of `rebalance` failed as expected, permuted RHG 3072 at
  1.5 missing: the job's log ends there)
  - Straggler: the same ranks are slow (r1346, r1038, r1052, r1055, ...) on a different set of nodes (i07 instead of
    i04): it follows the rank / its data, not the hardware. On those ranks the pattern build executes the same
    instructions (3.3e8) in 3.3x the cycles with 22x the dTLB load misses (6.8 M vs 0.3 M per build, about one per
    adjacency entry); page faults normal, no major faults. Their processes have ~15-20 MB fewer transparent huge pages
    (280 vs 297 MB). Input sizes (local / cut edges, wedges) do not separate them. Hypothesis: a ~32 MiB array
    (`owners_`, 4 B per ghost, ~8.38 M ghosts; also the 32-bit adjacency) lands below / above glibc's dynamic mmap
    threshold (at most 32 MiB) depending on the rank's allocation history, and is then not backed by huge pages.
    `straggler` now has allocator variants (GLIBC_TUNABLES hugetlb, fixed mmap threshold 1 MiB / 32 MiB) and records
    the glibc version and THP mode (adiak `memory.*`). Fix candidates once confirmed: madvise(MADV_HUGEPAGE) on the large
    arrays, or a pattern build without the `owners_` lookup (binary search over the per-owner ghost offsets, p + 1
    entries)
  - `ResourceCounters` cost ~40 ms per pattern build (avg; 3x the build itself on RGG / RHG), most likely reading
    smaps_rollup: now opt-in (KATRIC_RESOURCE_COUNTERS, set by `straggler` only). `rebalance_26_11_23`'s times include
    it (~0.04 s per pattern build)
  - Pre-check `degree-bound` (IDPD at 1.5, vs 2026-11-21): permuted inputs kept, load balancing 2.9 -> 2.3 s at 1536,
    the check costs +0.5 s (5%) over none instead of +1.0 s; RHG <= 1536 bound 2.0, IDPD 1.15 decides, kept (no
    redistribution); GNM +0-0.2 s, RGG +0.15 s over none (~0.04 s of it the counters). RHG 3072 still rebalances
    (IDPD 3.35) and is slower than none (9.4 vs 5.3 s)
  - Load balancing (proposal 2026-11-23; 1 and 2 implemented the same day, not yet measured on the cluster; 3:
    suite `rmat-volume` with per-rank counters, plus the counters local_nodes / out_edges / cut_out_edges /
    max_out_degree of the oriented graph in `preprocessing`):
    1. When the graph is kept, preprocessing orients with the same `degree_key` the cost function just evaluated:
       hand the cost function's orientation (is_out_edge / out-degrees) to `orient()` instead of recomputing it.
       Would make the check nearly free when it keeps the graph (preprocessing `orientation` ~0.36 s on permuted RGG).
       Done: Graph::keep_orientation() / orient_by_kept() (Lomuto partition on the kept flags), test
       KeptOrientationMatchesKey (DiTriC and CETRIC)
    2. Achievable balance: a vertex's cost cannot be split by contiguous ranges, so with a vertex costing more than
       the average rank, redistribution cannot reach the threshold (RHG 3072: one hub, rebalancing hurts). Compare the
       max/avg against max(max vertex cost, avg) instead of avg; check with the RHG seed rerun. Done: in
       exceeds_threshold() (counters `rank_cost_max_vertex`, `rank_cost_bound_max_vertex`); `--rebalance-threshold`
       now means max / max(avg, largest vertex)
    3. RMAT: after IDPD the work is balanced but the send volume is not (max/avg 2.3-3.0) and the global phase grows
       6.8 -> 52 s from 48 to 768 cores. Measure per rank where the volume comes from first (recorder on
       `rebalance-rmat`), then a volume term in the cost or idea 1 (compute at the tail, DPD)
    4. The redistribution itself (`from_kagen`, ~2.5 s at 1536 permuted): 2026-11-21 step 2

- Partial data 2026-11-23, evening (jobs still running: `straggler_rerun`, `rhg-seeds` beyond 1536 none,
  `rmat-volume` 192-768; `rmat-volume` 192 finished but not fetched)
  - `rmat-volume` 48 / 96 (per rank): send volume max/avg 2.3, tracking out_edges / cut_out_edges (max/avg 2.9-3.3,
    correlation 0.98-0.99) and local_nodes (max/avg 7-10). The top senders (every 16th rank: r15, r31, r47 at 48; r31,
    r63, r95 at 96) hold 15-40x the median vertices, low-degree ones: IDPD gives them many cheap vertices, so their
    out-edges (and cut out-edges) are ~3x the median, with fewer targets per cut edge (volume / cut out-edge 25-41 vs
    33-58). Work is balanced (wedge checks max/avg 1.05), receive volume nearly (1.18-1.25), and at 48 / 96 the global
    phase time too (max/avg 1.01-1.02): the volume imbalance does not cost time yet at this scale. Check at 192-768
    whether it does; if so, the cost needs a send term (Σ over a vertex's cut out-edges), not only IDPD's work
  - `rmat-volume` 192 / 384 (more data, same evening): every rank's global phase takes the same time (all wait in
    `termination`), so per-rank phase times say nothing; the rank finishing last is the one sending least. Per rank
    busy time (send + handle + poll) of the busiest rank ~ the phase (21.9 of 22.5 s at 384). Handling dominates:
    avg 5.95 -> 11.1 s from 48 to 384 (wedge checks per rank 4.6e8 -> 9.7e8: RMAT does not weak-scale), max/avg
    1.3-1.4 although wedge checks are balanced (1.03-1.09) -- the handling cost per wedge check differs between
    ranks. Polling grows with p (avg 0.8 -> 5.2 s, max 11.6 s at 384). Sending is small (avg 1.9 s at 384, max 5.9 s);
    up to 384 the send-volume imbalance (max/avg 2.3 -> 3.05) does not decide the time.
    At 768 it does: global phase 51.8 s, busy avg 28.3 / max 46.7 s; polling avg 11.5 / max 35.2 s (3.1x), sending
    avg 3.2 / max 10 s (3.2x), handling 1.25x; the busy time correlates 0.91 with send volume and with receive volume
    (receive max/avg 1.46), polling 0.87-0.90 with send volume / messages; wedge checks only 0.61. So the cost
    function needs a volume term from ~768 on (send: the vertex's cut out-edges x out-degree, as the model's
    DPD+reply), or idea 1; and polling is for BriefKAsten
  - `rhg-seeds` (1536 / 3072 except the `phases` seed's 3072 IDPD and seed 28475421 permuted 3072): the 19 M hub is
    an outlier of the `phases` seed (seeds 1 / 2: hub 1.0-1.4 M at 3072, RHG none 1.28-1.38 s at 3072, as at 1536).
    The defaults never rebalance RHG (IDPD max/avg 1.10-1.21 < 1.5), so the achievable-balance rule was not exercised
    yet (needs the `phases` seed at 3072). The degree bound is loose on RHG (max/avg 1.4-2.9, its largest vertex up to
    1.9x the average rank), so the out-degree exchange runs anyway: IDPD vs none +0.1-0.3 s on RHG, +0.6-1.3 s on
    permuted RHG (~10%). The orientation reuse shows: preprocessing on permuted RHG 0.46 s (2026-11-23 rebalance) ->
    0.20 s
  - `rhg-seeds`, `phases` seed at 3072 (the case for the achievable-balance rule): kept, by the degree-bound check
    already (its largest vertex sets the achievable); RHG 5.5 s vs none 5.2 s (was 9.4 s rebalanced), permuted
    16.3 vs 15.8 s (was 25.4 s). The rule works
  - `rebalance-rmat_26_11_23` (IDPD, degree-bound pre-check, before the orientation reuse): 10.0 / 13.7 / 20.3 /
    33.9 / 69.9 s at 48-768, as 2026-11-21 (9.7-68.4 s): the pre-check exceeds on RMAT and IDPD runs as before
  - `straggler_rerun` (all four variants; glibc 2.38, THP always, defrag madvise): the huge-page / mmap-threshold
    hypothesis is refuted. In every variant (hugetlb, fixed threshold 1 MiB / 32 MiB) the slow ranks stay (pattern
    max/avg 2.8-3.1) with the same 6.8 M dTLB misses; with a fixed threshold the median rank's misses drop 300 K ->
    1.3 K without changing its time. The slow ranks' misses do not come from where malloc places the arrays. Next:
    does the slow set follow the data (another seed) or the rank position (same seed at another p; the slow ranks are
    all >= ~1030, i.e. vertex ids >= 2^28)? Then perf record on one slow rank
- Straggler, 2026-11-24: perf on one rank is impractical on SuperMUC-NG, so printed diagnostics instead
  (`KATRIC_STRAGGLER_DEBUG`, include/katric/straggler_debug.h, before the region `pattern`): per rank the smaps VMAs
  and NUMA nodes of adjncy / owners and of two fresh copies of owners (a vector, a madvise(MADV_HUGEPAGE) buffer), and
  the pattern build's loop in stages (scan, is_local, owner lookup, dedupe; dedupe on both copies; the real build).
  Suite `straggler-debug` (1536): N18 with the `straggler` seed and seed 1 (data or rank?), N17 (no id >= 2^28) and
  N19 (ids >= 2^28 from rank 512). The Linux part is untested locally
- Results `straggler-debug_26_11_24` (all four inputs, 2 iterations, the Linux fields work): medians in ms
  | input / ranks       | scan | owner | dedupe | dedupe on copy | dedupe on THP copy | build |
  |---------------------|------|-------|--------|----------------|--------------------|-------|
  | N18, 1443-1493 normal | 14 | 144   | 153    | 153            | 153                | 306   |
  | N18, 43-93 slow     | 15   | 150   | 860-913| 116-125        | 120-137            | 886-977 |
  | N17, all            | 7    | 68    | 286    | 72             | 71                 | 352   |
  | N19, all            | 28   | 296   | 314    | 314            | 314                | 627   |
  - The time goes into the dedupe loop (owner lookup + last rank sent to) on the original `owners_` array only: the
    same loop on a fresh copy of the same data on the same rank is fast (7.5x less). Not the data, not the rank, not
    the node: the memory backing that one array. The owner lookup alone stays fast on it (independent loads overlap
    their misses; unexplained in detail)
  - Iteration 1 / N18: slow <=> `owners_` in [heap] (all slow ranks, no normal one), whose VMA is 21% huge pages vs
    96% for the normal ranks' mmapped arrays. Its 33.16 MB sit just below glibc's 32 MiB maximum mmap threshold, so
    heap or mmap depends on the rank's allocation history (seed 1: other ranks, also >= ~1048). N17 (16.6 MB, always
    heap): EVERY rank pays 4x in the loop (build 352 ms, more than N18's 306 at half the size); N19 (66 MB, always
    mmap, 97% huge): no rank. So the vertex ids >= 2^28 were a coincidence of the allocation history
  - A copy in a 2 MiB aligned MADV_HUGEPAGE buffer is fast everywhere. Open: why the `straggler` variant
    MALLOC_MMAP_THRESHOLD_=1 MiB (owners_ from mmap) did not help; check that it took effect (`memory.*` adiak)
  - Fix candidates: (a) large randomly accessed arrays (owners_, also the ghost arrays) from a huge-page allocator
    (2 MiB aligned + madvise(MADV_HUGEPAGE)); (b) no `owners_` at all: owner of ghost index i = upper_bound over the
    per-owner ghost offsets (p + 1 entries, cache resident), saves 4 B per ghost
  - Done (b), 2026-11-24: `VertexNumbering` keeps the per-owner ghost offsets instead of `owners_`; the pattern build
    takes recv counts / displacements from them. std::upper_bound was 13x an array lookup at p = 1536 (random owners
    mispredict its branches; laptop benchmark 8.4 M lookups: array 16 ms, upper_bound 207, branchless 52, bucket
    table + scan 45), so the search is branchless. Locally (4 ranks, permuted RGG 2^20) the dedupe loop costs 50 ms
    vs 45 ms with the array, same triangle count. Next: rerun `straggler-debug` (search vs array vs huge-page array
    at 1536)
  - Rerun `straggler-debug_rerun_26_11_24` (4c7789c): the straggler is gone. Pattern build max/median 1.10-1.13 on
    all inputs (was 3.4x on N18: max 1044-1049 ms vs median 306-315). Medians: N18 301 ms (was 306-315), N17 149 ms
    (was 352: every rank paid the array there), N19 609-612 ms (was 627). Triangle counts identical to the first run
    (all 8 runs). The search alone costs more than the array in the isolated dedupe stage (272 vs 155 ms at N18), but
    the real build is not slower: the search is compute and overlaps the build's other memory accesses, the array
    read was latency. The debug's own fresh owner array still turns slow on 51-246 ranks (max ~880 ms), so the
    backing effect stays for any large randomly read array in that size band (watch the ghost degree arrays)
- Global phase scaling, 2026-11-24: the growth with p is the message count (legacy queue: one global threshold,
  flush-all), not the work; plan for the BriefKAsten migration in doc/briefkasten-migration.md

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
- Superseded 2026-11-19 (see "Variants"): pre-CETRIC removed. Was: keep `--algorithm cetric` (pre-CETRIC, `run_cetric`). Not in the paper suites, but it orients cut edges outward
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

- [x] Primary load balancing moves the vertices with `dstl::redistribute` (decided 2026-11-18, `68bf08b`): the cost split is
      monotone in global-ID order, so the node and edge arrays are one redistribute each, with the receive counts
      from one `MPI_Reduce_scatter_block` (replaces `GraphCommunicator::relocate` and the hand-written NBX
      `sparse_all_to_all` with its size messages). Identical triangle counts on the 70 reference runs and 54 extra
      DPD / IDPD / N runs on 3 / 5 / 8 ranks; only the `redistribution` region's message counters change (no size
      messages, `received_messages` not reported). Needs the dstl fix for empty windows inside a range (a rank
      receiving nothing between two that do, e.g. after a hub whose cost exceeds C/p: stray zero-length message;
      a rank sending nothing: hang) and 64-bit displacements (more than 2^31 elements in total, e.g. the edges at 768
      cores), kamping-v2 PR #126, katric pinned to it. The DPD hang on graphs with fewer vertices than ranks was in the
      cost function (gone with stage 2)

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
- [x] Local indexers: not adopted. Local index = global id - first (arithmetic, in `VertexLayer`); the contracted graph
      maps a local vertex to its row with a dense array instead of `SparseIndexer`'s hash map
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
      owners sorted by global id, partition table; `CsrGraph`: rows over handles, optional ghosts' out-edges and rows in
      global ids), `graph/build.h` (input graph, `exchange_ghost_values` on push_to_ghosts, `orient`), `phases/`
      (preprocessing with the cost functions, local and global kernels, drivers). Identical counters on the 124
      reference runs. Notes:
      - the input graph is built inside the run (region `input_graph`, replacing `ghost_ranks`): one sort of the cut
        edges by (head, position) gives the ghosts and the handles. The old graph was built in the untimed `reset`
      - the global phase works on a copy of each out-row in global ids (`global_rows` in preprocessing): merging a
        message against handles needed a ghost-id lookup per step (1.65x slower message handling on permuted RGG).
        Oriented graph = m/2 handles + m/2 global ids, the old graph held m entries
        (renamed 2026-11-18 with the CSR naming: `first_out`/`head`, `neighbors_as_global_ids`; the region
        `global_rows` is `neighbors_as_global_ids` from the commit after `1145b8d` on (`global_neighbors` in `1145b8d`
        itself); "ghost rows" are the ghosts' out-edges)
      - laptop, 8 ranks, 2^18 vertices / 2^22 edges: permuted RGG 0.49 -> 0.41 s per iteration (preprocessing 0.16 ->
        0.06 s, global phase 0.29 -> 0.25 s, building 0.085 s), RHG 0.094 -> 0.082 s
      - the DPD hang on graphs with fewer vertices than ranks is gone for DiTriC (the old cost function exchanged the
        degrees a second time on ranks without ghosts, a collective only those ranks entered)
- [x] Stage 2, the swap, as a pipeline of immutable graphs built from KaCCv2's building blocks (decided 2026-11-13;
      adopt KaCCv2's interface and layering, not `SplitDistributedCSRGraph` or `FilteredGraph`):
      - input graph: what KaGen and the load balancer produce (`LocalGraphView`, global IDs), unchanged
      - oriented graph: `VertexPartition` + `DenseIndexer` for the local vertices; a ghost table (sorted ghost global
        IDs + owner per ghost, from `get_rank` once per ghost; global ID -> slot by binary search, no hash map: only
        needed while building and when depositing exchanged data); an out-CSR over compact IDs (katric's own builder:
        each undirected edge kept at its smaller endpoint in (degree, global ID)); per-ghost degrees from
        `push_to_ghosts` deposited into a `std::vector` indexed by `c - n`; for CETRIC the ghosts' out-edges (each ghost's
        out-edges into local vertices) from KaCCv2's `build_ghost_reverse_index`, adjusted to index by `c - n`
        instead of its hash slot map and to take an edge filter (the ghost -> local orientation); its targets come out
        sorted by local index, the order the clipping needs. No in-edges kept (only the ghosts' out-edges needed them)
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
      Done 2026-11-18: CETRIC and pre-CETRIC moved over, the old graph code deleted (`DistributedGraph`,
      `AuxiliaryNodeData`, `cost_function.h`, `preprocessing.h`, `contraction.h`, `execution.h`, `counters/`,
      `graph_communicator.h`, `RankEncodedNodeId`; -3000 lines). Identical counters and triangle counts on the 124
      reference runs; tests ported (input graph, degree exchange over all three transports, both orientations with
      ghosts' out-edges). How it differs from the design above:
      - no copy of `build_ghost_reverse_index`: the ghosts' out-edges come out of the orientation pass (count, then fill,
        indexed by c - n), sorted because the local vertices are visited in order. They replace the ghost expansion
      - `ContractedGraph` (`graph/graph.h`): rows for the vertices with cut out-edges, a dense row index per local
        vertex; pre-CETRIC orients it after its degree exchange (`preprocessing_global`, no `sorting` region: filtering
        keeps rows sorted)
      - plain member functions (`neighbors`, `is_owned`, `owner`, `global_id`, ...) instead of KaCCv2's concept
        overloads; `push_to_ghosts` works on both graph types
      - region changes: `input_graph` replaces `ghost_ranks` (and the graph construction formerly in `reset`), CETRIC's
        `ghost_expansion` is gone (its work is in `orientation`), DiTriC has `global_rows`
      - laptop, 8 ranks, 2^18 / 2^22, per iteration: CETRIC permuted RGG 0.76 -> 0.52 s (preprocessing 0.25 -> 0.08,
        local phase 0.20 -> 0.10), RHG 0.111 -> 0.081 s; pre-CETRIC permuted RGG 0.57 -> 0.46 s
        (`preprocessing_global` 0.16 -> 0.04)
- [x] Stage 2b (decided 2026-11-20, done 2026-11-20 in `6912799`, `a3b68a2`, `d1d36e8`; identical counters, region
      paths and triangle counts on the 124 reference runs, laptop timings within noise). How it differs from the design
      below: the global ids are not only the ghost heads. `Graph::to_global_ids(cut_edges_only)` replaces the handle
      neighborhoods by global ids after the local phase (DiTriC: all of N+(v), CETRIC: the cut edges) and frees the
      handle arrays, so the memory is that of the former `GlobalIdGraph` and the receiver does one merge per u. The
      ghost-heads-only variant (local heads compared by arithmetic) gave identical counters and CETRIC as fast, but
      DiTriC's global phase was +36..40% slower on GNM / permuted RGG: a received u touched four per-vertex arrays and
      two data ranges instead of one contiguous neighborhood. Regions `global_ids` / `contraction` kept. Also done:
      `--communication-policy` as a run-time enum (`c738be1`). Not done from the layout: `ghost_exchange.h` and the
      messaging headers unchanged (step-6 review). The original design:
      one `Graph` from orientation through the
      global phase; global ids only for the ghost heads.
      - Why: since `6e75890` every local vertex's sorted neighborhood is split into `local_heads` / `ghost_heads`.
        The global phase needs global ids only where a merge would otherwise look up a ghost id per step (the measured
        1.65x): the receiver's ghost heads. Everything else is cheap from handles: a sender writes its message as
        [ghosts below][local heads as first + index][ghosts above] (ascending); a received entry inside the rank's
        range compares with a local head by arithmetic (`local_heads(u) | transform(to global id)`, which the range
        interface of `intersection` takes, `2f5a95a`)
      - Graph: a third optional part next to the ghosts' out-edges and the sorted split, the ghost heads as global ids
        (`AdjacencyArray<VertexId>`: n + 1 offsets, one entry per cut edge; what `contract_to_global_ids` builds today),
        accessor `ghost_head_ids(v)` with a `KASSERT` that it is built. Building it needs the sorted split (ghost heads
        are already in ascending global-id order: one `ghost_ids_` read per cut edge) and drops the ghosts' out-edges
      - Replaces `GlobalIdGraph`, `to_global_ids` and `contract_to_global_ids`; the `global_ids` / `contraction`
        regions become one. DiTriC no longer allocates 8 B per out-edge (now 8 B per cut edge); fits stage 3 (only cut
        edges as 64-bit ids)
      - Global phase on `Graph const&`: the sender iterates only the ghost heads for destinations (no `is_local_id` per
        entry). The receiver splits each message once against the rank's range (two binary searches), then per local
        u: outside part against `ghost_head_ids(u)` (both variants), inside part against `local_heads(u)` (DiTriC
        only). DiTriC vs. CETRIC differ only in sending the local heads and intersecting the inside part, decided by
        `conf.contraction` (an algorithm choice; the graph no longer says once the ghosts' out-edges are dropped)
      - The steps that change the graph become in-place member functions of `Graph`: `orient(key, ghost_out_edges)`,
        `sort_neighborhoods()`, the ghost-head-ids step. The hooks that exist only for `build.h` (`mutable_neighbors`,
        `truncate_neighborhoods`, `release`) become private, so "neighborhoods do not change after the sort" is
        enforced by the class. The key, load balancing and the variant stay outside (`orient` takes the key).
        `build_input_graph` can become a constructor / `from_kagen`; `to_kagen_graph` stays free. Everything stays
        in `graph.h` (user, 2026-11-20)
      - Accepted: which optional parts are built is checked by assertions only (one type, not one per stage; the
        driver shows the order); the oriented graph lives through the global phase (still less memory for DiTriC)
      - Check: identical messages and triangle counts; `intersection_size_global` changes by design (counted per
        part). Measure DiTriC's conversion region vs. `global_ids`, `handle_ns` (split per message, transform per
        u), peak memory on RMAT. `feat/message-trimming` gets easier (degrees and handles available in the global
        phase)
      - Phase functions (user, 2026-11-20: wrappers that only add a region are indirection): one function per phase,
        named after its Caliper region, which opens the region and reports its counters itself: `local::run` ->
        `local_phase` (`local.h`, reports `local_triangles`), `global::run` -> `global_phase` (`global.h`, reports
        `type3_triangles`); the `local::` / `global::` namespaces with `run` go. A step that is a single call gets a
        scoped block at its call site instead of a wrapper: `input_graph` (block in `count_triangles`),
        `ghost_degrees` (block in `preprocessing`), the ghost-head-ids step (replacing `global_ids` / `contraction`).
        `Graph` members open no regions (regions belong to the algorithm layer, as around `orient` / `sort_neighborhoods`
        today). `drivers.h` keeps only `count_triangles`. Region paths stay as they are (eval and earlier cluster data),
        except the one region replacing `global_ids` / `contraction`
      - Undirected degrees in the graph (user, 2026-11-20): `undirected_degree(v)` for local vertices and ghosts, valid
        from the exchange on (assertion, like the other optional parts). Not `degree(v)`: after `orient` a local
        vertex's `degree` is its out-degree. Local vertices: recorded by `orient` before it truncates (from `xadj`);
        ghosts: the exchange, a member `exchange_ghost_degrees(transport)` (collective; region `degree_exchange` as a
        block at the call site). `degree_key(G)` reads it for every vertex (no separate array, no local/ghost case);
        `orient` keeps taking the key (other orders stay possible). Load balancing may use the same member on the input
        graph; IDPD's out-degree exchange stays in `vertex_costs`. Gives `feat/message-trimming` the entries' keys
        (`(undirected_degree, global id)`) in the global phase. Cost: (n + g) x 8 B for the run
      - Header layout (one header per phase: phase function, its counters, its helpers):
        `graph/graph.h` (`Graph`, numbering, `AdjacencyArray`, the members incl. `from_kagen`; free `to_kagen_graph`;
        `build.h` gone), `ghost_exchange.h` (`push_to_ghosts`), `phases/load_balancing.h` (`primary_load_balancing`,
        `vertex_costs`, the redistribution; from `load_balancing.h` and `preprocessing.h`), `phases/preprocessing.h`
        (`preprocessing(Graph&, conf)`: degree exchange, key, orient, sort), `phases/local.h` (`local_phase`,
        `intersect_edge`, `LocalPhaseCounters`), `phases/global.h` (`global_phase`, `process_message`, `make_message`,
        `owner_from`, `Merger`/`Splitter`, `GlobalPhaseCounters`, `get_threshold`), `intersection.h`,
        `triangle_counter.h`, `katric.h` (`count_triangles`, from `drivers.h`). `phase_counters.h` dissolves.
        `get_threshold` stays called in the driver after load balancing and before orientation (`local-edges` counts
        the undirected edges). The small messaging headers (`communication_policy.h`, `message_tags.h`,
        `message_statistics.h`, `indirect_message_queue.h`, `comm_utils.h`) are decided in the step-6 review;
        `algorithms.h`, `input.h`, `util.h`, `datastructures/span.h` checked then or at the end. Driver afterwards:
        input graph (block) -> `primary_load_balancing` (returns a new graph, vertices move) -> `get_threshold` ->
        `preprocessing(G)` -> `local_phase` -> ghost-head ids (block) -> `global_phase` -> `reduce`
- [ ] Stage 3: 32-bit compact IDs (n + g < 2^32 per rank) — measure the local phase
- Fixed by stage 2 (pre-existing issue): DPD load balancing on a graph with fewer vertices than ranks
  (`examples/triangle.metis` on 4 ranks) hung
- [x] (stage 2) Compact IDs in the adjacency, built once after `find_ghosts`; per-ghost arrays for global ID, rank and degree.
      In KaCCv2's concepts this is a neighbor handle (`neighbor_t` != `vertex_id_t`, projected by
      `neighbor_target`), so the indexers never see compact IDs
- [x] (stage 2) Routing, local tests, `id_outward` and `local_prefix_end` on compact IDs (array lookup / range test);
      `RankEncodedNodeId`, `find_ghost_ranks` (with `gather_PE_ranges` / `get_PE_from_node_ranges`),
      `--binary-rank-search` and `AuxiliaryNodeData` removed
- [x] (stage 2) Messages carry global IDs. The numbering preserves the global order, so a receiver merges the received
      list against `N+(u)` mapped through `to_global` (monotone) without translating the received IDs. Done with a
      copy of the rows in global ids instead (the mapping cost a ghost-id lookup per merge step)
- [x] (stage 2) Contraction (`compact()`, today a `SparseNodeIndexer`) and the cetric-x ghost CSR on the new IDs
- [x] Remove the `std::set<RankEncodedNodeId> ghosts` filter in `run_distributed`: every caller passed an empty set
      (removed in step 2, `5f5a54d`)
- [ ] Compare against `pre-refactor` in `phases`: preprocessing (orientation, degree exchange) and `send_ns`
- [x] (2026-11-18, superseded) A communication context next to the graph's communicator (`G.comm()`) for per-communicator state, e.g. the
      degree exchange's `grid_comm_cache`, which today reaches the callers through a non-owning pointer in `Config`
      (`Config::grid_comms`; a `Config` without it silently splits the communicator on every grid exchange). Decided
      2026-11-13 to keep the caller-owned cache over a static keyval and move it here when the graph's ID and
      partition layer is reworked. Done instead: the vertex layer builds the grid
      communicator on its first grid exchange and frees it with the graph, so every run pays for its split (a cache
      across iterations made only the first iteration pay, which the median over iterations hides). `grid_comm_cache`,
      `Config::grid_comms` and `GhostExchange` are gone; push_to_ghosts takes the transport only. With load balancing
      the rebalanced graph builds a second one

KaCCv2's graph class itself (`SplitDistributedCSRGraph`, CSR plus split-vertex replicas) is not adopted: katric
needs the out/in split per vertex after orientation, the ghost CSR and contraction, so it keeps its own oriented
graph on top of the shared partition/indexer layer.

### 5. Afterwards (not part of this plan)

The algorithm ideas (models in [model/](model/)):

1. Compute at the tail + one rebalancing step (cost-based, DPD/IDPD): the growing imbalance on skewed inputs
2. Forward-style message trimming: send volume of the global phase. Implemented on `feat/message-trimming`
   (2026-11-18), `--message-trimming` (off by default): for each destination rank r of N+(v), the message holds only
   the entries from r's first receiver on in orientation order (key = (degree, global id), kept with the oriented
   graph as `Orientation`); no message if nothing follows that receiver. The keys of a row are computed once per sender
   and the message is a filtered copy of the row in global ids. Verified on 216 runs (all algorithms, direct / grid /
   DPD, 2-8 ranks); identical counters with the option off. Laptop, 8 ranks: global-phase send volume x0.75 on GNM and
   permuted RGG, x0.85 on RMAT, x0.62 on com-amazon, x0.23 (DiTriC) / x0.57 (CETRIC) on RHG; messages x0.64-0.85;
   sending 1.1-1.5x slower (a filtered copy per destination instead of a memcpy), handling unchanged or slightly
   faster. Shared memory hides the saving; suite `trimming` (48-1536 cores) measures it on SuperMUC-NG. The model
   ([model/trim.py](model/trim.py)) said -64% on com-amazon counting only neighbor entries; the measured volume also
   counts the sender id and separator per message. Merged into `chore/modernize` 2026-11-20 (port `5a7b664`): the keys come from
   `Graph::undirected_degree`, kept next to the global ids by `to_global_ids(…, with_degrees)` (only with the flag,
   8 B per entry); each destination's message is built in one reused buffer and posted as a span. Same triangle counts
   with and without on 144 configurations; flag off: identical counters. Laptop, 8 ranks: send volume as above (RHG
   x0.13 / x0.36 at 2^18 / 2^22), `send_ns` x1.2-3.5 (one filtered scan of N+(v) per destination). Suites still use the
   old option names (`algorithm`, `communication-policy: new`), to be switched with the compatibility options
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
