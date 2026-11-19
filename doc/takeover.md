# Takeover note (2026-11-18, end of day)

Where things stand, for picking up tomorrow. The long-term record is [refactoring-plan.md](refactoring-plan.md); this
note is the short version plus the state of the code review.

## Branches

- `chore/modernize` @ `15cf444`. **11 commits not pushed** (`b0d8e51..15cf444`, all from today's review). Every one is
  checked: identical counters and triangle counts on the 124 reference runs, ctest passes.
- `feat/message-trimming` (pushed, `de243d7` = `8f66911` + merge of `chore/modernize` up to `d86687d`): idea 2,
  `--message-trimming`. **Needs adapting before the next merge** (see below).

## Since this note (2026-11-19)

- `1ae4a5f`: one algorithm, KaTric, `--contraction` (CETRIC) instead of `--algorithm`; pre-CETRIC removed (plan,
  section "Variants")
- the commit after it: namespace, include directory, CMake targets and options renamed `cetric` -> `katric`
  (`KATRIC_CXX_FLAGS` etc.; build directories configured with the old `CETRIC_*` options need them set again, e.g.
  by configuring with the preset)

## Review state

Reviewed and approved:

- everything before stage 2 ("stage 1"): `ghost_exchange.h`, `vertex_partition.h`, `load_balancing.h`, the harness,
  the leftover cleanup
- in stage 2 so far: the CSR vocabulary and `graph.h`'s layer (`VertexLayer` ownership, `comm()`), discussed below

Changes made during the review (all on `chore/modernize`, not pushed):

| Commit | What |
|---|---|
| `b0d8e51` | `push_to_ghosts` infers the message type through a type pool; dense exchange as `auto_counts \| auto_displs \| resize_v` |
| `7dd0ba6` | grid communicator per run, owned by `VertexLayer` (a cache across iterations made only iteration 0 pay) |
| `aeee3ca`, `10c6ab1` | input is a `kagen::Graph`; `LocalGraphView`/`NodeInfo` gone; load balancing: scan + bcast, `MPI_Reduce_scatter_block` kept |
| `e365b1b` | leftovers removed (`CommunicationUtility`, `timer.h`, most of `util.h`); `MessageTag` passed as enum |
| `1145b8d`, `dc19326`, `c1aa8ca` | CSR naming `xadj_`/`adjncy_`; `comm()` returns a `kamping::v2::comm_view`, no `rank()`/`size()` on graphs |
| `05e05f2` | `VertexLayer` documents its shared ownership and collective destruction |
| `9c64803` | global phase on `GlobalIdGraph`: oriented graph converted to global ids **in place** after the local phase; contraction in place; send loop finds destinations as runs (`VertexPartition::get_rank_from`), `last_proc` gone; `ContractedGraph` gone |
| `15cf444` | runs read the input (`kagen::Graph const&`), no copy per iteration in the harness |

Decisions taken (with reasons, so they are not re-opened by accident):

- graphs do **not** inherit from `kagen::Graph` (our `adjncy` holds handles, oriented, contracted, 32-bit handles later)
- `VertexLayer` was a `shared_ptr` (graphs of a run shared it); since 2026-11-19 each graph holds it by value and
  each step consumes its input graph and moves the layer on (`orient` takes `Graph&&`): the only overlap of two graphs
  was the orientation, and nothing holds the layer's address
- `GlobalIdGraph` stays a separate type (its neighbors are a different number space than `Graph`'s)
- no concat view for translating during the merge (the per-entry ghost-id lookup was the measured 1.65x cost);
  in-place conversion instead
- send loop: binary search over the partition table from the current owner; galloping/O(1) not worth it

Still to review in stage 2, in the order of the algorithm's flow (decided 2026-11-19): everything hangs off
`count_triangles` (`phases/drivers.h`), and each step reads what the previous one produced. Paths below
`include/katric/`; "(reviewed)" marks what was approved already.

0. (reviewed 2026-11-19) Entry: `apps/algorithm.cpp` (`make_algorithm`, `ingest`, `run`), `phases/drivers.h`
   `count_triangles`. Kept: `count_triangles` takes the input by `const&` (by value would cost a copy per iteration
   in the harness; KaGen's `SInt` vs. our `EdgeId`/`Handle` rules out reusing its buffers)
1. Input graph, `phases/preprocessing.h` `input_graph` (reviewed 2026-11-19, as is `get_threshold`): `graph/graph.h` (reviewed 2026-11-19, all of it: `VertexLayer`, `Graph`,
   `AdjacencyArray` from `2bfa430`, layer by value from `ee075c2`); `graph/build.h` `build_input_graph` (approved 2026-11-19; cut-edge sort, handles; reworked in `489746b`: partition table is `dstl::distribution`, `VertexPartition` gone, handles written through the layer; `VertexLayer` split into `LocalNumbering` (distribution, local handles; exists before the ghosts) and `VertexNumbering` (built from it + ghost ids), `G.layer()` -> `G.numbering()`, one pass again);
   `phase_counters.h` `get_threshold`. Changed during the review of `Graph` (2026-11-19): `VertexHandle` is an
   `enum class` without arithmetic (the layer converts: `to_local_index` / `to_ghost_index`, `local_handle` /
   `ghost_handle`, `first_ghost`), consistent member names (`num_local_vertices`, `num_ghosts`, `num_local_edges`,
   `local_vertices`, `is_local` / `is_ghost`, `is_local_id`, `local_range`), `NodeId` -> `VertexId` in `graph.h`,
   `EdgeId` -> `std::size_t`, `graph_definitions.h` gone (`Triangle` in `triangle_counter.h`, `Edge` in the tests),
   tlx dropped. `to_global_ids` / `contract_to_global_ids` now write a new array (a handle array cannot hold global
   ids): `global_ids` 0.6-1.8 ms per iteration on the laptop either way, +8 B per out-edge transiently
2. Primary load balancing, `preprocessing.h` `primary_load_balancing` (approved 2026-11-19, with `get_threshold` after it and
   `to_kagen_graph`; open: `exchange_ghost_values`): `vertex_costs` (approved 2026-11-19; N / DPD / IDPD, two ghost
   exchanges), `build.h` `exchange_ghost_values` -> `ghost_exchange.h` `push_to_ghosts` (reviewed), `build.h`
   `to_kagen_graph` -> `load_balancing.h` `run` (reviewed) -> `build_input_graph` again. Moves undirected
   neighborhoods (open point, plan section "Variants"). Changed in review (`53fef74`): threshold computed after
   the load balancing (only the global phase's message counts of load-balanced runs differ from `refcheck/ref`), step
   consumes its graph, orientation decided once per edge, regions split, `CostFunction` enum
3. Preprocessing, `preprocessing.h` `preprocessing`: `ghost_degrees`, `degree_key`, `orient_and_sort` -> `build.h`
   `orient` (the ghosts' out-edges with `--contraction`); `Graph::sort_neighborhoods`, `ghost_neighbors` / `ghost_degree`
4. Local phase, `drivers.h` `local_phase`: `phases/local.h` `run<with_ghosts>`, `neighbors_with_ghosts`, the edge
   kernel; `intersection.h`, `triangle_counter.h`, `phase_counters.h`
5. Into global ids, one of two: without contraction `preprocessing.h` `global_ids` -> `build.h` `to_global_ids`
   (translate + rotate); with contraction `preprocessing.h` `contraction` -> `report_contraction` -> `build.h`
   `contract_to_global_ids` (in-place compaction). Both give `graph.h` `GlobalIdGraph` (`kept` / `is_kept`,
   neighbors in global ids)
6. Global phase, `drivers.h` `global_phase`: `phases/global.h` `run` (send loop: owner runs via `owner_from` over `dstl::distribution`),
   `make_message`, the handler (`Merger` / `Splitter`); `communication_policy.h`, `message_tags.h`
7. `triangle_counter.h` `reduce`

## Open items

- **Push** `chore/modernize` (11 commits) and merge it into `feat/message-trimming`. The trimming code uses
  handle-based keys (`Orientation::degree(handle)`, `neighbors_as_global_ids`) that no longer exist after `9c64803`:
  it must compute the keys before the in-place conversion (e.g. keep each entry's degree next to it on the trimming
  path) — expect merge conflicts in `global.h`, `graph.h`, `preprocessing.h`
- **Memory** (measured 2026-11-18 evening, laptop, 4 ranks, 2^18 vertices / 2^22 edges in total, max RSS after each
  phase via a temporary getrusage patch; Caliper's memory services need GOTCHA or /proc, i.e. Linux): the peak is set
  by KaGen's generation, not by katric. Baseline 20 MB, graph 17 MB per rank; after generation + ingest the busiest
  rank is at 123 MB (RGG), 207 MB (permuted RGG), 120 MB (GNM), 257 MB (RMAT), i.e. KaGen needs 6-14x the graph
  transiently. katric adds +18 / +17 / +67 / +39 MB on top (GNM: input graph + handles + the 16-byte sorted cut pairs;
  DiTriC's message buffers up to 20 MB). So the RMAT jobs killed at 1536/3072 cores may die in KaGen's generation
  already. To check on SuperMUC: build Caliper with GOTCHA on Linux (enables `mem.highwatermark`), or add the
  getrusage marks properly (as a counter). katric's own lever: the cut pairs (16 B per cut edge) in
  `build_input_graph`
- **KaGen internals** (read 2026-11-18): edge-list generators (RGG, RMAT/Graph500, ...) are converted at the end
  (`EdgeListOnlyGenerator::FinalizeCSR`): the edge list (16 B per directed edge) and the CSR (8 B) are alive together.
  RMAT's `RedistributeEdges` (sort + dedup, round-robin remap) holds the edge list, per-rank send vectors of pairs, a
  flat send buffer and the receive buffer at once (several 16 B copies of the busiest rank's edges). `permute` adds
  send/receive buffers and a second CSR. Graph500 also offers `GraphRedistribution::BALANCE_EDGES`
  (`RedistributeEdgesBalanced`), which could balance RMAT's edges at generation time — worth checking against IDPD
- **SuperMUC**: `trimming` suite submitted from the branch; `lb-cost` `rerun2` (48/192/768) queued; old job files
  still pass `--global-synchronization` (accepted and ignored since `8d1eca2`) — remove the flag once they have run;
  use `--copy-binary` for new submissions
- **kamping-v2 PR #126** (dstl::redistribute fixes): katric pinned to `7634a78`; re-pin to `main` once merged
- follow-up: `VertexNumbering::owners_` could go too (the run trick works for `push_to_ghosts` as well)

Results of 2026-11-18's runs are analyzed in the plan ("Results 2026-11-19"). They confirm the order below;
communication overhead (polling/progress, trimming, grid) waits for BriefKAsten. `5618dc5` adds `poll_ns` and
`termination_handle_ns` to the global phase.

Next steps after the review (plan, step 0 "Next steps"): (1) rebalance only when the IDPD cost's max/avg exceeds a
threshold, (2) bucketed input-graph build + linear-merge ghost-value deposit, (3) idea 1 vs. DiTriC + IDPD on RMAT;
then stage 3 (32-bit handles).

## Checking a change

`build-release/refcheck/check.sh <name>` builds nothing; it copies `build-release/apps/katric`, runs the 70 reference
runs + 54 load-balancing runs, and compares with the reference (binary from `466103e`). Build first with
`cmake --build build-release -j10`; ctest in `build/` (Debug, assertions on).
