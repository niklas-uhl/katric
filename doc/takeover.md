# Takeover note (2026-11-18, end of day)

Where things stand, for picking up tomorrow. The long-term record is [refactoring-plan.md](refactoring-plan.md); this
note is the short version plus the state of the code review.

## Branches

- `chore/modernize` @ `15cf444`. **11 commits not pushed** (`b0d8e51..15cf444`, all from today's review). Every one is
  checked: identical counters and triangle counts on the 124 reference runs, ctest passes.
- `feat/message-trimming` (pushed, `de243d7` = `8f66911` + merge of `chore/modernize` up to `d86687d`): idea 2,
  `--message-trimming`. **Needs adapting before the next merge** (see below).

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
- `VertexLayer` stays a `shared_ptr` (graphs of a run share it, lifetimes do not nest)
- `GlobalIdGraph` stays a separate type (its neighbors are a different number space than `CsrGraph`'s)
- no concat view for translating during the merge (the per-entry ghost-id lookup was the measured 1.65x cost);
  in-place conversion instead
- send loop: binary search over the partition table from the current owner; galloping/O(1) not worth it

Still to review in stage 2 (current files, in data-flow order):

1. `include/cetric/graph/graph.h`: `CsrGraph` (`release()`, ghost out-edges), `GlobalIdGraph` (new: `is_kept`,
   contracted vs. full)
2. `include/cetric/graph/build.h`: `build_input_graph` (cut-edge sort), `exchange_ghost_values`, `orient`,
   `to_global_ids` (translate + rotate), `contract_to_global_ids` (in-place compaction, `keep` filter)
3. `include/cetric/phases/preprocessing.h`: `vertex_costs`, per-algorithm preprocessing, `contraction`,
   `contraction_and_preprocessing_global` (pre-CETRIC: degree exchange on the oriented graph, then filter)
4. `include/cetric/phases/local.h`, `phases/global.h` (owner runs in the send loop), `phases/drivers.h`
5. small: `phase_counters.h`, `triangle_counter.h`, `intersection.h`

## Open items

- **Push** `chore/modernize` (11 commits) and merge it into `feat/message-trimming`. The trimming code uses
  handle-based keys (`Orientation::degree(handle)`, `neighbors_as_global_ids`) that no longer exist after `9c64803`:
  it must compute the keys before the in-place conversion (e.g. keep each entry's degree next to it on the trimming
  path) — expect merge conflicts in `global.h`, `graph.h`, `preprocessing.h`
- **Memory**: the in-place conversion and the dropped input copy lower the summed RSS only ~3-4% each and the busiest
  rank hardly at all (laptop, permuted RGG, 4 ranks: ~230 MB). The peak is elsewhere — suspects: KaGen generation,
  and `build_input_graph`'s sorted (head, position) pairs (16 bytes per cut edge). Find it with Caliper's memory
  high-water mark (`--caliper-config "spot(mem.highwatermark),..."`, untested) before optimizing further. Relevant for
  the RMAT runs killed at 1536/3072 cores
- **SuperMUC**: `trimming` suite submitted from the branch; `lb-cost` `rerun2` (48/192/768) queued; old job files
  still pass `--global-synchronization` (accepted and ignored since `8d1eca2`) — remove the flag once they have run;
  use `--copy-binary` for new submissions
- **kamping-v2 PR #126** (dstl::redistribute fixes): katric pinned to `7634a78`; re-pin to `main` once merged
- follow-up: `VertexLayer::owners_` could go too (the run trick works for `push_to_ghosts` as well)

Next steps after the review (plan, step 0 "Next steps"): (1) rebalance only when the IDPD cost's max/avg exceeds a
threshold, (2) bucketed input-graph build + linear-merge ghost-value deposit, (3) idea 1 vs. DiTriC + IDPD on RMAT;
then stage 3 (32-bit handles).

## Checking a change

`build-release/refcheck/check.sh <name>` builds nothing; it copies `build-release/apps/katric`, runs the 70 reference
runs + 54 load-balancing runs, and compares with the reference (binary from `466103e`). Build first with
`cmake --build build-release -j10`; ctest in `build/` (Debug, assertions on).
