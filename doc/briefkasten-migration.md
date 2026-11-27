# BriefKAsten migration of the global phase (plan, 2026-11-24)

Replaces the legacy message queue (`legacy-message-queue`, `message_queue::make_buffered_queue`) and katric's grid
adapter (`include/katric/indirect_message_queue.h`) in the global phase by BriefKAsten, configured as in KaCCv2:
a capacity per destination buffer instead of one global threshold, a bounded number of outstanding sends
(request slots), credit-based flow control, and BriefKAsten's own grid indirection. The degree exchange is not
affected (dense / sparse / grid all-to-all, no message queue).

## Why

Results 2026-11-24 (doc/refactoring-plan.md; `rebalance_26_11_23`, `baseline_26_11_20`, DiTriC, trimming, no load
balancing): from 192 to 3072 cores the global phase grows 4.1 -> 7.0 s on permuted RGG and 1.1 -> 2.9 s on GNM, while
the work per rank is constant (49.7 M wedge checks, 49 M words sent, handling 1.8 s). Messages per rank double with p
(35 K -> 578 K) and polling grows with them (0.85 -> 2.76 s), sending too (1.24 -> 1.84 s).

Cause: the legacy queue has one global threshold (`--threshold local-nodes` = n words, ~2 MB) and on overflow flushes
all p destination buffers (`buffered_queue.h:39-41` in legacy-message-queue). The traffic of permuted inputs is spread
over all ranks, so about volume / n overflows happen (~188 on permuted RGG) at any p, each sending to every rank:
messages = 188 p, messages shrink to ~85 words at 3072. With a fixed total buffer B any scheme gets messages ~ p V / B
for uniform traffic; the levers are the buffer per destination (memory grows with p) and the fan-out (grid: sqrt p).

A fixed capacity C per destination buffer makes every message a full buffer: messages per rank = V / C at any p
(permuted RGG: ~96 K at C = 4 KiB, ~12 K at C = 32 KiB, vs 35 K-578 K now). Memory: up to p C of partially filled
buffers (12 MB at 3072 with 4 KiB, 96 MB with 32 KiB), bounded at larger scale by the grid (fan-out ~ sqrt p). The
request slots and credits bound the outstanding sends, so polling no longer scans an unbounded set of requests (also
the suspected cause of the HoreKa2 UCX failures on permuted inputs).

The grid already scales this way today (`baseline_26_11_20`, 192 -> 3072: permuted RGG 5.7 -> 6.6 s, GNM 1.4 -> 1.7 s,
46 K messages at 3072) but costs ~1.6 s more than direct at small p (polling 1.5-1.9 s even with few messages).

## Findings in BriefKAsten (read 2026-11-24, `feat/idle-flush` @ 1c5ed3d; KaCCv2 pins 5664f7f)

- **Mergers / splitters.** Shipped: `AppendMerger` / `NoSplitter` (no message boundaries), `SentinelMerger` /
  `SentinelSplitter` (no receiver, so no indirection), `TupleMerger` / `TupleSplitter` (fixed-size tuples, KaCCv2),
  `EnvelopeSerializationMerger` / `Splitter` (variable length: [receiver, size, payload]). The last one has the right
  format but (a) builds a temporary `std::vector` per message before appending it, (b) yields each message as a
  transform view (not contiguous; the intersection takes a `std::span`), (c) reports sender 0. So katric gets its
  own pair (step 2).
- **A message larger than a buffer livelocks.** `post_message_impl` (`buffered_queue.hpp:996-1041`) loops while
  `buffer.size() + message.size() > local threshold`: it flushes the destination's buffer and takes a fresh one. If the
  message alone exceeds the threshold, the fresh buffer is empty, flushing it is a no-op (`flush_buffer_impl` returns
  at once on an empty buffer), and the check fails again, forever; the blocking variant polls in that loop, so the
  rank keeps serving others but never posts. Behind it: the receive buffers are sized to the threshold
  (`reserved_receive_buffer_size`), a larger buffer needs the `LARGE_MESSAGE_TAG` path of `detail/queue.hpp:119-126`,
  which `BufferedMessageQueue` never enables (`allow_large_messages` is only on the inner `MessageQueue`; otherwise it
  throws). Flow control already lets a buffer larger than a credit window pass (`num_oversize_passes`). Found by
  reading; step 1 reproduces it in a test first.
- **It will happen.** A message is [receiver, length, v, N+(v)] (trimmed), 8 B per word. The largest out-degree
  (`max_out_degree`, preprocessing): RGG 45-47, RHG 84-103, RMAT 1071 (48 cores) -> 3186 (768 cores), growing with the
  graph (d+ <= sqrt(2m)). RMAT's largest message is ~25 KB at 768, beyond C = 4 / 16 KiB; real-world hubs are larger.
- **Builds.** BriefKAsten fetches KaMPIng v0.2.0 and kassert v1.0.0 (katric declares KaMPIng v0.3.2 first, so that
  one is used; check it compiles), and needs C++23 or, with `BRIEFKASTEN_USE_CXX23=OFF`, range-v3. katric is C++20
  (`cxx_std_20`).

## Steps

0. **Dependency.** FetchContent BriefKAsten pinned to a SHA (not a branch; not shallow), as KaCCv2 does. Which SHA:
   `feat/idle-flush` @ 1c5ed3d (newest, includes idle flush) or KaCCv2's 5664f7f; the step-1 fix goes on top of it.
   C++23 for katric (GCC 15.1 on SuperMUC-NG) or range-v3 (decision below). Build with the debug and release presets.
   Done 2026-11-24: katric-lib `cxx_std_23` (clang-tidy / clangd flags, README), BriefKAsten fetched at b94940a
   (`fix/oversized-messages`, PR #24 against feat/idle-flush; re-pin once merged) with `BRIEFKASTEN_USE_CXX23`,
   linked as `BriefKAsten::BriefKAsten` next to the legacy `message-queue`. Debug and release build; same triangle
   count as before on permuted RGG 2^20 (4 ranks). Nothing includes BriefKAsten yet (step 2 does).
1. **Oversized messages (BriefKAsten, own branch / PR).**
   - Test first: post one message larger than the local threshold (direct, grid, flow control on / off; exactly the
     threshold, threshold + 1 element, 100x) and check that it arrives; it hangs today.
   - Fix in `post_message_impl`: if the message does not fit an empty buffer, flush the destination's current buffer
     (if not empty), merge the message into a fresh buffer and flush that one at once as its own send. The queue enables
     its large-message path whenever a local threshold is set (LARGE_MESSAGE_TAG, probing receiver that allocates the
     buffer). A relay (grid) that receives such a message reposts it and takes the same path. Counter
     `num_large_messages` (and its elements).
   - Done 2026-11-24, BriefKAsten branch `fix/oversized-messages` @ b94940a (on feat/idle-flush, local, not pushed):
     the hang reproduced (all variants hung), fixed as above (`Config::allow_large_messages`, default on). It also
     needed two flow-control fixes for the grid: the proxied credit sum lost and re-added the overshoot of an
     oversize pass, and the high-water assertion fired on grants to non-proxied peers while a proxy forwarded an
     oversize buffer. `large_message_test` passes direct / grid x flow control on / off on 1-9 ranks; the existing
     tests pass at 4 and 9 ranks. Cost: one extra MPI_Improbe per poll for the large-message tag.
   - Alternative, chunking (post_message consumes what fits, the rest goes into the next buffer): needs a merger
     that can stop mid-message and a katric format change. A chunk of N+(v) is only useful with the targets u on the
     receiving rank, which the receiver finds in the whole message today, so every chunk carries them as a header
     ([v, k, targets..., slice]) and the handler intersects each target's N+(u) with each slice (every w in exactly
     one slice, so no duplicates). More volume (k + 2 words per chunk), N+(u) scanned once per chunk unless
     restricted to the slice's range, and a hub with more targets on one rank than fit a buffer needs the header
     itself split (targets x slices). Gain: every buffer within the credit window and no extra probe. Kept as an
     option if large messages turn out costly (only RMAT / real-world hubs produce them).
   - Earlier note, katric-side chunking: a chunk of N+(v) is only useful at the receiver with the targets u on
     that rank (the receiver finds them in N+(v) today), so every chunk would carry them as a header. More format, more
     volume; only if the library fix is not wanted.
2. **Merger / splitter: BriefKAsten's, fixed there instead of katric's own (2026-11-24).** katric's record format
   is exactly `EnvelopeSerializationMerger` / `Splitter`'s ([receiver, size, payload]); its problems were generic, so
   they are fixed in BriefKAsten PR #25 (`feat/contiguous-envelope-serialization` @ b981e72, stacked on #24): the
   merger appends in place, the splitter yields a `std::span` into the buffer when the message type is the buffer
   type (katric: `std::uint64_t` global ids in a `std::uint64_t` buffer). Test `aggregators_test`; BriefKAsten's MPI
   tests pass at 4 ranks. katric pinned to b981e72. The global phase uses the pair directly (step 3). Original idea:
   **katric merger / splitter** (`include/katric/phases/global.h`, replacing `Merger` / `Splitter`). Buffer of
   `std::uint64_t` (global ids); a record is [receiver, length, v, N+(v)...]: the receiver for the indirection (one
   format for direct and grid; today the sentinel costs one word, the grid adapter two more), the length instead of the
   sentinel (no scan for `message_end`). The merger appends in place (no temporary) and implements
   `estimate_new_buffer_size`; the splitter yields `MessageEnvelope{std::span<VertexId const>, origin, receiver, 0}`.
   Unit tests: round trip, empty / one-element / oversized messages, through the grid adapter.
3. **Port the global phase.** Queue on `G.comm()` (drops the MPI_COMM_WORLD check), built from a
   `briefkasten::Config`; direct, or `IndirectionAdapter<GridIndirection>` for `--communication-policy grid`.
   `post_message_blocking(span, destination, on_message)` in the send loop (never fails), `poll`, `terminate`.
   Timing points unchanged (send / handle / poll / termination, `termination_handle_ns`), so the counters stay
   comparable. Messages and volume from `message_counts()`; new counters: overflows, buffer stalls, send capacity waits,
   parked for credit / capacity, oversize passes, large messages, idle flushes, polls / unproductive polls, peak
   send backlog.
   Done 2026-11-24 (with step 4): `global_phase` runs one send loop / termination (`exchange`) on either queue;
   BriefKAsten from `with_briefkasten_queue` (`--receiver` as a template parameter, grid via
   `IndirectionAdapter{queue, GridIndirectionScheme{comm}}`) on `G.comm()`, `EnvelopeSerializationMerger` /
   `Splitter<VertexId>` (spans into the buffer). `post_message_blocking` may handle messages while it waits; their
   time is in `handle_ns`, not `send_ns`. Counters (`report_briefkasten_counters`, under the grid adapter its
   queue): messages = buffers per hop, `send_volume` = elements flushed per hop incl. the 2-word header,
   `receive_volume` = delivered messages incl. header (relayed ones not counted); plus overflows, buffer_stalls,
   send_capacity_waits, parked_for_credit / _capacity, oversize_passes, large_messages (+ _elements),
   forced_flushes, idle_flushes (+ _elements), polls, unproductive_polls, termination_rounds, peak_send_backlog.
   Laptop, permuted RGG 2^16, 4 ranks, DiTriC: work counters identical to legacy; volume +6 % (header 2 words vs
   sentinel 1); 6178 buffers at 4 KiB vs 540 legacy messages (n-word threshold at 4 ranks is ~130 KiB).
4. **Options.** `--message-queue legacy|briefkasten` for the transition (legacy keeps `--threshold` /
   `--threshold-scale`); for BriefKAsten `--buffer-size` (local threshold, bytes), `--request-slots`,
   `--flow-control` / `--no-flow-control`, `--receiver persistent|probe|posted`, `--poll-skip`. Suites that compare
   with earlier data pin `message-queue: legacy`, as for the defaults on 2026-11-23.
   Done 2026-11-24: defaults `--message-queue legacy` (until step 7, so existing suites need no pin yet),
   `--buffer-size 4096`, `--request-slots 8`, flow control on, `--receiver persistent` (BriefKAsten's defaults
   except the buffer size, KaCCv2's 4 KiB), `--poll-skip 1`. Changed the same day: the queue runs in synchronous
   mode (handling a message creates no new work; a proxy's forwarding is counted as pending), so one `terminate`
   call returns once all ranks are done, draining every buffer, relays included, before each counting round. That
   drain makes the idle flush (which acts only on entering `terminate`) irrelevant, so `--idle-flush` was dropped
   (BriefKAsten's default `all` stays). `--poll-skip n` polls after every n-th sender vertex (`poll_throttled`;
   BriefKAsten's default is 100); the legacy queue polls after every one. ctest configs `ditric-briefkasten` and
   `cetric-briefkasten-grid` (256 B buffers, so large messages occur).
5. **Local validation.** Identical triangle counts on the refcheck inputs (DiTriC / CETRIC, trimming on / off,
   direct / grid, 2-8 ranks), plus a tiny buffer (e.g. 64 B) so every message takes the oversized path, and
   `--no-flow-control`. One small run per variant, not the whole test suite.
   Run 2026-11-24 (4 inputs: hiv, com-amazon, permuted RGG 2^14, RMAT 2^14; 2 / 4 / 9 ranks; DiTriC with trimming,
   CETRIC without; direct / grid; 7 BriefKAsten variants vs legacy). Found two BriefKAsten hangs, both only when
   messages are larger than a buffer:
   - **Credit stall (fixed).** The receiver grants once half the window W has freed up, so with nothing in flight
     the credit stays anywhere in (W/2, W]; `has_credit` passed only buffers > W, so a large-message buffer in
     (credit, W] waited forever (RMAT, direct, 64 B and 256 B buffers). Normal buffers are <= W/8 (8 credit
     buffers); with `num_credit_buffers = 1` normal buffers could hit it too. Fix: pass buffers > W/2 once any
     credit is left. BriefKAsten `fix/oversized-messages` @ b75b61f (pushed, PR #24), with a test (message
     sizes 5 and 7 buffers, hung at 4 / 9 ranks) and a barrier between the tests of `large_message_test` (shared
     communicator and tags; a previous queue's persistent receive could match the next test's message, receiver
     destructor assertion). `feat/contiguous-envelope-serialization` rebased on it @ 0062f64 (PR #25); katric pinned to 0062f64.
     `idle_flush_test` at 9 ranks fails that destructor assertion in ~1/3 of runs with and without the fix
     (same isolation problem: KaTestrophe's listener does not synchronize the ranks between tests,
     kamping-site/KaTestrophe.cmake#7).
   - **Grid without flow control: expected, not a bug.** Proxies forward from inside the receive handler with
     `post_message_blocking`; each waits for a send slot to the other while its own receiver is busy (nested):
     a cyclic wait. Bounded memory plus blocking is only safe with credit-based flow control (a proxy then parks
     instead of blocking); direct mode is safe without it (no handler posts). Showed up only with 64 B buffers
     (every message large, rendezvous sends hold their slot); with a lower eager limit on the cluster, normal
     buffers could hit it too. katric rejects `--communication-policy grid --no-flow-control` with BriefKAsten (`parse_config`).

6. **Cluster suite `briefkasten`.** Permuted RGG N18 and GNM N16 (then permuted RHG, RMAT): legacy direct / grid vs
   BriefKAsten direct / grid x C in {4, 16, 64} KiB, defaults otherwise, 3 iterations; cores set at job creation
   (192-3072 shows the trend). Questions: does the global phase stay flat with p; polling cost per message; which C;
   where grid overtakes direct now.
   Suites written 2026-11-24: `experiments/briefkasten.suite.yaml` (permuted RGG N18, GNM N16, permuted RHG N18;
   no load balancing, trimming; legacy direct / grid at threshold local-nodes vs BriefKAsten direct / grid x
   `--buffer-size` 4096 / 16384 / 65536, at 16384 also `--poll-skip 100`; 3 iterations; `--min-cores 192
   --max-cores 3072`) and
   `briefkasten-rmat.suite.yaml` (the same configurations on RMAT N16 with the default load balancing,
   `--max-cores 768`). Memory at 3072 with 64 KiB: up to ~2 p buffers per rank with flow control, ~384 MiB.
   **Results 2026-11-25** (`eval/data/supermuc/briefkasten_26_11_24`, `briefkasten-rmat_26_11_24`; some 3072 runs
   missing). Global phase, max over ranks, mean of 3 iterations, s (brd / brg = BriefKAsten direct / grid with C,
   led / leg = legacy direct / grid, /s100 = `--poll-skip 100`):

   | input, cores | led | leg | brd4k | brd16k | brd16k/s100 | brd64k | brg4k | brg16k | brg64k |
   |---|---|---|---|---|---|---|---|---|---|
   | RGG perm. 192 | 4.26 | 5.81 | 4.58 | 4.32 | 3.96 | 4.32 | 5.85 | 5.45 | 5.31 |
   | RGG perm. 3072 | 7.66 | 7.82 | 5.33 | 6.12 | 8.39 | 6.79 | 7.72 | 6.46 | 6.50 |
   | GNM 192 | 1.11 | 1.42 | 1.13 | 1.02 | 0.90 | 1.02 | 1.46 | 1.25 | 1.27 |
   | GNM 1536 | 3.05 | 1.49 | 1.38 | 1.29 | 1.06 | 1.41 | 1.62 | 1.48 | 1.47 |
   | RHG perm. 192 | 3.75 | 5.30 | 4.06 | 3.83 | 3.36 | 3.93 | 5.31 | 4.86 | 4.91 |
   | RHG perm. 1536 | 5.25 | 5.77 | 4.67 | 4.95 | 4.31 | 5.61 | 6.18 | 5.41 | 5.54 |
   | RMAT 48 | 6.70 | 11.9 | 6.95 | 10.1 | 10.1 | 9.68 | 7.62 | 8.08 | 7.87 |
   | RMAT 768 | 44.5 | crash | 27.8 | 25.5 | 38.4 | 23.6 | 30.6 | crash | crash |

   - Messages per rank constant in p with BriefKAsten (V / C: RGG ~105 K at 4 KiB, ~26 K at 16 KiB), legacy direct
     doubles with p (35 K -> 578 K on RGG). BriefKAsten direct 4 KiB is the flattest (RGG 4.6 -> 5.3 s from 192 to
     3072, legacy direct 4.3 -> 7.7 s); legacy direct on GNM falls apart from 1536 on (3.1 s vs ~1.1-1.4).
   - Grid is still ~1 s slower than direct up to 3072, mostly polling (proxies poll about twice as long); not
     needed at this scale.
   - 64 KiB direct: handle time max grows with p (RGG 2.0 -> 3.7 s) while the average (~1.9 s) and the work
     (50 M wedge checks) stay constant; 4 KiB stays at 2.0-2.2 s. A few slow ranks, presumably memory pressure of
     the larger pool (up to 2 buffers per peer); per-rank data would tell.
   - `--poll-skip 100` (direct) gains ~10 % up to 1536 (RGG 3.96 vs 4.32, RHG 3.36 vs 3.83), loses at the top:
     RGG 3072 8.39 vs 6.12, RMAT 768 38.4 vs 25.5. Send completions are only noticed on a poll, so buffers park for a
     request slot (RGG 3072: 1 K -> 16 K parked for capacity per rank). In grid no gain: proxies grant credit late
     (parked for credit ~0.1 K -> 22 K per rank).
   - Request slots are the bottleneck on RMAT: almost every buffer parks for capacity (16 KiB at 768: 91 K of 102 K
     buffers per rank) with 8 slots.
   - **Large messages (RMAT only).** At 4 KiB they carry 19 % (48 cores) to 60 % (768) of the volume, ~110 K per
     rank at 768, averaging ~1000 words (~8 KB, just above the buffer), so not hub-sized; at 16 KiB 4 % of the
     volume, at 64 KiB none. 4 KiB is still the best BriefKAsten setting at 48 cores (6.95 s) and 3-18 % behind
     16 / 64 KiB at 384-768 (the busiest rank sends 3.3x more messages: 957 K vs 293 K). The RMAT global phase is
     dominated by the send loop of the busiest rank, ~10 s at 768 with every queue (legacy direct 9.9 s): it sends
     539 M words, 3x the average (the volume imbalance after load balancing, cf. `rmat-volume`), plus imbalanced
     handle time (~11 s). Checked 2026-11-25: `post_trimmed_messages` scans N+(v) once per destination, i.e. the
     sent plus the trimmed entries (~960 M on that rank), at most ~1.8x the copying it has to do anyway, so building
     the messages in one pass would save only the comparisons on the trimmed part; not pursued.
   - Counter bug: `oversize_passes` counts `has_credit` returning true for an oversized buffer, which is re-checked
     while it waits for a request slot (480 K "passes" vs 110 K large messages per rank at 768, 4 KiB). Count at the
     send instead.
   - **Crashes.** RMAT 384 / 768: BriefKAsten grid 16 KiB, 64 KiB, 16 KiB/s100 and legacy grid (768) abort in Intel
     MPI (`intel_transport_recv.h:1267: cma_read_nbytes == size`, the single-copy CMA path for large intra-node
     messages). The legacy grid adaptor too, so not BriefKAsten-specific; seems tied to large relayed buffers
     (4 KiB grid never crashed). Intel support suggested `I_MPI_SHM_CMA=0` for the same assertion (community thread
     1329425), without a root cause: suite `briefkasten-cma-debug`.
   Consequences: default `--message-queue briefkasten --buffer-size 4096 --poll-skip 1` (step 7). BriefKAsten:
   more request slots, `poll_throttled` that does not skip while buffers are parked, count `oversize_passes` at the
   send. katric: nothing queue-specific (the RMAT cost is the busiest rank's send volume).
7. **Default BriefKAsten** after the comparison; the legacy queue (with katric's grid adapter, which it needs) stays
   behind `--message-queue legacy` for reproducing earlier results.
   Done 2026-11-25: default `--message-queue briefkasten` (`--buffer-size 4096`, `--poll-skip 1` were already the
   defaults); every cluster suite except `briefkasten*` pins `message-queue: legacy` (the local smoke / phases
   suites take the new default). ctest: `ditric-legacy` keeps the legacy queue covered.
8. **BriefKAsten tweaks** (2026-11-25). BriefKAsten branch `feat/throttled-poll-and-pass-count` @ 75d4f95 (PR #26,
   on `feat/contiguous-envelope-serialization`; katric pinned to it): `poll_throttled` polls on every call while buffers are parked or
   every send slot is taken (it skipped them, which parked buffers for a slot at the top scale); `oversize_passes`
   counted once per send in `track_send` (was per `has_credit` check; `large_message_test` checks passes <= large
   messages, which failed at 9 ranks with the old count). BriefKAsten tests pass; katric counts unchanged with
   `--poll-skip 100` (direct / grid, RMAT and permuted RGG, 4 / 9 ranks, also 256 B / 1 slot). Suites
   `briefkasten-tuning` (+ `-rmat`): direct 4 KiB x `--request-slots` 8 / 32 x `--poll-skip` 1 / 100. Decide the
   defaults of both from it.
   **Results 2026-11-25** (`briefkasten-tuning_26_11_24`, `-rmat`; 3072 missing). Global phase, s:

   | input, cores | slots 8 / skip 1 | 8 / 100 | 32 / 1 | 32 / 100 |
   |---|---|---|---|---|
   | RGG perm. 192 | 4.60 | 4.04 | 5.01 | 4.23 |
   | RGG perm. 1536 | 4.92 | 4.48 | 5.52 | 4.68 |
   | GNM 1536 | 1.43 | 1.08 | 1.56 | 1.33 |
   | RHG perm. 1536 | 4.54 | 4.09 | 5.07 | 4.28 |
   | RMAT 48 | 6.93 | 6.83 | 7.20 | 6.89 |
   | RMAT 768 | 27.5 | 27.6 | 27.0 | 25.8 |

   With 75d4f95, `--poll-skip 100` is best or equal everywhere (no blow-up up to 1536). 32 slots are slower on
   RGG / GNM / RHG; on RMAT 32 / 100 is ~7 % faster at 768, but buffers still park for a slot (105 K per rank vs
   145 K with 8): the busiest rank's volume, not the slot count. Decided: default `--poll-skip 100`, request slots
   stay 8 (revisit RMAT with the 3072 data). Suites that ran with skip 1 (`briefkasten`, `-rmat`, `-cma-debug`) pin
   `poll-skip: 1`.
   **CMA debug, first run** (`briefkasten-cma-debug_26_11_24`): only c4 (direct) passed; all grid runs completed
   iteration 1 with the right count and aborted in iteration 2, also those with `I_MPI_SHM_CMA=0`, in the same CMA
   code: the variable is not documented for Intel MPI 2021.15 and had no effect. The suite now uses
   `I_MPI_SHM=off` (no shared-memory transport) with `I_MPI_DEBUG=5`. Earlier, at 768 cores, the grid runs (and
   legacy grid) aborted in iteration 1, so it is timing-dependent, not a carry-over between iterations as such.
   **Not a capacity problem** (2026-11-25): CMA does not go through the shared-memory cells; the receiver copies
   straight from the sender's memory (`process_vm_readv`), and a short read means part of that memory was not
   mapped any more, a buffer freed or reallocated under MPI (or an Intel MPI bug). Chunking (or `I_MPI_SHM=off`)
   would hide either. **Buffer lifetime check:** BriefKAsten `feat/buffer-lifetime-check` @ d9bf03e (PR #27,
   on #26; katric pinned to it), CMake option `BRIEFKASTEN_CHECK_BUFFERS`: send buffers unchanged (address, size, checksum) from
   `MPI_Isend` to completion, MPI has released the request when a send is reported done, no slot reused or Sender
   destroyed with a send in flight, armed persistent receive buffers not moved. Caught an injected modification of
   in-flight buffers. Locally (Open MPI, katric `build-check`): RMAT 2^14 / 2^16, 9 / 16 / 25 / 36 ranks, grid and
   direct, 16 / 64 KiB, 3 iterations each: no violation, counts match. Legacy grid (a separate queue) aborts the
   same way at 768, which points at Intel MPI rather than a shared bug. Next: the check build on SuperMUC-NG
   (`briefkasten-cma-debug` with a binary built with `-DBRIEFKASTEN_CHECK_BUFFERS=ON`); also, `resize_buffers` of
   the persistent receiver re-inits its receives without starting them (unused by katric).
   **CMA debug, second run** (check build, `full`, 3 iterations, same directory as the first run, which it replaced):
   all five runs passed, also grid 16 / 64 KiB with the normal transport, which had aborted every time; no check
   fired. Inconclusive: the full check reads every send buffer twice and made an iteration ~45 % slower (45 vs 31 s),
   and the abort is timing-dependent. `I_MPI_SHM=off` took effect (`I_MPI_DEBUG`: psm2 provider for intra-node
   traffic) and cost nothing measurable in that build. Hence the check mode is chosen at run time now (BriefKAsten
   5a84bb9, environment variable `BRIEFKASTEN_CHECK_BUFFERS`: `full` default, `light` = addresses / sizes / requests
   only, `off`), and katric pins 5a84bb9 (PR #26 was squash-merged as 9c4bbf0 and #27 rebased onto it, so the
   earlier pin d9bf03e is on no branch any more). **Third run:** `briefkasten-cma-debug`, grid 16 / 64 KiB x checks
   off / light / full, plus `I_MPI_SHM=off` with checks off, 5 iterations each.
   **CMA debug, third run** (`briefkasten-cma-debug_26_11_25`, 8 nodes i01r01c02s[06-11], c04s[04-05]): only c4 / c5
   (full check) passed, 5 iterations, ~46.5 s each, right count. c0-c3 (off, light) and c6 / c7 (`I_MPI_SHM=off`)
   all completed iteration 1 (~31 s, right count) and aborted in iteration 2. The light check never fired.
   c0-c3: the Intel shm CMA assertion as before. c6 / c7: PSM2's intra-node path (libfabric psm2 provider, ptl_am)
   fails the same way, `Reading from remote process' memory failed. Disabling CMA support`, then
   `ptl.c:153: nbytes == req->req_data.recv_msglen`. So `I_MPI_SHM=off` is **no workaround**: it only swaps one
   `process_vm_readv` user for another. Every abort at 384 (this run and `briefkasten-rmat_26_11_24`) involves the
   last node (ranks 336-383), sometimes also the node of ranks 192-239. Two independent CMA implementations plus
   the legacy queue failing alike point at `process_vm_readv` itself failing on that node (memory gone from the
   sender's or the receiver's address space, or the call failing for want of memory), not at one transport's bug;
   the light check rules out BriefKAsten's own send slots and armed persistent receives. Open: the errno and the
   node's memory state at the failure (no memory data in these runs; THP `always`, defrag `madvise`).
   **CMA debug, fourth run** (2026-11-25, same directory as run 3, whose `.cali` / log / err it replaced; strace of
   failed `process_vm_readv` and a per-node memory sampler, `briefkasten-cma-debug.command-template.txt`): **the node
   runs out of memory.** strace: a rank is `killed by SIGKILL` first (c2: rank 340), milliseconds later a peer's
   `process_vm_readv` from it fails with `ESRCH (No such process)` and the transport asserts (Intel shm and PSM2
   alike); no `EFAULT`, so no buffer was freed under MPI. The node of ranks 336-383 has MemAvailable 0 in every failing
   run (kernel OOM killer: the job cgroup has no limit and counted no OOM kill). c0 (grid 16 KiB) on that node:
   anonymous memory 11 GB after generation, ~81 GB at the peak of iteration 1 (MemAvailable touches 0, survives),
   **~73 GB between the iterations**, more in iteration 2 -> killed; grid 4 KiB peaks at ~49 GB and stays there
   through iterations 2 / 3. Largest rank ~2 GB: every rank on the node grows, not one. katric keeps only the input
   between iterations (`count_triangles` frees the rest, the queue is created per global phase), so the memory held
   after an iteration is malloc's free memory or memory someone else holds (the user has seen MPI's eager buffers
   stay before). BriefKAsten's aggregation pool is capped (parked buffers count against it). Not pursued: releasing
   pages (`malloc_trim` and the like), which lost messages under MPI in an earlier project.
   Next: suite `rmat-memory`, the read-only `KATRIC_MEMORY_PROBE` (`include/katric/memory_probe.h`): per rank RSS,
   peak RSS, glibc heap used / free / mmap at the phase boundaries.
   **Root cause: a leak in BriefKAsten's buffer pool** (`rmat-memory_26_11_25`, 2026-11-25). Grid 4 KiB, node of ranks
   336-383: heap in use 2.4 GB after the local phase, 28.3 GB at the end of the send loop, 37.5 GB after termination
   (everything delivered, queue alive), 2.4 GB once the queue is destroyed (then 36.4 GB free in malloc, reused by the
   next iteration, so 4 KiB plateaus); up to 1.05 GB on one rank. Direct 4 KiB: 5 GB. Grid 16 KiB was killed in
   iteration 1. So the queue held it, not MPI. Since BriefKAsten 86fbe94 (2026-09-18) the overflow path flushed the
   current buffer with `erase=false`, leaving a moved-from shell (capacity 0) in the destination's entry, and
   `replace_flushed_buffer` recycled that shell into the free list; the sent buffer came back as well. One extra
   free-list entry per overflow flush, filling a shell allocated a buffer the pool never counted, and the free list
   was never empty, so the pool cap and buffer stalls never took effect (`buffer_stalls` = 0 in every BriefKAsten
   run). Locally (RMAT 2^18, 16 ranks, grid 4 KiB): free list 19 249 entries / 45 MB against a counted pool of 8.
   **Fix** BriefKAsten 82d1283 (`fix/buffer-pool-shells`, on `feat/buffer-lifetime-check`; katric pins it): a flush
   that moves the current buffer out erases the entry; `recycle_buffer` asserts free list <= pool; test
   `buffer_pool_test` (failed on every variant before). With the pool bounded the proxy's overdraft path became
   reachable and two latent problems surfaced: own messages could take the overdraft buffers (now they acquire only
   below the cap), and the overdraft ceiling assumed buffers full of redirected payload, while the proxy's buffer for
   a destination also carries own messages (bound now `redirect_high_water + 1` buffers, per element; KASSERT only,
   level 0 in experiment builds). Open design question: own payload riding in overdraft buffers. Locally (RMAT 2^18,
   9 ranks, 3 iterations, Release) same counts; max rank RSS 164-175 -> 131-133 MB; min time grid 4 / 16 KiB
   0.52-0.53 -> 0.50-0.52 s, direct 4 KiB 0.54 -> 0.48 s. Consequences: every BriefKAsten measurement since
   2026-11-24 ran with the leak (absolute times and memory; no pool stalls); KaCCv2 pins it since 2026-09-18
   (`5664f7f`), with 7-50 K overflows per second on the busiest rank in its async configurations. Next: re-run
   `rmat-memory` and `briefkasten-rmat` with 82d1283.
   **Reserve bound** (BriefKAsten 1404471, `fix/reserve-own-pause`, PR #30 on #29; katric pins it). With #29 the
   proxy's reserve (buffers beyond the pool cap) was bounded only per forwarded element: own messages merge into the
   proxy's shared buffer for a destination without acquiring, a forwarded message overflows it, and the next
   forwarded message draws another reserve buffer, so memory could exceed O(w p' Delta) (the cc-paper's claim) by up
   to Delta / message size. Now an own post (outside a handler; only forwarding posts from handlers, asserted in
   `split_handler`) waits while the reserve holds more than one buffer per peer. No deadlock: forwarding never
   waits, so every rank keeps consuming and granting while it polls. Reserve <= 2 ceil(redirect_high_water / buffer)
   + 3 p' + send slots + 2, asserted. Shared buffers (aggregation) unchanged. katric reports
   `peak_reserve_buffers`, `redirect_overdraft`, `reserve_waits` in the global phase. Locally (RMAT 2^18, 9 ranks,
   grid 4 KiB): same count and time within noise of 82d1283; peak reserve 50 buffers, 11 reserve waits in total,
   and `buffer_stalls` up to 445 K per rank (0 with the leak: the pool's backpressure is back). On the cluster,
   watch `buffer_stalls` next to the global-phase time.
   **Cluster, leak fix only** (katric 392b106 = BriefKAsten 82d1283; 2026-11-25). `rmat-memory_fix` (384): grid 16 KiB
   passes (OOM before); busiest node's anonymous memory ~13.7 GB in every configuration (32-55 GB before); heap in use
   per rank after termination 1.05 -> 0.10 GB (grid 4 KiB); global phase per iteration grid 4 KiB 22.1 -> 19.4 s,
   direct 4 KiB 19.4 -> 18.9 s (all runs still hang until I_MPI_JOB_TIMEOUT after the last iteration, as before:
   the suite, likely the per-rank recorder). `briefkasten-rmat` (768, killed after c3): **direct is slower**, global
   phase (3 iterations, max) 4 KiB 83.5 -> 130.4 s, 16 KiB 76.4 -> 123.8 s, buffer_stalls 0 -> 60 K, polls 41 M ->
   71 M on the busiest rank, same messages and overflows; legacy direct +3 % in the same job; legacy grid died again
   (its own memory problem, not this pool). The pool default (2 buffers per peer, BriefKAsten ec55432) was set on
   2026-09-18 after the leak, i.e. never measured with a working cap. Locally a larger pool is not faster (stalls do
   not drop either: the busiest sender outruns its 8 send slots with any finite pool). katric `--buffers-per-peer`
   (default 2); suite `briefkasten-pool-768` (one short job: direct 4 KiB with 2 / 32 per peer, grid 16 KiB).
   The debugging suites (`briefkasten-cma-debug`, `-spread`, their command template, `rmat-memory`) and the memory
   probe were removed once the cause was fixed (2026-11-25); they are in git history before that commit.
   **`briefkasten-pool-768`** (768, one iteration each): grid 16 KiB passes (peak reserve 500 buffers, 1 reserve
   wait). Direct 4 KiB: global phase 42.8 s with 2 and with 32 buffers per peer (stalls 20.7 K and 0): the pool size
   is not the cause. The time is in the busiest rank's send loop: `send_ns` 16.4 s (11-24, before the fix, iteration
   1) -> 30.0 / 29.9 s; handle / poll times unchanged. Locally (9 ranks) the send loop also doubles (0.027 -> 0.06 s),
   but there only through stalls: with 1000 buffers per peer it is back at 0.022 s. (Correction: the earlier local
   "2 per peer ~ the leak" sweep compared two fixed builds; `build-perf` had re-fetched the new pin.)
   **Cause of the slower send loop: the buffer check was on.** Every cluster run since the fix (katric 392b106 /
   7378e6a: `briefkasten-rmat_26_11_25`, `rmat-memory_fix`, `briefkasten-pool-768`) was built with
   `BRIEFKASTEN_CHECK_BUFFERS` (the CMake cache of the CMA debugging, run 3) and set no environment variable, so the
   check ran in its default mode, full (checksums of every send buffer at the send and at completion), on the
   sender. The 11-24 baseline had no check. So the memory results stand, the 384-core runs were faster than before
   even with the check, and the 768-core time comparison is void. Next: rebuild with
   `-DBRIEFKASTEN_CHECK_BUFFERS=OFF` and re-run `briefkasten-pool-768`.
   **Cleanup once the re-run confirms the fix** (decided 2026-11-25: remove the debugging features added on the way):
   - [x] BriefKAsten: `BRIEFKASTEN_CHECK_BUFFERS` dropped: #27 closed (branch kept), #29 / #30 moved onto
         `feat/contiguous-envelope-serialization` (e694854 / 675048a); katric pins 675048a.
   - [x] katric: `--buffers-per-peer` (the default of 2 beat 32) and the suite `briefkasten-pool-768` removed.
   - Keep: the assertions (free list <= pool, reserve <= ceiling, no posting from handlers), `num_free_buffers`,
     the reserve counters and katric's report of them.
   - Kept for now (user, 2026-11-25): BriefKAsten `BRIEFKASTEN_STALL_TRACE` and the 2026-09 transport-pressure
     counters, katric's straggler instrumentation (`straggler_debug.h`, `resource_counters.h`).
   **Verified** (`briefkasten-pool-768_no-debug`, 768 cores, one iteration, no check): direct 4 KiB global phase
   28.0 -> 25.1 s (2 buffers per peer; 27.1 s with 32), busiest rank's send loop 16.4 -> 12.7 s; grid 16 KiB passes,
   22.7 s (grid 4 KiB before the fix: 31.0 s); peak reserve 497 buffers, 1 reserve wait.
   **Correction (same day):** these before / after numbers are not clean. The post-fix builds were configured fresh
   with the experiments preset (-march=native, KaMPIng assertions none), the pre-fix ones (briefkasten-rmat,
   briefkasten-tuning*, 11-24) came from an older build cache (no -march=native, KaMPIng types assertions at level
   10), and the 768-core baseline (briefkasten-rmat_26_11_24, BriefKAsten 0062f64) predates #26. So neither the
   speed-up nor "no degradation" is shown; 2 vs 32 buffers per peer was the only clean comparison (same build).
   Instead of two builds (separate build directories mix build states), one binary: BriefKAsten exp/pool-fix-switch
   (d2b5a24 = 675048a + `Config::bounded_pool`, a run-time switch back to the pool handling from before #29; not for
   main), katric `--legacy-pool` and a temporary pin to it. Suite `briefkasten-pool-ab`: the tuned default (direct
   4 KiB, 8 slots, poll-skip 100, default load balancing) at 768 cores on permuted RGG and RMAT, fix vs
   `--legacy-pool`.
   **Result** (`briefkasten-pool-ab_26_11_25`, one binary, 2 iterations): permuted RGG global phase 4.46 / 4.41 s
   with the fix vs 4.51 / 4.28 s without (no stalls either way: no difference); RMAT 25.0 / 25.0 s vs 28.1 / 28.2 s
   (-11 %; busiest rank's send loop 13.1 vs 16.6 s; ~20 K stalls with the fix). The legacy setting reproduces the
   pre-fix baselines (28.0 s briefkasten-rmat, 27.6 s briefkasten-tuning-rmat), so the build differences did not
   matter there. **No degradation for direct at 768; the tuned defaults stay.**
   Open: the grid (small cap ~2 sqrt(p) peers, the only policy with the reserve pause; most overflow flushes before
   the fix) and high p. The switch stays until these are measured: `briefkasten-pool-ab-grid` (768, test partition,
   grid 4 / 16 KiB) and `briefkasten-pool-ab-large` (1536-6144, direct and grid). Afterwards pin back to 675048a (or
   the merged #30) and drop `--legacy-pool` and the three suites; BriefKAsten exp/pool-fix-switch is not for main.
   **Grid result** (`briefkasten-pool-ab-grid_26_11_25`, 768 cores, one binary, 1 iteration; global phase fix vs
   legacy): permuted RGG 4 KiB 6.11 vs 6.45 s, 16 KiB 4.92 vs 5.51 s; RMAT 4 KiB 28.1 vs 31.5 s (-11 %), 16 KiB 23.3 s
   vs **crash**: the legacy run died with the Intel MPI CMA assertion (`cma_read_nbytes == size`, the leak's original
   symptom). With the fix the cap binds hard (RMAT buffer_stalls 0.7 M at 4 KiB, 1.05 M at 16 KiB; peak reserve 293 /
   499 buffers, 3 reserve waits) and is still faster: the stalls cost less than the leak's allocations. No
   degradation on the grid either. (Legacy peak_reserve_buffers is negative: the leaked free list outgrows the count.)
   KaCCv2 (same switch, branch exp/briefkasten-pool-ab, notes/briefkasten-pool-fix-findings.md): grid 7-8 % faster
   with up to 34 % less memory; flat ms-async on lake-lanier +7 % (not the cap: 12-15 % more, less full sends for
   the same elements). Still open: `briefkasten-pool-ab-large` (katric) and KaCCv2 at 1536-6144.
   **1536 cores** (`briefkasten-pool-ab-large_26_11_25`, one binary, 1 iteration; 3072 / 6144 pending; global
   phase fix vs legacy): RMAT direct 39.2 vs 51.5 s (-24 %; busiest rank's send loop 23.7 vs 38.9 s, with 2.2 M
   stalls), RMAT grid 40.8 vs 49.1 s (-17 %; 0.88 M stalls, peak reserve 458 buffers, no reserve waits), permuted
   RGG grid 5.91 vs 6.67 s (-11 %), direct 4.55 vs 4.62 s (no difference). The fix's lead grows with p (RMAT direct
   -11 % at 768, -24 % at 1536). No memory numbers: the suite records no RSS.
   KaCCv2 at 1536: grid lake-lanier -8 % and 24 % less RSS; flat ms-async on lake-lanier now +24 % (+7 % at 384),
   in label propagation with zero stalls: +39 % `terminate()` calls and so +28 % drain flushes of partial buffers,
   same elements. Not the cap; the trigger is open (KaCCv2 notes/briefkasten-pool-fix-findings.md, suite
   `briefkasten-pool-ab-rgg` for a synthetic repro). katric cannot show this effect: its global phase calls
   `terminate()` once, after all posts, so there is no repeated idle / drain cycle.
   **3072 / 6144** (same suite and binary, 81e91dc; 1 iteration; global phase fix vs legacy): RMAT direct 97.9 vs
   148.0 s at 3072 (-34 %; busiest send loop 77 vs 126 s, 9.3 M stalls), at 6144 fix 224 s while legacy exceeded
   the 6-minute run limit; RMAT grid 64.4 s at 3072 and 81.0 s at 6144 with the fix, legacy **killed by signal 9
   on whole nodes** at both (3 nodes at 3072, 2 at 6144: the OOM killer, the leak); permuted RGG grid 6.29 vs
   7.69 s (-18 %) and 7.17 vs 8.55 s (-16 %), direct equal (4.87 / 4.90 s, 8.39 / 8.33 s). Peak reserve 765 / 1083
   buffers, no reserve waits. **The fix wins more the higher p, and legacy does not run RMAT at high p at all.**
   KaCCv2 (notes on exp/briefkasten-pool-ab, 2026-11-26): the flat slowdown there comes from the overflow erase
   changing the termination drain's order (libstdc++ iteration order), not from the bounded pool; BriefKAsten
   `Config::drop_shells` (drop the moved-from shells instead of erasing the entry) is as fast as legacy and as lean
   as the fix. Candidate replacement for the erase of #29; katric needs to be checked with it.
   **Candidate checked** (`briefkasten-drop-shells_26_11_26`, BriefKAsten exp/drain-policy cad2789, 768, one binary;
   `--drop-shells` = the pool fix with #30, overflow flushes keep the entry and drop the shell instead of #29's
   erase): global phase equal to the fix within 0.5 % (RMAT direct 24.94 vs 25.08 s, grid 27.70 vs 27.82 s; permuted
   RGG direct 4.53 / 4.53 s, grid 5.70 / 5.65 s), same stalls and reserve. Found on the way: with the entry kept, a
   proxy's handler could merge into the shell and grow it into an uncounted buffer (RMAT grid abort, fixed in
   cad2789). KaCCv2 on the same base: candidate = fix as well; the base's idle flush makes the drain order moot.
9. **Buffer memory bound** (2026-11-27). Goal: O(B p') aggregation buffer memory per rank for direct and grid, with
   B the buffer size (local threshold, in elements) and p' the peers (direct p, grid ceil(p/g) + g with
   g = round(sqrt p), i.e. ~2 sqrt p). C = credit window per peer in buffers (`--credit-buffers`, default 8),
   b = pool buffers per peer (2), s = send slots (8). **Assumptions:** no large messages (every message fits a
   buffer), flow control with C >= 2, and for the grid BriefKAsten `separate_forwarding` + `exact_redirect_limit`
   (exp/split-forwarding 7736e33, `--separate-forwarding --exact-redirect-limit`).
   - **Direct:** at most **b p' + s** buffers. Own posts wait at the pool cap, nothing is forwarded.
   - **Grid:** own + forwarded <= (b p' + s) + (2 C p' + p' + s) = **(2C + b + 1) p' + 2s** buffers.
     Own: as direct (own posts wait at the cap and never merge into forwarded buffers).
     Forwarded: (1) the exact limit keeps the forwarded elements a proxy holds (filling, parked, in flight)
     <= C B p'; only buffers larger than half the window may pass with too little credit, and with C >= 2 and no
     large messages there are none. (2) Per destination, a forwarded buffer that leaves before it overflowed (idle
     flush, termination drain) is sent at once and holds a send slot; every other one was flushed by the message
     that went into the next, so two consecutive ones hold more than B elements: <= 2 E_d / B + 1 + (in slots).
     (3) Summed: <= 2 C p' + p' + s.
   - Defaults: C = 8: 19 p' + 16; C = 4: **11 p' + 16**; C = 2: 7 p' + 16 (768 cores, p' = 56: 1080 / 632 / 408
     buffers, 2.5 MiB at 4 KiB for C = 4). Direct 2p + 8. The grid's bound is lower above p ~ 120 (11 * 2 sqrt p
     = 2p), 7x lower at 6144.
   - **Without the switches** (#30 as is): (2C + b + 3) p' + 2C + 2s + 2 (21 p' + 34), still O(B p'): #30's pause
     threshold (one buffer per peer of own payload in the reserve), a second partial buffer per destination, and one
     window of overshoot at the redirect limit. The old bound also missed the termination drain parking partial
     forwarded buffers drain after drain (bounded only per element); `separate_forwarding` closes that.
   - Not counted: receive buffers (a fixed number of slots x B) and the credit state (O(p') words).
   - Measured, busiest rank: 22-30 % of the bound at 768-6144 cores (RMAT grid, pool A/B), so usage scales with it.
   **`briefkasten-credit-window_26_11_27`** (768, grid 4 KiB, 1 iteration, C = 8 / 4 / 2): global phase RMAT 26.77 /
   26.93 / 27.47 s, permuted RGG 5.73 / 5.68 / 5.81 s; busiest rank's buffers RMAT 421 / 268 / 204, RGG 504 / 280 /
   165; same sends; parked for credit RMAT 6 K / 18 K / 62 K. C = 4 is free; C = 2's +2.6 % is within the run-to-run
   noise (identical configurations differ by 1.8-3.6 % between suites).
   **`briefkasten-split-forwarding_26_11_27`** (768, 4 KiB, C = 8; base / separate / exact / both): katric global
   phase RMAT 26.30 / 25.83 / 26.26 / 25.87 s, RGG 5.53 / 5.58 / 5.70 / 5.56 s, same sends, no reserve waits.
   KaCCv2 (exp/split-forwarding 0e0f81e, grid best setting, 2 iterations): lake-lanier cc 7.40 / 7.43 / 7.37 /
   7.42 s (mean), +5.0 % sends and +6 % idle flushes with separate forwarding (own and forwarded messages no longer
   share partial flushes), 153 reserve waits in the base arm and none with it; twitter, rgg3d perm, RMAT neutral;
   peak RSS equal. The switches cost nothing measurable; they shorten the analysis, not the memory used.
   **`briefkasten-bound-candidate_26_11_27`** (768, 3 iterations, C = 8 off / C = 4 both / C = 2 both; iterations
   within 0.3 % of each other): global phase RMAT 26.30 / 25.97 / 26.46 s, permuted RGG 5.58 / 5.58 / 5.64 s; busiest
   rank's buffers RMAT 418 / 266 / 196, RGG 504 / 285 / 188; same sends. **C = 4 with both switches is free for
   katric, C = 2 costs ~1 %.** KaCCv2 differs: lake-lanier +3.9 % (C = 4) / +7.3 % (C = 2). The 2 x 2
   `briefkasten-window-mechanism` (KaCCv2 notes) separates it: separate forwarding +0.2 % (more sends and idle
   flushes, but cheap), C = 4 alone +1.5 % through flow control at the proxies (grants x2, withheld grants x16 with
   the halved redirect limit), not idle cycles. With its tiny messages KaCCv2 has the factor ~1 instead of 2, so
   C = 8 gives it ~11 p' + 16 as well. **Decision candidate: same bound, window per application: katric C = 4,
   KaCCv2 C = 8, both with both switches** (KaCCv2 lake-lanier then 7.44 s at 768, -12 % vs the paper).
   **Open:** C = 4 / C = 2 with both switches against today's default, 3 iterations (`briefkasten-bound-candidate`,
   768, test partition; `briefkasten-bound-candidate-large`, 1536-6144, general partition, 2 iterations; KaCCv2
   suites of the same names). Risk at larger p: separate forwarding on strong-scaled lake-lanier (smaller flushes,
   more of them partial). Not pursued: a shared forwarding budget R < C p' (needs a guaranteed window per link,
   since credit cannot be revoked; ~1 MiB per rank at 768).
   **One setup for both applications (2026-11-27): C = 8 with both switches**, unless a later run contradicts it.
   KaCCv2's lake-lanier pays for C = 4 (+1.5-3.9 % at 768, +9 % at 1536 in one job, 0 at 3072) and katric gains
   nothing from it, so C = 4 would trade time for a tighter constant (11 p' vs 19 p'; ~5 MiB per rank at 6144).
   **`briefkasten-withhold-diagnosis_26_11_27`** (768, both switches, C = 8 / 4; BriefKAsten 9a7e264 samples a
   proxy's redirect budget whenever it starts withholding a grant: unused credit of proxied peers, forwarded
   elements filling / parked for credit / parked for a send slot / in flight, as share of the budget C B p'):
   | input | C = 8 / 4 | withhold episodes | withheld link-s | credit / filling / parked credit / parked slot / in flight |
   |---|---|---|---|---|
   | KaCCv2 lake-lanier, label propagation | 7.34 / 7.57 s cc (+3.1 %) | 1.2 K / 22.7 K | 0.9 / 19.0 | 37 / 6 / 34 / 21 / 2 % (C = 4) |
   | KaCCv2 lake-lanier, reachability | | 1.4 K / 5.8 K | 9 / 36 | 26 / 4 / 67 / 1 / 0 % |
   | KaCCv2 webbase-2001, reachability | 0.416 / 0.415 s | 0.6 K / 1.3 K | 0.1 / 0.2 | 39 / 6 / 37 / 16 / 1 % |
   | katric RMAT | 25.91 / 26.15 s global (+0.9 %) | 136 K / 375 K | 548 / 1379 | 25 / 4 / 56 / 13 / 3 % |
   | katric RGG perm | 5.66 / 5.63 s | 3.5 K / 7.6 K | 22 / 32 | 13 / 6 / 2 / 76 / 3 % |
   (link-s summed over ranks and links, first iteration.) Of the three suspects, **onward credit wins**: forwarded
   payload parked because the final receiver has not granted is the largest data share (34-67 %), then forwarded
   payload waiting for a send slot; filling buffers (2-6 %) and in-flight sends (0-3 %) are small, so neither a
   flush on withhold nor more polling at the proxy would help. The rest is credit granted to row peers and not yet
   used (13-39 %), reserved whether a peer sends or not. Withholding then blocks the whole link from a row peer,
   also for destinations that are not congested: head-of-line blocking at the proxy, behind slow receivers. katric
   withholds far more (RMAT 1379 link-s at C = 4) without paying for it (its exchange is one bulk transfer, ending
   when the volume is through); in KaCCv2 the time goes into label propagation (cc +0.23 s: coloring +0.17 s,
   reachability +0.06 s), whose many dependent rounds wait on the stalled links. The window is therefore not
   oversized for KaCCv2: C = 8 is the buffer that absorbs hot receivers.
   **`briefkasten-validation_26_11_27`** (768, today's default grid C = 8 off vs the candidate C = 8 with both
   switches; mean / best of the iterations): KaCCv2 (3 iterations, cc) twitter -5.6 / -1.6 %, friendster -4.2 /
   -2.4 %, uk-2007-05 -0.9 / -1.4 %, webbase-2001 -2.0 / -2.4 %, lake-lanier 0.0 / +1.0 %, rgg2d(18,21) -9.5 /
   -22.5 %, permuted -10.1 / -2.3 %, rgg3d(20,21) perm +3.8 / +0.9 %, rmat(16,19) +6.3 / -2.1 % (rgg3d and rmat
   with one slow iteration each, spread larger than the difference). katric (2 iterations, global phase) orkut
   -0.7 / 0 %, live-journal 0 / 0 %, twitter +0.8 / +0.4 %, friendster -1.9 / -2.2 %, webbase-2001 +1.9 / 0 %;
   the candidate withholds less often in total (twitter 5093 -> 2449 link-s). **No regression on any input**: the
   candidate stands.
   **`briefkasten-bound-candidate-large_26_11_27`** (2 iterations, global phase mean; C = 8 off / C = 4 both / C = 2
   both): RGG perm 1536 5.75 / 5.92 / 6.01 s, 3072 6.23 / 6.28 / 6.32 s; RMAT 1536 39.06 / 38.73 / 39.66 s, 3072
   63.25 / 62.05 / 62.64 s (6144 not back yet). Within +-3 % (RGG's spread between iterations is that large), no
   trend with p: katric stays indifferent to the window. Busiest rank's buffers 708 / 414 / 262 (RGG 1536), 937 / 567
   / 369 (RGG 3072), 606 / 394 / 294 (RMAT 1536), 698 / 496 / 367 (RMAT 3072), i.e. 33-66 % of the pool ceiling
   (1527 / 719 / 403 at 1536, 2135 / 1007 / 563 at 3072); the average rank uses 165-421.
   KaCCv2 `briefkasten-window-mechanism` at 1536 (lake-lanier, 3 iterations, median cc): C = 8 off 3.86 s, C = 4 off
   4.02 s (+4 %), C = 8 both 3.92 s (+1.6 %; one slow first iteration, mean +4 %), C = 4 both 4.02 s (+4 %). The +9 %
   of the single -large job at 1536 does not repeat; the window costs ~4 % at 1536 as at 768, and nothing at 3072.
   **Credit budget, first version (BriefKAsten exp/credit-budget f776a37, `--credit-budget`):** links start at two
   buffers, grow by one buffer per due grant up to the largest window while all inbound windows plus the forwarded
   payload held fit the budget, and halve at their next grant when they do not; the budget is also the redirect
   limit, so a budget of 4 per peer has fixed C = 4's bound. **`briefkasten-credit-budget_26_11_27`** (768, both
   switches; fixed C = 8 / fixed C = 4 / budget 4 up to 8 / budget 4 up to 16): KaCCv2 lake-lanier (median cc) 7.42 /
   7.65 / 7.77 / 7.72 s (+3.0 / +4.6 / +4.0 %), webbase-2001 0.410 / 0.413 / 0.415 / 0.415 s, rgg2d(18,21) perm 1.71
   / 1.74 / 2.02 / 1.78 s (budget up to 8 noisy: 2.25 / 2.02 / 1.79); katric (global phase mean) RGG perm 5.65 / 5.71
   / 5.73 / 5.76 s, RMAT 27.13 / 27.31 / 27.52 / 27.53 s. **The budget is no better than fixed C = 4 at the same
   bound, slightly worse.** The budget stays full (peak inbound credit 103-111 K of 115 K elements) and the windows
   churn: lake-lanier label propagation 8.5 M grows and 5.1 M halvings, grants 17.0 M (C = 8) / 33.5 M (C = 4) /
   39.8 M (budget); katric RMAT 68 M grows, 41 M halvings, grants +18 % over C = 4. It withholds less than C = 4
   (lake-lanier 117 K vs 255 K) but does not recover C = 8's time. Reason: the demand is spread over many links at
   once (as the feeder split showed, most withheld row peers send a little everywhere), so a budget of 4 per peer
   ends up near 4 per link plus the churn. A budget only helps where few links are busy at a time; that is the
   direct case at large p (C B p of inbound credit, mostly idle), not the grid. **Not pursued for the grid; fixed
   C = 8 stays.** Open: direct sending at large p, where the budget would cut the C B p of inbound credit (what MPI
   may have to buffer) if the idle links really are idle.
   **Head-of-line switches (BriefKAsten exp/credit-budget 288781b), `briefkasten-hol_26_11_27`** (768, both switches):
   the delivery window (`--delivery-credit-buffers`: links that are only delivered over, proxy -> final receiver, get
   a larger window; the proxy's bound depends only on the window of the proxied links, so C = 4 keeps 11 p' + 16) and
   feeder-first (above 3/4 of the redirect limit, withhold first from proxied peers whose recent forwarded payload
   went >= 25 % to destinations with payload parked). Relative to C = 8 (median cc / mean global phase):
   | arm | KaCCv2 lake-lanier | webbase-2001 | katric RGG perm | katric RMAT |
   |---|---|---|---|---|
   | C = 8 | 7.49 s | 0.409 s | 5.71 s | 27.34 s |
   | C = 4 | +3.6 % | +1.4 % | +1.4 % | +0.8 % |
   | C = 4, delivery 8 | **+1.3 %** | +0.9 % | +0.8 % | +1.3 % |
   | C = 4, delivery 16 | +1.7 % | +0.6 % | +1.3 % | +0.3 % |
   | C = 4, feeder-first | +4.5 % | +2.4 % | +2.6 % | +1.2 % |
   | C = 4, delivery 8, feeder-first | +3.9 % | +2.1 % | +2.8 % | +1.4 % |
   | C = 8, feeder-first | +2.0 % | +1.2 % | +1.1 % | 0.0 % |
   (KaCCv2 rgg2d perm: C = 8's iterations spread 1.68-2.13 s, all other arms 1.72-1.86 s; no signal.) **The
   delivery window recovers about two thirds of lake-lanier's C = 4 cost at C = 4's bound**; it does so through
   fewer grants (label propagation 33.4 M at C = 4, 25.4 M with delivery 8, 17.0 M at C = 8), not fewer withholds
   (256 K vs 247 K), and 16 is no better than 8. **Feeder-first makes it worse everywhere**: it withholds more (lake-
   lanier 664 K, 700 K of them feeder withholds) without keeping the bystanders' grants flowing, because the
   congested payload comes from all row peers a little each, so stalling the feeders earlier does not stop the
   limit from filling. Dropped. Peak RSS is the same in every arm (katric RGG 536-538 MiB, RMAT 1982-1984 MiB):
   the 200 buffers between C = 8 and C = 4 (0.8 MiB) do not show at 768.
   **`briefkasten-grant-cost_26_11_27`** (KaCCv2 lake-lanier 768, both switches, median cc; `grant_divisor` d: a grant
   falls due after 1/d of the window, default 2): C = 8 d = 2 7.47 s; C = 8 d = 4 +0.7 % (label propagation grants
   33.5 M, as many as C = 4); C = 8 d = 8 +3.0 % (65.0 M); C = 4 +3.5 % (33.5 M, withheld 240 K vs 20 K, withheld
   link-s 17.9 vs 1.0); C = 4 + delivery 8 +1.8 % (25.4 M, withheld 248 K). **The grants explain about 0.7 of C = 4's
   3.5 %; the rest is the halved window / redirect limit** (12x the withholds, more payload parked for onward credit).
   The delivery window removes about half of that rest, more than its saved grants account for (parked for credit
   638 K -> 570 K, reachability withheld link-s 32.8 -> 25.3): proxies pass forwarded payload on faster. Batching or
   piggybacking grants would gain at most ~0.7 %: not pursued.
   **Combining at proxies (the classic hot-spot remedy): not worth it.** On lake-lanier, where the C = 4 cost lives,
   92 % of reachability and 76 % of label propagation sends are idle flushes of 5-18 % full buffers, and only 0-1 %
   of the messages in a buffer share a target vertex (`briefkasten-combine-potential_26_11_27`), so combining would
   neither shrink nor save sends. Where buffers fill and duplicates exist (KaCCv2 reachability on web and social
   graphs), sender-side caching in KaCCv2 removes them; that is a KaCCv2 question (KaCCv2 notes: forward caching
   -55 % on uk-2007-05, +67 % on rgg2d perm, decided by the share of messages it removes).
   **Not counted in the bound: MPI's unexpected messages.** Only the receive slots are posted, so a buffer beyond
   them arrives as an eager unexpected message that MPI buffers internally. Per receiver that is at most the credit
   granted to it, C B per sender: C B p for direct sending (8 x 4 KiB x 6144 = 192 MiB at 6144 cores if every
   rank sends to one), C B p' with indirection. Not measured; in practice senders rarely target one rank at once.
   **Open design direction (dissertation): a per-receiver credit budget** (credit where the traffic is, a minimum
   of two buffers per link, active links growing up to a larger window), which would bound what a receiver can be
   sent independently of p and stop reserving a full window for idle row peers. It would not remove the
   head-of-line blocking behind slow receivers, which the diagnosis shows is what C = 4 costs KaCCv2.

## Decisions (2026-11-24)

- katric moves to C++23 (no range-v3).
- BriefKAsten pinned to `feat/idle-flush` (1c5ed3d at the time), plus the step-1 fix on a branch on top of it.
- The legacy queue stays behind `--message-queue legacy` after the comparison, for reproducing earlier results
  (step 7 removes only katric's own grid adapter once BriefKAsten's grid replaces it, and the transition pins).

## Open decisions

- Default buffer size C: 4 KiB (decided 2026-11-25 after step 6, as KaCCv2).
