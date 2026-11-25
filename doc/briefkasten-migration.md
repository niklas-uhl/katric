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

## Decisions (2026-11-24)

- katric moves to C++23 (no range-v3).
- BriefKAsten pinned to `feat/idle-flush` (1c5ed3d at the time), plus the step-1 fix on a branch on top of it.
- The legacy queue stays behind `--message-queue legacy` after the comparison, for reproducing earlier results
  (step 7 removes only katric's own grid adapter once BriefKAsten's grid replaces it, and the transition pins).

## Open decisions

- Default buffer size C: 4 KiB (decided 2026-11-25 after step 6, as KaCCv2).
