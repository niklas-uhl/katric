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
4. **Options.** `--message-queue legacy|briefkasten` for the transition (legacy keeps `--threshold` /
   `--threshold-scale`); for BriefKAsten `--buffer-size` (local threshold, bytes), `--request-slots`,
   `--flow-control` / `--no-flow-control`, `--receiver persistent|probe|posted`, `--idle-flush`. Suites that compare
   with earlier data pin `message-queue: legacy`, as for the defaults on 2026-11-23.
5. **Local validation.** Identical triangle counts on the refcheck inputs (DiTriC / CETRIC, trimming on / off,
   direct / grid, 2-8 ranks), plus a tiny buffer (e.g. 64 B) so every message takes the oversized path, and
   `--no-flow-control`. One small run per variant, not the whole test suite.
6. **Cluster suite `briefkasten`.** Permuted RGG N18 and GNM N16 (then permuted RHG, RMAT): legacy direct / grid vs
   BriefKAsten direct / grid x C in {4, 16, 64} KiB, defaults otherwise, 3 iterations; cores set at job creation
   (192-3072 shows the trend). Questions: does the global phase stay flat with p; polling cost per message; which C;
   where grid overtakes direct now.
7. **Default BriefKAsten** after the comparison; the legacy queue (with katric's grid adapter, which it needs) stays
   behind `--message-queue legacy` for reproducing earlier results.

## Decisions (2026-11-24)

- katric moves to C++23 (no range-v3).
- BriefKAsten pinned to `feat/idle-flush` (1c5ed3d at the time), plus the step-1 fix on a branch on top of it.
- The legacy queue stays behind `--message-queue legacy` after the comparison, for reproducing earlier results
  (step 7 removes only katric's own grid adapter once BriefKAsten's grid replaces it, and the transition pins).

## Open decisions

- Default buffer size C (after step 6); KaCCv2 uses 4 KiB.
