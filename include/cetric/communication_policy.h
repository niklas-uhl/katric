#pragma once

namespace cetric {
/// Tags selecting how the global phase sends neighborhoods (`--communication-policy`): `new` sends directly through
/// the buffered message queue, `grid` routes through a 2D grid of indirections.
struct MessageQueuePolicy {};
struct GridPolicy {};
} // namespace cetric
