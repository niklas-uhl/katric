// The tags of katric's point-to-point exchanges.
#pragma once

namespace katric {

/// Tags of the point-to-point exchanges (kamping-v2 and dSTL take scoped enums
/// as tags). Consecutive sparse (NBX) exchanges need distinct tags: a rank that
/// has left one may already send for the next while another rank still probes
/// for the first.
enum class MessageTag {
  CostFunction = 0,  // ghost degrees for the cost function
  CostFunctionOutDegree =
      1,  // ghost out-degrees for the cost function, right after the degrees
  LoadBalancing = 5,
  Orientation = 31,
};

}  // namespace katric
