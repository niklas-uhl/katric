/*
 * Copyright (c) 2020-2023 Tim Niklas Uhl
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <cstdint>

#include <caliper/ConfigManager.h>
#include <caliper/cali.h>

namespace cetric::profiling {

/// Adds `value` to the counter `name` in the currently open Caliper region, the equivalent of kamping's
/// `counter().add(name, value)`. Counters need no declaration; the `katric-counters` config reports them per region
/// with min/max/sum across ranks. Each call takes a snapshot, so call it once per phase (not in hot loops) and from
/// the thread that opened the regions.
inline void counter_add(char const* name, std::int64_t value) {
    // One generic name/value pair instead of an attribute per counter: the aggregation keys on the name, so counters
    // can be created on the fly. (Attributes created after a region's first snapshot would be ignored there.)
    static cali_id_t const attributes[] = {
        cali_create_attribute("counter.name", CALI_TYPE_STRING, CALI_ATTR_SKIP_EVENTS),
        cali_create_attribute(
            "counter.value",
            CALI_TYPE_INT,
            CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE | CALI_ATTR_SKIP_EVENTS
        ),
    };
    cali_variant_t const values[] = {cali_make_variant_from_string(name), cali_make_variant_from_int64(value)};
    cali_push_snapshot(CALI_SCOPE_THREAD | CALI_SCOPE_PROCESS, 2, attributes, values);
}

/// `katric-timers`: inclusive region times in seconds (min/max/avg across ranks), the maximum exclusive time and the
/// number of calls per region path, written by rank 0 as JSON.
inline constexpr char const* timers_config_spec = R"json(
{
 "name": "katric-timers",
 "description": "Region times aggregated across ranks, as JSON",
 "categories": ["output"],
 "services": ["aggregate", "event", "timer", "mpireport", "adiak_import"],
 "config": {
  "CALI_CHANNEL_FLUSH_ON_EXIT": "false",
  "CALI_MPIREPORT_LOCAL_CONFIG":
   "let t=scale(sum#time.duration.ns,1e-9) select inclusive_sum(t),sum(t),sum(sum#region.count) group by prop:nested",
  "CALI_MPIREPORT_CONFIG":
   "select *,min(inclusive#t) as min,max(inclusive#t) as max,avg(inclusive#t) as avg,max(sum#t) as exclusive_max,max(sum#sum#region.count) as calls group by prop:nested format json(object)"
 },
 "defaults": { "output": "katric.timers.json" }
}
)json";

/// `katric-counters`: the counter_add() values per region path and counter (min/max/sum across ranks), written by
/// rank 0 as JSON. A separate channel, as keying spot's profile on the counter name would split the region times.
inline constexpr char const* counters_config_spec = R"json(
{
 "name": "katric-counters",
 "description": "Counters aggregated across ranks, as JSON",
 "categories": ["output"],
 "services": ["aggregate", "mpireport", "adiak_import"],
 "config": {
  "CALI_CHANNEL_FLUSH_ON_EXIT": "false",
  "CALI_MPIREPORT_LOCAL_CONFIG": "select sum(sum#counter.value) group by prop:nested,counter.name",
  "CALI_MPIREPORT_CONFIG":
   "select *,min(sum#sum#counter.value) as min,max(sum#sum#counter.value) as max,sum(sum#sum#counter.value) as sum group by prop:nested,counter.name format json(object)"
 },
 "defaults": { "output": "katric.counters.json" }
}
)json";

/// Registers `katric-timers` and `katric-counters` with `manager`. With a non-empty `output_prefix`, their output
/// files (and spot's) default to `<prefix>.timers.json`, `<prefix>.counters.json` and `<prefix>.cali`.
inline void add_katric_configs(cali::ConfigManager& manager, std::string const& output_prefix) {
    manager.add_config_spec(timers_config_spec);
    manager.add_config_spec(counters_config_spec);
    if (!output_prefix.empty()) {
        manager.set_default_parameter_for_config("katric-timers", "output", (output_prefix + ".timers.json").c_str());
        manager.set_default_parameter_for_config("katric-counters", "output", (output_prefix + ".counters.json").c_str());
        manager.set_default_parameter_for_config("spot", "output", (output_prefix + ".cali").c_str());
    }
}

} // namespace cetric::profiling
