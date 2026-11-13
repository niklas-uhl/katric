// Choosing between sequential and multithreaded (TBB) execution from the config.
#pragma once

#include <cstddef>
#include <utility>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include "cetric/config.h"
#include "cetric/util.h"

namespace cetric {

/// Calls `f` with the execution policy: parallel on `conf.num_threads` threads if `parallel`, else sequential. `f`
/// gets the policy by value and should pass it on as an rvalue (`std::move`): the graph's methods compare the policy
/// type exactly.
template <typename F>
decltype(auto) with_execution_policy(Config const& conf, bool parallel, F&& f) {
    if (parallel) {
        return f(execution_policy::parallel{conf.num_threads});
    }
    return f(execution_policy::sequential{});
}

/// with_execution_policy(), parallel if there is more than one thread.
template <typename F>
decltype(auto) with_execution_policy(Config const& conf, F&& f) {
    return with_execution_policy(conf, conf.num_threads > 1, std::forward<F>(f));
}

/// Calls `f(node)` for every local vertex of `G`, in a task arena of `conf.num_threads` threads if `parallel`.
template <typename Graph, typename F>
void for_each_local_node(Graph& G, Config const& conf, bool parallel, F&& f) {
    auto nodes = G.local_nodes();
    if (!parallel) {
        for (auto node: nodes) {
            f(node);
        }
        return;
    }
    tbb::task_arena arena(conf.num_threads);
    arena.execute([&] {
        tbb::parallel_for(tbb::blocked_range(nodes.begin(), nodes.end()), [&](auto const& r) {
            for (auto node: r) {
                f(node);
            }
        });
    });
}

} // namespace cetric
