//
// Created by Tim Niklas Uhl on 26.10.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_CONFIG_H
#define PARALLEL_TRIANGLE_COUNTER_CONFIG_H

#include <limits>
#include <optional>

#include <cereal/cereal.hpp>
#include <cereal/types/optional.hpp>
#include <graph-io/definitions.h>
#include <graph-io/gen_parameters.h>
#include <magic_enum.hpp>

#include "cetric/datastructures/graph_definitions.h"
#include "cetric/util.h"

namespace cetric {

template <
    class Archive,
    cereal::traits::EnableIf<cereal::traits::is_text_archive<Archive>::value> = cereal::traits::sfinae,
    class T>
std::enable_if_t<std::is_enum_v<T>, std::string> save_minimal(Archive&, const T& h) {
    return std::string(magic_enum::enum_name(h));
}

template <
    class Archive,
    cereal::traits::EnableIf<cereal::traits::is_text_archive<Archive>::value> = cereal::traits::sfinae,
    class T>
std::enable_if_t<std::is_enum_v<T>, void> load_minimal(Archive const&, T& enumType, std::string const& str) {
    enumType = magic_enum::enum_cast<T>(str).value();
}

enum class CacheInput { Filesystem, InMemory, None };
static const std::map<std::string, CacheInput> cache_input_map{
    {"none", CacheInput::None}, {"fs", CacheInput::Filesystem}, {"mem", CacheInput::InMemory}};
template <class Archive>
void load_minimal(const Archive& ar [[maybe_unused]], CacheInput& cache_input, const std::string& value) {
    cache_input = cache_input_map.at(value);
}

template <class Archive>
std::string save_minimal(const Archive& ar [[maybe_unused]], const CacheInput& cache_input) {
    for (const auto& kv: cache_input_map) {
        if (kv.second == cache_input) {
            return kv.first;
        }
    }
    return "";
}
enum class Algorithm { Patric, Cetric, CetricX };
static const std::map<std::string, Algorithm> algorithm_map{
    {"patric", Algorithm::Patric}, {"cetric", Algorithm::Cetric}, {"cetric-x", Algorithm::CetricX}};

template <class Archive>
void load_minimal(const Archive& ar [[maybe_unused]], Algorithm& algorithm, const std::string& value) {
    algorithm = algorithm_map.at(value);
}

template <class Archive>
std::string save_minimal(const Archive& ar [[maybe_unused]], const Algorithm& algorithm) {
    for (const auto& kv: algorithm_map) {
        if (kv.second == algorithm) {
            return kv.first;
        }
    }
    return "";
}

enum class Threshold { local_nodes, local_edges, none };
static const std::map<std::string, Threshold> threshold_map{
    {"local-nodes", Threshold::local_nodes}, {"local-edges", Threshold::local_edges}, {"none", Threshold::none}};

template <class Archive>
void load_minimal(const Archive& ar [[maybe_unused]], Threshold& threshold, const std::string& value) {
    threshold = threshold_map.at(value);
}

template <class Archive>
std::string save_minimal(const Archive& ar [[maybe_unused]], const Threshold& threshold) {
    for (const auto& kv: threshold_map) {
        if (kv.second == threshold) {
            return kv.first;
        }
    }
    return "";
}

enum class IntersectionMethod { merge, binary_search, binary, hybrid };
static const std::map<std::string, IntersectionMethod> intersection_method_map{
    {"merge", IntersectionMethod::merge},
    {"binary-search", IntersectionMethod::binary_search},
    {"binary", IntersectionMethod::binary},
    {"hybrid", IntersectionMethod::hybrid}};

template <class Archive>
void load_minimal(const Archive& ar [[maybe_unused]], IntersectionMethod& method, const std::string& value) {
    method = intersection_method_map.at(value);
}

template <class Archive>
std::string save_minimal(const Archive& ar [[maybe_unused]], const IntersectionMethod& method) {
    for (const auto& kv: intersection_method_map) {
        if (kv.second == method) {
            return kv.first;
        }
    }
    return "";
}
enum class ParallelizationMethod { tbb, omp_for, omp_task };
enum class OMPSchedule { stat, dynamic, guided, standard };
enum class TBBPartitioner { stat, simple, standard, affinity };

struct Config {
    Config() = default;
    std::string           input_file;
    std::string           output_file;
    bool                  read_edge_partitioned = false;
    graphio::InputFormat  input_format;
    std::string           partitioning;
    bool                  partitioned_input                 = false;
    size_t                buffer_threshold                  = std::numeric_limits<size_t>::max();
    double                max_degree_threshold_alpha        = 1.0;
    bool                  empty_pending_buffers_on_overflow = false;
    size_t                iterations                        = 1;
    std::string           primary_cost_function             = "N";
    std::string           secondary_cost_function           = "none";
    Algorithm             algorithm                         = Algorithm::Cetric;
    Threshold             threshold                         = Threshold::local_nodes;
    double                threshold_scale                   = 1.0;
    double                high_degree_threshold_scale       = 1.0;
    std::string           communication_policy              = "new";
    bool                  local_parallel                    = false;
    bool                  global_parallel                   = false;
    bool                  parallel_compact                  = false;
    ParallelizationMethod parallelization_method            = ParallelizationMethod::tbb;
    bool                  edge_partitioning                 = false;
    bool                  edge_partitioning_static          = false;
    TBBPartitioner        tbb_partitioner                   = TBBPartitioner::standard;
    OMPSchedule           omp_schedule                      = OMPSchedule::standard;
    size_t                omp_chunksize                     = 0;
    size_t                local_degree_of_parallelism       = 1;
    size_t                global_degree_of_parallelism      = 1;
    IntersectionMethod    intersection_method               = IntersectionMethod::merge;
    size_t                binary_intersection_cutoff        = 1000;
    double                hybrid_cutoff_scale               = 1.0;

    bool dense_degree_exchange   = false;
    bool compact_degree_exchange = false;
    bool global_synchronization  = false;
    bool binary_rank_search      = false;
    bool id_node_ordering = false;

    bool degree_filtering        = false;
    bool orient_locally          = false;
    bool pseudo2core             = false;
    bool dense_load_balancing    = false;
    bool flag_intersection       = false;
    bool skip_local_neighborhood = false;
    bool synchronized            = false;

    int         verbosity_level = 0;
    CacheInput  cache_input     = CacheInput::None;
    std::string json_output     = "";
    std::string hostname;
    PEID        PEs;
    PEID        rank;
    size_t      num_threads = 1;
    size_t      grainsize   = 1;

    std::string git_commit;

    // Generator parameters
    graphio::GeneratorParameters gen;

    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            CEREAL_NVP(input_file),
            CEREAL_NVP(read_edge_partitioned),
            CEREAL_NVP(partitioning),
            CEREAL_NVP(partitioned_input),
            CEREAL_NVP(hostname),
            CEREAL_NVP(PEs),
            CEREAL_NVP(num_threads),
            CEREAL_NVP(grainsize),
            CEREAL_NVP(cache_input),
            CEREAL_NVP(algorithm),
            CEREAL_NVP(communication_policy),
            CEREAL_NVP(global_parallel),
            CEREAL_NVP(local_parallel),
            CEREAL_NVP(parallel_compact),
            CEREAL_NVP(parallelization_method),
            CEREAL_NVP(edge_partitioning),
            CEREAL_NVP(edge_partitioning_static),
            CEREAL_NVP(tbb_partitioner),
            CEREAL_NVP(omp_schedule),
            CEREAL_NVP(omp_chunksize),
            CEREAL_NVP(threshold),
            CEREAL_NVP(local_degree_of_parallelism),
            CEREAL_NVP(global_degree_of_parallelism),
            CEREAL_NVP(intersection_method),
            CEREAL_NVP(binary_intersection_cutoff),
            CEREAL_NVP(hybrid_cutoff_scale),
            CEREAL_NVP(dense_degree_exchange),
            CEREAL_NVP(compact_degree_exchange),
            CEREAL_NVP(global_synchronization),
            CEREAL_NVP(binary_rank_search),
            CEREAL_NVP(id_node_ordering),
            CEREAL_NVP(threshold_scale),
            CEREAL_NVP(high_degree_threshold_scale),
            CEREAL_NVP(primary_cost_function),
            CEREAL_NVP(secondary_cost_function),
            CEREAL_NVP(orient_locally),
            CEREAL_NVP(pseudo2core),
            CEREAL_NVP(dense_load_balancing),
            CEREAL_NVP(flag_intersection),
            CEREAL_NVP(skip_local_neighborhood),
            CEREAL_NVP(git_commit)
        );
        if (input_file.empty()) {
            archive(
                cereal::make_nvp("gen", gen.generator),                 //
                cereal::make_nvp("gen_n", gen.n),                       //
                cereal::make_nvp("gen_m", gen.m),                       //
                cereal::make_nvp("gen_r", gen.r),                       //
                cereal::make_nvp("gen_p", gen.p),                       //
                cereal::make_nvp("gen_gamma", gen.gamma),               //
                cereal::make_nvp("gen_d", gen.d),                       //
                cereal::make_nvp("gen_a", gen.a),                       //
                cereal::make_nvp("gen_b", gen.b),                       //
                cereal::make_nvp("gen_c", gen.c),                       //
                cereal::make_nvp("gen_seed", gen.seed),                 //
                cereal::make_nvp("gen_verify_graph", gen.verify_graph), //
                cereal::make_nvp("gen_statistics", gen.statistics)      //
            );
        }
    }
};
} // namespace cetric

#endif // PARALLEL_TRIANGLE_COUNTER_CONFIG_H
