//
// Created by Tim Niklas Uhl on 26.10.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_CONFIG_H
#define PARALLEL_TRIANGLE_COUNTER_CONFIG_H

#include <datastructures/graph_definitions.h>
#include <io/definitions.h>
#include <util.h>
#include <cereal/cereal.hpp>
#include <cereal/types/optional.hpp>
#include <limits>
#include <optional>

namespace cetric {
enum class CacheInput { Filesystem, InMemory, None };
static const std::map<std::string, CacheInput> cache_input_map{{"none", CacheInput::None},
                                                               {"fs", CacheInput::Filesystem},
                                                               {"mem", CacheInput::InMemory}};
template <class Archive>
void load_minimal(const Archive& ar [[maybe_unused]], CacheInput& cache_input, const std::string& value) {
    cache_input = cache_input_map.at(value);
}

template <class Archive>
std::string save_minimal(const Archive& ar [[maybe_unused]], const CacheInput& cache_input) {
    for (const auto& kv : cache_input_map) {
        if (kv.second == cache_input) {
            return kv.first;
        }
    }
    return "";
}
enum class Algorithm { Patric, Cetric };
static const std::map<std::string, Algorithm> algorithm_map{{"patric", Algorithm::Patric},
                                                            {"cetric", Algorithm::Cetric}};

template <class Archive>
void load_minimal(const Archive& ar [[maybe_unused]], Algorithm& algorithm, const std::string& value) {
    algorithm = algorithm_map.at(value);
}

template <class Archive>
std::string save_minimal(const Archive& ar [[maybe_unused]], const Algorithm& algorithm) {
    for (const auto& kv : algorithm_map) {
        if (kv.second == algorithm) {
            return kv.first;
        }
    }
    return "";
}

enum class Threshold { local_nodes, local_edges, none };
static const std::map<std::string, Threshold> threshold_map{{"local-nodes", Threshold::local_nodes},
                                                            {"local-edges", Threshold::local_edges},
                                                            {"none", Threshold::none}};

template <class Archive>
void load_minimal(const Archive& ar [[maybe_unused]], Threshold& threshold, const std::string& value) {
    threshold = threshold_map.at(value);
}

template <class Archive>
std::string save_minimal(const Archive& ar [[maybe_unused]], const Threshold& threshold) {
    for (const auto& kv : threshold_map) {
        if (kv.second == threshold) {
            return kv.first;
        }
    }
    return "";
}

struct Config {
    Config() = default;
    std::string input_file;
    std::string output_file;
    InputFormat input_format;
    size_t seed = 28475421;
    size_t buffer_threshold = std::numeric_limits<size_t>::max();
    double max_degree_threshold_alpha = 1.0;
    bool empty_pending_buffers_on_overflow = false;
    size_t iterations = 1;
    std::string primary_cost_function = "N";
    std::string secondary_cost_function = "none";
    Algorithm algorithm = Algorithm::Cetric;
    Threshold threshold = Threshold::local_nodes;
    double threshold_scale = 1.0;
    std::string communication_policy = "new";
    bool local_parallel = false;
    bool global_parallel = false;
    size_t degree_of_parallelism = 1;

    bool full_all_to_all = false;

    bool degree_filtering = false;
    bool orient_locally = false;
    bool pseudo2core = false;
    bool dense_load_balancing = false;
    bool flag_intersection = false;
    bool skip_local_neighborhood = false;
    bool synchronized = false;

    int verbosity_level = 0;
    CacheInput cache_input = CacheInput::None;
    std::string json_output = "";
    std::string hostname;
    PEID PEs;
    PEID rank;
    size_t num_threads = 0;
    size_t grainsize = 1;

    // Generator parameters
    std::string gen;
    cetric::graph::NodeId gen_n = 10;
    cetric::graph::EdgeId gen_m = 0;
    float gen_r = 0.125;
    float gen_r_coeff = 0.55;
    float gen_p = 0.0;
    bool gen_periodic = false;
    size_t gen_k = 0;
    float gen_gamma = 2.8;
    float gen_d = 16;
    bool gen_scale_weak = false;
    bool rhg_fix = false;
    double false_positive_rate = 0.01;

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(input_file), CEREAL_NVP(hostname), CEREAL_NVP(PEs), CEREAL_NVP(num_threads), CEREAL_NVP(grainsize),
                CEREAL_NVP(cache_input), CEREAL_NVP(algorithm), CEREAL_NVP(communication_policy),
                CEREAL_NVP(global_parallel), CEREAL_NVP(local_parallel), CEREAL_NVP(threshold),
                CEREAL_NVP(threshold_scale), CEREAL_NVP(primary_cost_function), CEREAL_NVP(secondary_cost_function),
                CEREAL_NVP(orient_locally), CEREAL_NVP(pseudo2core), CEREAL_NVP(dense_load_balancing),
                CEREAL_NVP(flag_intersection), CEREAL_NVP(skip_local_neighborhood));
        if (input_file.empty()) {
            archive(CEREAL_NVP(gen), CEREAL_NVP(gen_n), CEREAL_NVP(gen_m), CEREAL_NVP(gen_r), CEREAL_NVP(gen_r_coeff),
                    CEREAL_NVP(gen_p), CEREAL_NVP(gen_gamma), CEREAL_NVP(gen_d), CEREAL_NVP(gen_scale_weak));
        }
    }
};
}  // namespace cetric

#endif  // PARALLEL_TRIANGLE_COUNTER_CONFIG_H
