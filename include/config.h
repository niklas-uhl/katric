//
// Created by Tim Niklas Uhl on 26.10.20.
//

#ifndef PARALLEL_TRIANGLE_COUNTER_CONFIG_H
#define PARALLEL_TRIANGLE_COUNTER_CONFIG_H

#include <limits>
#include <datastructures/graph_definitions.h>
#include <util.h>
#include <io/definitions.h>
#include <nlohmann/json.hpp>

struct Config {
    Config() = default;
    bool sort_neighborhood = false;
    bool prune_intersection = false;
    size_t colors = 2;
    bool prune_kernelization = false;
    double kernelization_treshold = 0.1;
    bool use_hashtable = false;
    bool use_binary_search = false;
    bool use_hash_coloring = false;
    bool shrink_neighborhood = false;
    std::string input_file;
    std::string output_file;
    InputFormat input_format;
    size_t seed = 28475421;
    bool lcc = false;
    size_t bloom_k = 0;
    size_t buffer_threshold = std::numeric_limits<size_t>::max();
    double max_degree_threshold_alpha = 1.0;
    bool empty_pending_buffers_on_overflow = false;
    size_t iterations = 1;
    std::string primary_cost_function = "N";
    std::string secondary_cost_function = "";

    bool full_all_to_all = false;
    bool use_two_phases = false;

    bool degree_filtering = false;

    int verbosity_level = 0;
    bool json_output = false;
    std::string hostname;
    PEID PEs;
    PEID rank;

    // Generator parameters
    std::string gen;
    cetric::graph::NodeId gen_n = 100;
    cetric::graph::EdgeId gen_m = 0;
    float gen_r = 0.125;
    float gen_r_coeff = 0.55;
    float gen_p = 0.0;
    bool gen_periodic = false;
    size_t gen_k = 0;
    float gen_gamma = 0;
    float gen_d =0;
    bool gen_scale_weak = false;
    bool rhg_fix = false;
    double false_positive_rate = 0.01;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Config, input_file, hostname, PEs, gen, gen_n, gen_m, gen_r, gen_r_coeff, gen_p, gen_gamma, gen_d, gen_scale_weak, primary_cost_function, secondary_cost_function)

#endif //PARALLEL_TRIANGLE_COUNTER_CONFIG_H
