#ifndef DEFINITIONS_H_DZ2WBS3H
#define DEFINITIONS_H_DZ2WBS3H

#include <map>
#include <string>

enum class InputFormat {
    metis,
    binary,
    partitioned_edgelist_dimacs,
    edge_list
};

const std::map<std::string, InputFormat> input_types = {
    {"metis", InputFormat::metis},
    {"binary", InputFormat::binary},
    {"partitioned-dimacs", InputFormat::partitioned_edgelist_dimacs},
    {"edge-list", InputFormat::edge_list}
};

#endif /* end of include guard: DEFINITIONS_H_DZ2WBS3H */
