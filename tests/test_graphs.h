// Test inputs read with KaGen: the whole graph on every rank (the expected result) and its distributed local part.
#pragma once

#include <ostream>
#include <string>
#include <vector>

#include <kagen.h>
#include <mpi.h>

#include "katric/graph/graph.h"
#include "katric/input.h"

namespace katric::testing {

/// An edge of the whole graph by global ids.
struct Edge {
    graph::VertexId tail;
    graph::VertexId head;

    friend bool operator==(Edge const&, Edge const&) = default;
    friend std::ostream& operator<<(std::ostream& out, Edge const& edge) {
        return out << "(" << edge.tail << ", " << edge.head << ")";
    }
};

inline std::string metis_option_string(std::string const& metis_file) {
    return "file;filename=" + metis_file + ";input_format=metis";
}

/// The whole graph as one edge list per node, read independently on each rank.
inline std::vector<std::vector<Edge>> read_full_graph(std::string const& metis_file) {
    kagen::Graph graph  = generate_input(metis_option_string(metis_file), MPI_COMM_SELF);
    auto const&  xadj   = graph.xadj;
    auto const&  adjncy = graph.adjncy;
    std::vector<std::vector<Edge>> edges(xadj.size() - 1);
    for (graph::VertexId tail = 0; tail + 1 < xadj.size(); ++tail) {
        for (auto e = xadj[tail]; e < xadj[tail + 1]; ++e) {
            edges[tail].emplace_back(tail, adjncy[e]);
        }
    }
    return edges;
}

/// The local part of the graph, distributed over MPI_COMM_WORLD with KaGen's default distribution.
inline kagen::Graph read_local_graph(std::string const& metis_file) {
    return generate_input(metis_option_string(metis_file), MPI_COMM_WORLD);
}

} // namespace katric::testing
