//
// Created by Tim Niklas Uhl on 13.08.20.
//

#ifndef TRIANGLE_COUNTER_GRAPH_IO_H
#define TRIANGLE_COUNTER_GRAPH_IO_H

#include <sstream>
#include <fstream>
#include <datastructures/graph_definitions.h>
#include "config.h"

namespace cetric {
    using namespace cetric::graph;

template<typename HeaderFunc, typename NodeFunc, typename EdgeFunc>
void read_metis(const std::string& input, HeaderFunc on_header, NodeFunc on_node, EdgeFunc on_edge) {
    std::ifstream stream(input);
    if (stream.fail()) {
        throw std::runtime_error("Could not open input file for reading: " + input);
    }
    std::string line;
    std::getline(stream, line);
    std::istringstream sstream(line);
    NodeId node_count;
    EdgeId edge_count;
    sstream >> node_count >> edge_count;
    if (sstream.bad()) {
        throw std::runtime_error("Failed to parse header.");
    }
    on_header(node_count, edge_count);


    size_t line_number = 1;
    NodeId node = 0;
    EdgeId edge_id = 0;
    while (node < node_count && std::getline(stream, line)) {
        sstream = std::istringstream(line);
        // skip comment lines
        if (line.rfind('%', 0) == 0) {
            line_number++;
            continue;
        }
        on_node(node);
        NodeId head_node;
        while (sstream >> head_node) {
            if (head_node >= node_count + 1) {
                throw std::runtime_error("Invalid node id " + std::to_string(head_node) + " in line " + std::to_string(line_number) + ".");
            }
            on_edge(Edge(node, head_node - 1));
            edge_id++;
        }
        if (sstream.bad()) {
            throw std::runtime_error("Invalid input in line " + std::to_string(line_number) + ".");
        }
        node++;
        line_number++;
    }
    if (node != node_count) {
        throw std::runtime_error("Number of nodes does not match header.");
    }
    if (edge_id != edge_count * 2) {
        std::stringstream out;
        out << "Number of edges does not mach header (header: " << edge_count << ", actual: " << edge_id << ")";
        throw std::runtime_error(out.str());
    }
}

inline void read_metis_header(const std::string& input, cetric::graph::NodeId& node_count, cetric::graph::EdgeId& edge_count) {
    std::ifstream stream(input);
    if (stream.fail()) {
        throw std::runtime_error("Could not open input file for reading: " + input);
    }
    std::string line;
    std::getline(stream, line);
    std::istringstream sstream(line);
    sstream >> node_count >> edge_count;
    if (sstream.bad()) {
        throw std::runtime_error("Failed to parse header.");
    }
}

inline void read_metis(const std::string& input, std::vector<cetric::graph::EdgeId>& first_out, std::vector<cetric::graph::NodeId>& head) {
    cetric::graph::NodeId node_count;
    cetric::graph::EdgeId edge_count;
    first_out.clear();
    head.clear();
    cetric::graph::EdgeId edge_id = 0;
    auto on_header = [&](NodeId node_count_, EdgeId edge_count_) {
        node_count = node_count_;
        edge_count = edge_count_ * 2;
        first_out.resize(node_count + 1);
        head.resize(edge_count);
    };
    auto on_node = [&](NodeId node) {
        first_out[node] = edge_id;
    };
    auto on_edge = [&](Edge edge) {
        head[edge_id] = edge.head;
        edge_id++;
    };
    read_metis(input, on_header, on_node, on_edge);
    first_out[node_count] = edge_id;
}


template<typename EdgeFunc>
void read_edge_list(const std::string& input, EdgeFunc on_edge, 
        cetric::graph::NodeId starts_at = 0, std::string edge_prefix="", std::string comment_prefix="#") {
    std::ifstream stream(input);
    if (stream.fail()) {
        throw std::runtime_error("Could not open input file for reading: " + input);
    }

    std::string line;
    size_t line_number = 0;
    while (std::getline(stream, line)) {
        if (line.rfind(comment_prefix, 0) == 0) {
            line_number++;
            continue;
        }
        if (line.rfind(edge_prefix, 0) == 0) {
            line = line.substr(edge_prefix.size());
            std::istringstream sstream(line);
            cetric::graph::NodeId tail;
            cetric::graph::NodeId head;
            sstream >> tail >> head;
            if (sstream.bad()) {
                throw std::runtime_error("Invalid input in line " + std::to_string(line_number) + ".");
            }
            on_edge(Edge(tail - starts_at, head - starts_at));
        }
        ++line_number;
    }
}

inline void read_edge_list(const std::string& input, std::vector<cetric::graph::EdgeId>& first_out, std::vector<cetric::graph::NodeId>& head) {
    std::vector<cetric::graph::Edge> edges;
    cetric::graph::NodeId max_node_id = 0;
    cetric::graph::Edge previous_edge;
    bool sorted = true;
    auto on_edge  = [&](cetric::graph::Edge edge) {
        max_node_id = std::max({max_node_id, edge.tail, edge.head});
        edges.push_back(edge);
        if (std::tie(previous_edge.tail, previous_edge.head) > std::tie(edge.tail, edge.head)) {
           sorted = false;
        }
        previous_edge = edge;
    };
    read_edge_list(input, on_edge);
    NodeId node_count = max_node_id + 1;

    if (!sorted) {
        std::sort(edges.begin(), edges.end(), [&](Edge a, Edge b) {
            return std::tie(a.tail, a.head) < std::tie(b.tail, b.head);
        });
    }

    first_out.clear();
    first_out.reserve(node_count + 1);
    first_out.push_back(0);
    head.clear();
    head.reserve(edges.size());
    NodeId current_node = 0;

    for(const Edge& edge : edges) {
        while (current_node != edge.tail) {
            first_out.push_back(head.size());
            ++current_node;
        }
        head.push_back(edge.head);
    }
    while (current_node != node_count) {
        first_out.push_back(head.size());
        ++current_node;
    }

}
}
#endif //TRIANGLE_COUNTER_GRAPH_IO_H
