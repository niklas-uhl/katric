#include <io/distributed_graph_io.h>
#include <limits>
#pragma GCC diagnostic push
#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#pragma push_macro("PTR")
#undef PTR
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include "kagen_interface.h"
#undef BOOST_BIND_GLOBAL_PLACEHOLDERS
#pragma pop_macro("PTR")
#pragma GCC diagnostic pop


namespace cetric {

void read_metis_distributed(const std::string& input, const GraphInfo& graph_info, std::vector<Edge>& edge_list, node_set& ghosts, PEID rank, PEID size) {
    (void) rank;
    (void) size;

    auto on_head = [](NodeId, EdgeId) {};

    bool skip = true;
    auto on_node = [&](NodeId node) {
        if (node >= graph_info.local_from) {
            skip = false;
        }
        if (node >= graph_info.local_to) {
            skip = true;
        }
    };

    auto on_edge = [&](Edge edge) {
        if (!skip) {
            if (edge.head < graph_info.local_from || edge.head >= graph_info.local_to) {
                ghosts.insert(edge.head);
            }
            edge_list.emplace_back(edge);
        }
    };

    read_metis(input, on_head, on_node, on_edge);
}

void read_metis_distributed(const std::string& input, const GraphInfo& graph_info, std::vector<EdgeId>& first_out, std::vector<NodeId>& head, PEID rank, PEID size) {
    (void) rank;
    (void) size;

    auto on_head = [](NodeId, EdgeId) {};

    bool skip = true;
    auto on_node = [&](NodeId node) {
        if (node >= graph_info.local_from && node < graph_info.local_to) {
            first_out.emplace_back(head.size());
            skip = false;
        } else {
            skip = true;
        }
    };

    auto on_edge = [&](Edge edge) {
        if (!skip) {
            head.emplace_back(edge.head);
        }
    };

    read_metis(input, on_head, on_node, on_edge);
    first_out.emplace_back(head.size());
}

void gather_PE_ranges(NodeId local_from, NodeId local_to, std::vector<std::pair<NodeId, NodeId>>& ranges, const MPI_Comm& comm, PEID rank, PEID size) {
    (void) rank;
    (void) size;
    MPI_Datatype MPI_RANGE;
    MPI_Type_vector(1, 2, 0, MPI_NODE, &MPI_RANGE);
    MPI_Type_commit(&MPI_RANGE);
    std::pair<NodeId, NodeId> local_range(local_from, local_to);
    MPI_Allgather(&local_range, 1, MPI_RANGE, ranges.data(), 1, MPI_RANGE, comm);
#ifdef CHECK_RANGES
    if (rank == 0) {
        NodeId next_expected = 0;
        for (size_t i = 0; i < ranges.size(); ++i) {
            std::pair<NodeId, NodeId> &range = ranges[i];
            if (range.first == range.second) {
                continue;
            }
            if (range.first > range.second) {
                throw std::runtime_error("[R" + std::to_string(i) + "] range [" + std::to_string(range.first) + ", " + std::to_string(range.second) + "] is invalid");
            }
            if (range.first > next_expected) {
                throw std::runtime_error("[R" + std::to_string(i) + "] range [" + std::to_string(range.first) + ", " + std::to_string(range.second) + "] has a gap to previous one");
            }
            if (range.first < next_expected) {
                throw std::runtime_error("[R" + std::to_string(i) + "] range [" + std::to_string(range.first) + ", " + std::to_string(range.second) + "] overlaps with previous one");
            }
            next_expected = range.second;
        }
    }
#endif
}

PEID get_PE_from_node_ranges(NodeId node, const std::vector<std::pair<NodeId, NodeId>>& ranges) {
    NodeId local_from;
    NodeId local_to;
    for (size_t i = 0; i < ranges.size(); ++i) {
        std::tie(local_from, local_to) = ranges[i];
        if (local_from <= node && node <= local_to) {
            return i;
        }
    }
    std::stringstream out;
    out << "Node " << node << " not assigned to any PE";
    throw std::runtime_error(out.str());
}

LocalGraphView gen_local_graph(const Config& conf_, PEID rank, PEID size) {
    Config conf = conf_;
    NodeId n_pow_2 = 1 << conf.gen_n;
    conf.gen_r = conf.gen_r_coeff * sqrt(log(n_pow_2) / n_pow_2);
    if (conf.gen_scale_weak) {
        conf.gen_r /= sqrt(size);
    }
    kagen::EdgeList edge_list;
    node_set ghosts;
    ghosts.set_empty_key(-1);
    std::vector<std::pair<NodeId, NodeId>> ranges(size);

    kagen::KaGen gen(rank, size);
    kagen::SInt n = static_cast<kagen::SInt>(1) << conf.gen_n;
    kagen::SInt m = static_cast<kagen::SInt>(1) << conf.gen_m;
    if (conf.gen == "gnm_undirected") {
        if (m * 2 > (n * n) / 2) {
            throw std::runtime_error("m is to high");
        }
        edge_list = gen.GenerateUndirectedGNM(n, m, conf.gen_k, conf.seed);
    } else if (conf.gen == "rdg_2d") {
        edge_list = gen.Generate2DRDG(n, conf.gen_k, conf.seed);
    } else if (conf.gen == "rdg_3d") {
        edge_list = gen.Generate3DRDG(n, conf.gen_k, conf.seed);
    } else if (conf.gen == "rgg_2d") {
        edge_list = gen.Generate2DRGG(n, conf.gen_r, conf.gen_k, conf.seed);
    } else if (conf.gen == "rgg_3d") {
        edge_list = gen.Generate3DRGG(n, conf.gen_r, conf.gen_k, conf.seed);
    } else if (conf.gen == "rhg") {
        edge_list = gen.GenerateRHG(n, conf.gen_gamma, conf.gen_d, conf.gen_k, conf.seed);
    } else if (conf.gen == "ba") {
        edge_list = gen.GenerateBA(n, conf.gen_d, conf.gen_k, conf.seed);
    } else if (conf.gen == "grid_2d") {
        edge_list = gen.Generate2DGrid(n, m, conf.gen_p, conf.gen_periodic, conf.gen_k, conf.seed);
    } else {
        throw std::runtime_error("Generator not supported");
    }
    NodeId local_from = edge_list[0].first;
    NodeId local_to = edge_list[0].second + 1;
    if (conf.verbosity_level > 1) {
        atomic_debug(std::to_string(local_from) + ", " + std::to_string(local_to));
    }
    NodeId local_node_count = local_to - local_from;
    edge_list.erase(edge_list.begin());

    std::sort(edge_list.begin(), edge_list.end());

    //kagen sometimes produces duplicate edges
    auto it = std::unique(edge_list.begin(), edge_list.end());
    edge_list.erase(it, edge_list.end());

    for (auto &edge : edge_list) {
        NodeId tail = edge.first;
        NodeId head = edge.second;
        //atomic_debug(std::to_string(tail) + " " + std::to_string(head));
        if (tail >= local_from && tail < local_to) {
            if (head < local_from || head >= local_to) {
                ghosts.insert(head);
            }
        }
    }
    NodeId total_node_count;
    MPI_Allreduce(&local_node_count, &total_node_count, 1, MPI_NODE, MPI_SUM, MPI_COMM_WORLD);

    gather_PE_ranges(local_from, local_to, ranges, MPI_COMM_WORLD, rank, size);

    if (conf.rhg_fix) {
        fix_broken_edge_list(edge_list, ranges, ghosts, rank, size);
    }

    NodeId current_node = std::numeric_limits<NodeId>::max();
    Degree degree_counter = 0;
    std::vector<LocalGraphView::NodeInfo> node_info;
    std::vector<NodeId> edge_heads;
    for (auto& edge : edge_list) {
        NodeId tail = edge.first;
        NodeId head = edge.second;
        if (tail >= local_from && tail < local_to) {
            Edge e{tail, head};
            if (current_node != e.tail) {
                if (current_node != std::numeric_limits<NodeId>::max()) {
                    node_info.emplace_back(current_node, degree_counter);
                }
                degree_counter = 0;
            }
            edge_heads.emplace_back(e.head);
            degree_counter++;
        }
    }
    node_info.emplace_back(current_node, degree_counter);

    return LocalGraphView {std::move(node_info), std::move(edge_heads)};
}

LocalGraphView read_local_metis_graph(const std::string& input, const GraphInfo& graph_info, PEID rank, PEID size) {
    std::vector<EdgeId> first_out;
    std::vector<NodeId> head;
    read_metis_distributed(input, graph_info, first_out, head, rank, size);

    std::vector<std::pair<NodeId, NodeId>> ranges(size);
    gather_PE_ranges(graph_info.local_from, graph_info.local_to, ranges, MPI_COMM_WORLD, rank, size);

    std::vector<LocalGraphView::NodeInfo> node_info(first_out.size() - 1);
    for(size_t i = 0; i < first_out.size() - 1; ++i) {
        node_info[i].global_id = i + graph_info.local_from;
        node_info[i].degree = first_out[i + 1] - first_out[i];
    }
    return LocalGraphView {std::move(node_info), std::move(head)};;
}

LocalGraphView read_local_partitioned_edgelist(const std::string& input, const Config& conf, PEID rank, PEID size) {
    node_set ghosts;
    ghosts.set_empty_key(-1);
    std::vector<Edge> edges;
    std::vector<std::pair<NodeId, NodeId>> ranges(size);


    auto input_path = std::filesystem::path(input + "_" + std::to_string(rank));

    NodeId local_from = std::numeric_limits<NodeId>::max();
    NodeId local_to = std::numeric_limits<NodeId>::min();
    read_edge_list(input_path.string(), [&](Edge e) {
        if (e.tail > local_to) {
            local_to = e.tail;
        } 
        if (e.tail < local_from) {
            local_from = e.tail;
        }
        edges.emplace_back(e);
    },  1, "e", "c"); 
    local_to++;

    if (conf.verbosity_level > 1) {
        atomic_debug(std::to_string(local_from) + ", " + std::to_string(local_to));
    }

    NodeId local_node_count = local_to - local_from;

    std::sort(edges.begin(), edges.end(), [&](const Edge& e1, const Edge& e2) {
            return std::tie(e1.tail, e1.head) < std::tie(e2.tail, e2.head);
    });

    //kagen sometimes produces duplicate edges
    auto it = std::unique(edges.begin(), edges.end());
    edges.erase(it, edges.end());

    NodeId total_node_count;
    MPI_Allreduce(&local_node_count, &total_node_count, 1, MPI_NODE, MPI_SUM, MPI_COMM_WORLD);

    gather_PE_ranges(local_from, local_to, ranges, MPI_COMM_WORLD, rank, size);

    if (conf.rhg_fix) {
        fix_broken_edge_list(edges, ranges, ghosts, rank, size);
    }

    NodeId current_node = std::numeric_limits<NodeId>::max();
    Degree degree_counter = 0;
    std::vector<LocalGraphView::NodeInfo> node_info;
    std::vector<NodeId> edge_heads;
    for (auto& edge : edges) {
        NodeId tail = edge.tail;
        NodeId head = edge.head;
        if (tail >= local_from && tail < local_to) {
            Edge e{tail, head};
            if (current_node != e.tail) {
                if (current_node != std::numeric_limits<NodeId>::max()) {
                    node_info.emplace_back(current_node, degree_counter);
                }
                degree_counter = 0;
            }
            edge_heads.emplace_back(e.head);
            degree_counter++;
        }
    }
    node_info.emplace_back(current_node, degree_counter);

    return LocalGraphView {std::move(node_info), std::move(edge_heads)};
}

LocalGraphView read_local_binary_graph(const std::string& input, const GraphInfo& graph_info, PEID rank, PEID size) {
    auto input_path = std::filesystem::path(input);
    auto basename = input_path.stem();
    auto path = input_path.parent_path();
    auto first_out_path = path / (basename.string() + ".first_out");
    auto head_path = path / (basename.string() + ".head");
    if (!std::filesystem::exists(first_out_path)) {
        throw std::runtime_error("File " + first_out_path.string() + " does not exist.");
    }
    if (!std::filesystem::exists(head_path)) {
        throw std::runtime_error("File " + head_path.string() + " does not exist.");
    }


    size_t first_index = graph_info.local_from * sizeof(EdgeId);
    size_t bytes_to_read = (graph_info.local_node_count() + 1) * sizeof(EdgeId);
    std::vector<EdgeId> first_out(graph_info.local_node_count() + 1);
    {
        std::ifstream in(first_out_path, std::ios::binary);
        in.seekg(first_index, std::ios::beg);
        in.read(reinterpret_cast<char*>(&first_out[0]), bytes_to_read);
    }
    std::vector<NodeId> head(first_out[graph_info.local_node_count()] - first_out[0]);
    first_index = first_out[0] * sizeof(NodeId);
    bytes_to_read = head.size() * sizeof(NodeId);
    {
        std::ifstream in(head_path, std::ios::binary);
        in.seekg(first_index, std::ios::beg);
        in.read(reinterpret_cast<char*>(&head[0]), bytes_to_read);
    }
    std::vector<std::pair<NodeId, NodeId>> ranges(size);
    gather_PE_ranges(graph_info.local_from, graph_info.local_to, ranges, MPI_COMM_WORLD, rank, size);

    std::vector<LocalGraphView::NodeInfo> node_info(first_out.size() - 1);
    for(size_t i = 0; i < first_out.size() - 1; ++i) {
        node_info[i].global_id = i + graph_info.local_from;
        node_info[i].degree = first_out[i + 1] - first_out[i];
    }
    return LocalGraphView {std::move(node_info), std::move(head)};;
}

void read_graph_info_from_binary(const std::string& input, NodeId& node_count, EdgeId& edge_count) {
    auto input_path = std::filesystem::path(input);
    auto basename = input_path.stem();
    auto path = input_path.parent_path();
    auto first_out_path = path / (basename.string() + ".first_out");
    auto head_path = path / (basename.string() + ".head");
    if (!std::filesystem::exists(first_out_path)) {
        throw std::runtime_error("File " + first_out_path.string() + " does not exist.");
    }
    if (!std::filesystem::exists(head_path)) {
        throw std::runtime_error("File " + head_path.string() + " does not exist.");
    }
    {
        std::ifstream in(first_out_path, std::ios::binary);
        in.seekg(0, std::ios::end);
        unsigned long long file_size = in.tellg();
        if (file_size % sizeof(uint64_t) != 0) {
            throw std::runtime_error("Filesize is no multiple of 64 Bit");
        }
        node_count = file_size / sizeof(uint64_t) - 1;
    }
    {
        std::ifstream in(head_path, std::ios::binary);
        in.seekg(0, std::ios::end);
        unsigned long long file_size = in.tellg();
        if (file_size % sizeof(uint64_t) != 0) {
            throw std::runtime_error("Filesize is no multiple of 64 Bit");
        }
        edge_count = file_size / sizeof(uint64_t);
    }
}

LocalGraphView read_local_graph(const std::string& input, InputFormat format, PEID rank, PEID size) {
    NodeId total_node_count;
    EdgeId total_edge_count;
    if (format == InputFormat::metis) {
        read_metis_header(input, total_node_count, total_edge_count);
    } else if (format == InputFormat::binary) {
        read_graph_info_from_binary(input, total_node_count, total_edge_count);
    } else {
        throw std::runtime_error("Unsupported format");
    }

    GraphInfo graph_info = GraphInfo::even_distribution(total_node_count, rank, size);

    //atomic_debug("[" + std::to_string(graph_info.local_from) + ", " + std::to_string(graph_info.local_to) + ")");
    if (format == InputFormat::metis) {
        return read_local_metis_graph(input, graph_info, rank, size);
    } else if (format == InputFormat::binary) {
        return read_local_binary_graph(input, graph_info, rank, size);
    } else {
        throw std::runtime_error("This should not happen.");
    }
}

std::pair<NodeId, NodeId> get_node_range(const std::string& input, PEID rank, PEID size) {
    (void) size;
    std::ifstream stream(input);
    if (stream.fail()) {
        throw std::runtime_error("Could not open input file for reading: " + input);
    }
    PEID pe = 0;
    std::string line;
    while (pe <= rank && std::getline(stream, line)) {
        pe++;
    }
    std::istringstream sstream(line);
    NodeId from, to;
    sstream >> pe >> from >> to;
    if (pe != rank) {
        throw std::runtime_error("Something went wrong");
    }
    return std::make_pair(from, to);
}

}
