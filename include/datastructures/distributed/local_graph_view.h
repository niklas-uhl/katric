#ifndef LOCAL_GRAPH_VIEW_H_Z5NCVYVP
#define LOCAL_GRAPH_VIEW_H_Z5NCVYVP

#include <vector>
#include "../graph_definitions.h"

namespace cetric {
    namespace graph {

        struct LocalGraphView {
            struct NodeInfo {
                NodeInfo() = default;
                NodeInfo(NodeId global_id, Degree degree): global_id(global_id), degree(degree) {};
                NodeId global_id = 0;
                Degree degree = 0;
            };
            std::vector<NodeInfo> node_info;
            std::vector<NodeId> edge_heads;
        };

        inline std::ostream& operator<<(std::ostream& os, const LocalGraphView::NodeInfo& c) {
            os << "(" << c.global_id << ", " << c.degree << ")";
            return os;
        }
    }
}

#endif /* end of include guard: LOCAL_GRAPH_VIEW_H_Z5NCVYVP */
