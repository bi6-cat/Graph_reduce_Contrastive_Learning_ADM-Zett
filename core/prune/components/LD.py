import networkx as nx
import networkit as nk
import dgl
import logging
from core.prune.components.utils_prune import __convert_to_networkit, __convert_to_networkx

logger = logging.getLogger(__name__)

def prune_graph_local_degree(g, targetRatio):
    # Chuyển đổi đồ thị sang định dạng networkit
    nk_graph, reverse_mapping = __convert_to_networkit(g)
    logger.info(f"Before prune: {nk_graph.numberOfNodes()} nodes - {nk_graph.numberOfEdges()} edges")
    
    # Áp dụng thuật toán LocalDegreeSparsifier để làm thưa đồ thị
    local_degree = nk.sparsification.LocalDegreeSparsifier()
    nk_pruned = local_degree.getSparsifiedGraphOfSize(nk_graph, targetRatio)
    logger.info(f"After pruned: {nk_pruned.numberOfNodes()} nodes - {nk_pruned.numberOfEdges()} edges")
    
    # Chuyển đồ thị đã được làm thưa về định dạng NetworkX và DGL
    G_original = __convert_to_networkx(nk_pruned, reverse_mapping)
    G = G_original.copy()    
    G = nx.convert_node_labels_to_integers(G)
    dg = dgl.from_networkx(G)    
    return G_original, dg