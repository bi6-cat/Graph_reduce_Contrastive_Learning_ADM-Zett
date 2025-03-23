import logging
import networkx as nx
import networkit as nk
import dgl
from core.prune.components.utils_prune import __convert_to_networkit, __convert_to_networkx

logger = logging.getLogger(__name__)

def prune_graph_scan(g, targetRatio):
    # Chuyển đồ thị sang định dạng networkit
    nk_graph, reverse_mapping = __convert_to_networkit(g)
    logger.info(f"Before prune: {nk_graph.numberOfNodes()} nodes - {nk_graph.numberOfEdges()} edges")
    
    # Sử dụng SCAN để giảm kích thước đồ thị
    scan_degree = nk.sparsification.SCANSparsifier()
    nk_pruned = scan_degree.getSparsifiedGraphOfSize(nk_graph, targetRatio)
    logger.info(f"After pruned: {nk_pruned.numberOfNodes()} nodes - {nk_pruned.numberOfEdges()} edges")
    
    # Chuyển đồ thị đã cắt tỉa về định dạng networkx
    G_original = __convert_to_networkx(nk_pruned, reverse_mapping)
    G = G_original.copy()
    
    # Đánh số lại các node và chuyển sang định dạng DGL
    G = nx.convert_node_labels_to_integers(G)
    dg = dgl.from_networkx(G)
    
    return G_original, dg