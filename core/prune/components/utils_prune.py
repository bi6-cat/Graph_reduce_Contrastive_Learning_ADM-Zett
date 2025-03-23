import networkx as nx
import networkit as nk

def __convert_to_networkit(nx_graph):
    """ Chuyển đổi từ networkx sang networkit """
    node_mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
    reverse_mapping = {i: node for node, i in node_mapping.items()}
    nk_graph = nk.graph.Graph(weighted=True, directed=nx_graph.is_directed())

    for _ in node_mapping.values():
        nk_graph.addNode()

    for u, v, data in nx_graph.edges(data=True):
        weight = data["weight"] if "weight" in data else 1.0
        nk_graph.addEdge(node_mapping[u], node_mapping[v], weight)
    nk_graph.indexEdges()
    return nk_graph, reverse_mapping

def __convert_to_networkx(nk_graph, reverse_mapping):
    """ Chuyển đổi từ networkit sang networkx """
    nx_graph = nx.DiGraph() if nk_graph.isDirected() else nx.Graph()
    # Loại bỏ các nút bị tách rời khỏi đồ thị
    for u, v, w in nk_graph.iterEdgesWeights():
        nx_graph.add_edge(reverse_mapping[u], reverse_mapping[v], weight=w)
    return nx_graph
