import dgl
from core.data.checkpoint_graph import load_graph

def add_to_loop_dgl(graphs):
    return [dgl.add_self_loop(g) for g in graphs]

def get_graphs_from_list(paths):
    graphs = []
    for path in paths:
        # Tải đồ thị từ đường dẫn
        graph = load_graph(path)
        graphs.append(graph[0][0])        
    return graphs