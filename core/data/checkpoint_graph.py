# Module xử lý lưu và tải đồ thị sử dụng DGL và NetworkX

from dgl import load_graphs
import dgl
import pickle as pkl
import logging

logger = logging.getLogger(__name__)

# Tải đồ thị DGL từ file
def load_graph(path):
    graph = load_graphs(path)
    logger.info(f"Loaded graph from file {path}")
    return graph

# Lưu đồ thị DGL vào file
def save_graph(path, graphs):
    dgl.data.utils.save_graphs(str(path), graphs)
    logger.info(f"Saved graph to file {path}")
   
# Tải đồ thị NetworkX từ file pickle
def load_graph_nx(path):
    with open(path, "rb") as file:
        graph = pkl.load(file)
    logger.info(f"Loaded graphNX to file {path}")
    return graph
   
# Lưu đồ thị NetworkX vào file pickle
def save_graph_nx(path, graph):
    with open(path, "wb") as file:
        pkl.dump(graph, file)
    logger.info(f"Saved graphNX to file {path}")

