import rootutils

rootutils.setup_root(__file__,
                     indicator=(".project-root", "setup.cfg", "setup.py", ".git", "pyproject.toml"),
                     pythonpath=True)
import networkx as nx
import dgl
import logging

from core.utils.write_read_file import read_txt

logger = logging.getLogger(__name__)

# Hàm này đọc file chứa danh sách API nhạy cảm và trả về một list các API
def get_sensitive_apis(api_file):
    api_text = read_txt(api_file)
    apis = [api.strip() for api in api_text.strip().split("\n")]
    return apis

# Hàm chuẩn hóa tên lớp, loại bỏ ký tự L đầu tiên nếu tên lớp thuộc gói android, java hoặc org
def standard_class_name(name):
    if any(True if i in str(name) else False for i in ["Landroid",  "Ljava", "Lorg"]): return name[1:]
    return name

# Hàm tạo ra tên đầy đủ của một hàm/phương thức trong đồ thị (class_name->method_name)
def get_function_node(node):
    return f"{standard_class_name(node.class_name)}->{node.method.name.replace('>', '').replace('<', '')}"

# Hàm lấy ra các node trong FCG (Function Call Graph) tương ứng với các API nhạy cảm
def get_apis_of_fcg(graph, apis):
    nodes = graph.nodes()
    sensitive_nodes = [node for node in nodes if get_function_node(node) in apis]
    return sensitive_nodes

# Hàm lấy các node lân cận của một node theo thứ tự (order) cho trước
# Sử dụng thuật toán BFS để tìm các node nằm trong phạm vi order từ node ban đầu
def getNeiOrder(g, node, order):
    neighbors = set()
    queue = [(node, 0)]
    
    while queue:
        current_node, current_distance = queue.pop(0)
        if current_distance > order:
            continue
        neighbors.add(current_node)
        for neighbor in g.neighbors(current_node):
            if neighbor not in neighbors:
                queue.append((neighbor, current_distance + 1))
    
    return neighbors

# Hàm chính để cắt tỉa đồ thị dựa trên API nhạy cảm
# - Đọc danh sách API nhạy cảm từ file
# - Chuyển đồ thị sang dạng vô hướng
# - Tìm các node nhạy cảm trong đồ thị
# - Tìm các node nằm trong phạm vi order từ các node nhạy cảm
# - Tạo đồ thị con chỉ chứa các node đã tìm được và các cạnh kết nối chúng
# - Trả về đồ thị ban đầu và đồ thị chuyển đổi dạng DGL graph
def prune_graph_api(g, api_file, order=2):    
    # Lấy danh sách API nhạy cảm
    apis = get_sensitive_apis(api_file)
    
    # Chuyển đồ thị sang dạng vô hướng
    g = g.to_undirected()
    sensitive_nodes = get_apis_of_fcg(g, apis)
    
    # Lấy tất cả node trong phạm vi order từ các node nhạy cảm
    nodes = set()
    for si in sensitive_nodes:
        nodes.update(getNeiOrder(g, si, order))
        
    # Lấy các cạnh kết nối giữa các node trong tập nodes
    edges = set()
    for node1 in nodes:
        for node2 in nodes:
            if g.has_edge(node1, node2):
                edges.add((node1, node2))
    
    # Tạo đồ thị con (subgraph) từ các node và cạnh đã tìm được
    sfcg = nx.DiGraph()
    sfcg.add_nodes_from(nodes)
    sfcg.add_edges_from(edges)
    
    # Lưu bản sao của đồ thị con
    sfcg_origin = sfcg.copy()
    # Chuyển đổi nhãn của node thành số nguyên
    sfcg = nx.convert_node_labels_to_integers(sfcg)
    # Chuyển đổi NetworkX graph sang DGL graph
    dg = dgl.from_networkx(sfcg)
    
    return sfcg_origin, dg
    
    
# Ví dụ gọi hàm prune_graph_api với file chứa danh sách API nhạy cảm
# prune_graph_api("data_storage/processed/sensitive_apis/sensitive_apis.txt")
