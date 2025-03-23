import dgl
import networkx as nx
from core.embedding_data.embedding_codebert import embedding_codebert
from core.embedding_data.extract_feature import extract_5_feature

# Hàm nhúng đồ thị - chuyển đồ thị thành biểu diễn số học cho học máy
def embedd_graph(G_original, embedding_dim, cache_dir):
    # Trích xuất 5 đặc trưng thống kê từ đồ thị
    mapping_5_features = extract_5_feature(G_original)
    # Tạo embedding sử dụng mô hình CodeBERT
    mapping_embedd = embedding_codebert(G_original, embedding_dim, cache_dir=cache_dir)
    # Gán các đặc trưng vào đồ thị
    nx.set_node_attributes(G_original, mapping_5_features, "5_features")
    nx.set_node_attributes(G_original, mapping_embedd, "embedding")
    
    # Chuyển nhãn nút thành số nguyên để tương thích với DGL
    G = nx.convert_node_labels_to_integers(G_original)
    # Chuyển đổi từ NetworkX sang DGL để sử dụng trong học sâu
    dg = dgl.from_networkx(G, node_attrs=["5_features", "embedding"])
    return G_original, dg


