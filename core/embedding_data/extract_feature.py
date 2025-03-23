import networkx as nx

def extract_5_feature(G):
    # Hàm trích xuất 5 đặc trưng từ đồ thị G
    mapping_feature = {}
    
    # Tính các đặc trưng trung tâm và phân cụm
    katz = nx.katz_centrality_numpy(G)  # Độ trung tâm Katz
    closeness = nx.closeness_centrality(G)  # Độ gần nhau
    clustering = nx.clustering(G)  # Hệ số phân cụm
    
    for node in G.nodes():
        mapping_feature[node] = [G.in_degree(node),  # Bậc vào của đỉnh
                                G.out_degree(node),  # Bậc ra của đỉnh
                                katz[node],  # Độ trung tâm Katz
                                closeness[node],  # Độ gần nhau
                                clustering[node]]  # Hệ số phân cụm
    return mapping_feature