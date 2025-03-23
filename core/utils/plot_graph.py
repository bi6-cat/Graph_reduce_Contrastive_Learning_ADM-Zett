import networkx as nx
import matplotlib.pyplot as plt

def plot(nx_original, nx_pruned):
    """ Vẽ đồ thị gốc và đồ thị sau khi cắt tỉa để so sánh """
    # Tạo figure với 2 axes để hiển thị 2 đồ thị song song
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Tính toán vị trí cho các nodes (dùng chung cho cả 2 đồ thị để dễ so sánh)
    pos = nx.spring_layout(nx_original)

    # Vẽ đồ thị gốc
    nx.draw(nx_original, pos, with_labels=False, ax=axes[0], node_color="skyblue", edge_color="gray", node_size=100, font_size=10)
    nx.draw_networkx_edge_labels(nx_original, pos, ax=axes[0])
    axes[0].set_title("Đồ thị ban đầu")

    # Vẽ đồ thị sau khi cắt tỉa
    nx.draw(nx_pruned, pos, with_labels=False, ax=axes[1], node_color="lightgreen", edge_color="gray", node_size=100, font_size=10)
    nx.draw_networkx_edge_labels(nx_pruned, pos, ax=axes[1])
    axes[1].set_title("Đồ thị sau khi cắt tỉa")

    # Hiển thị đồ thị
    plt.tight_layout()
    plt.show()