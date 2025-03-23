import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import torch
from core.data.checkpoint_graph import load_graph
from dgl import batch

class BatchGraphDataset(DGLDataset):
    """
    # Lớp quản lý tập dữ liệu đồ thị, kế thừa từ DGLDataset
    # Lưu trữ các đồ thị và nhãn tương ứng
    """
    def __init__(self, graphs, labels):
        super().__init__(name='graph_dataset')
        self.graphs = graphs
        self.labels = labels
        
    def process(self):
        # Tiền xử lý dữ liệu nếu cần thiết
        pass

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)
        
def create_batch_dataset(paths, labels):
    """
    # Hàm tạo batch dữ liệu từ các đường dẫn đồ thị và nhãn
    # Input: danh sách đường dẫn đồ thị và nhãn tương ứng
    # Output: batch dữ liệu để huấn luyện mô hình
    """
    graphs = []
    for path in paths:
        # Tải đồ thị từ đường dẫn
        graph = load_graph(path)
        graphs.append(graph[0][0])
    # Thêm self-loop vào mỗi đồ thị
    graphs = [dgl.add_self_loop(g) for g in graphs]
 
    # Tạo dataset từ danh sách đồ thị và nhãn
    batch_mini = BatchGraphDataset(graphs, torch.Tensor(list(labels)).long())
    
    # Tạo DataLoader để xử lý batch dữ liệu
    batch_mini_loader = GraphDataLoader(
        dataset=batch_mini,
        batch_size=len(graphs),        
    )
    
    # Trả về batch đầu tiên
    return iter(batch_mini_loader).__next__()
