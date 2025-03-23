import dgl
import torch
from dgl.dataloading import GraphDataLoader
from torch.utils.data import Dataset

from core.contrastive_learning.data.components.utils import add_to_loop_dgl, get_graphs_from_list

class GraphDatasetCL(Dataset):
    def __init__(self, graphs, graphs_positive, graphs_negative, labels):
        super(GraphDatasetCL, self).__init__()
        self.graphs = graphs
        self.graphs_positive = graphs_positive
        self.graphs_negative = graphs_negative
        self.labels = labels    
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.graphs[index], 
                self.graphs_positive[index], 
                self.graphs_negative[index], 
                self.labels[index])
    
def create_batch_dataset_cl(paths, paths_positive, paths_negative, labels):
    """
    # Hàm tạo batch dữ liệu từ các đường dẫn đồ thị và nhãn
    # Input: danh sách đường dẫn đồ thị và nhãn tương ứng
    # Output: batch dữ liệu để huấn luyện mô hình
    """
    graphs = get_graphs_from_list(paths)
    graphs_positive = get_graphs_from_list(paths_positive)
    graphs_negative = get_graphs_from_list(paths_negative)

    graphs = add_to_loop_dgl(graphs)
    graphs_positive = add_to_loop_dgl(graphs_positive)
    graphs_negative = add_to_loop_dgl(graphs_negative)        
    
    # Tạo dataset từ danh sách đồ thị và nhãn
    batch_mini = GraphDatasetCL(graphs=graphs,
                           graphs_positive=graphs_positive,
                           graphs_negative=graphs_negative,
                           labels=torch.Tensor(list(labels)).long())
    
    # Tạo DataLoader để xử lý batch dữ liệu
    batch_mini_loader = GraphDataLoader(
        dataset=batch_mini,
        batch_size=len(graphs),        
    )
    
    # Trả về batch đầu tiên
    return iter(batch_mini_loader).__next__()
    
    