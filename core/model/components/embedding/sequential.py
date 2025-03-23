import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class SequentialGNN(nn.Sequential):
    """
    # Lớp tuần tự kết hợp các mô-đun GNN
    # Xử lý dữ liệu đồ thị theo thứ tự
    """
    def forward(self, g, infeats):
        """
        # Hàm truyền tải dữ liệu qua mạng
        # Tham số:
        #   g: đồ thị đầu vào
        #   infeats: đặc trưng đầu vào
        # Trả về: đặc trưng sau khi xử lý
        """        
        for name, module in self._modules.items():
            if "relu" in name:
                # Áp dụng hàm kích hoạt ReLU
                infeats = module(infeats)
            elif "gnn" in name:
                # Áp dụng lớp GNN với đồ thị và đặc trưng
                infeats = module(g, infeats)                
        return infeats