from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv, TAGConv, SAGEConv, GraphConv

# Hàm trả về lớp tích chập đồ thị tương ứng với tên được chỉ định
def get_convolution_layer(
            input_dimension: int,
            output_dimension: int,
            name: str
    ) -> Optional[nn.Module]:
    return {
        "GCN": GraphConv(
            input_dimension,
            output_dimension,
            activation=F.relu  # Lớp tích chập GCN cơ bản với hàm kích hoạt ReLU
        ),
        "SAGE": SAGEConv(
            input_dimension,
            output_dimension,
            activation=F.relu,
            aggregator_type='mean',  # GraphSAGE với phương pháp tổng hợp trung bình
            norm=F.normalize
        ),
        "GAT": GATConv(
            input_dimension,
            output_dimension,
            num_heads=1  # Lớp tích chập chú ý đồ thị với 1 đầu chú ý
        ),
        "TAG": TAGConv(
            input_dimension,
            output_dimension,
            k=4  # Lớp tích chập đồ thị topo-adaptive với bước nhảy k=4
        )
    }.get(name, None)