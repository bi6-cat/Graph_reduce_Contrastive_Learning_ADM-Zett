from collections import OrderedDict
import torch.nn as nn
import torch
import torch.nn.functional as F
from dgl.nn import MaxPooling
from core.model.components.embedding.sequential import SequentialGNN
from core.model.components.utils.get_convolution_layer import get_convolution_layer
from core.utils.capture_output import Capturing
import logging

logger = logging.getLogger(__name__)

# Các hàm hỗ trợ để tạo mô hình GNN
def get_model_gnn():...
def get_classify():...

class ModelModule(nn.Module, object):
  """
  # Mô hình mạng nơ-ron đồ thị (GNN) cho bài toán phân loại
  """
  def __init__(self, typemodel, infeats, gnn_feats, fc_feats, outclass, features) -> None:
    super(ModelModule, self).__init__()
    
    self.features = features
    
    # Khởi tạo các lớp mạng nơ-ron đồ thị
    logger.info(f"Init Graph Neural Network layers")
    hidden_size_gnn = [infeats] + gnn_feats
    gnn_layers = []
    for i in range(len(hidden_size_gnn) - 1):
      gnn_layers.append((f"gnn_{i + 1}", get_convolution_layer(input_dimension=hidden_size_gnn[i],
                                                  output_dimension=hidden_size_gnn[i + 1],
                                                  name=typemodel)))
      if i < len(hidden_size_gnn) - 2:
        gnn_layers.append((f"gnn_relu_{i + 1}", nn.ReLU()))          
    self.gnn = SequentialGNN(OrderedDict(gnn_layers))
    
    # Khởi tạo lớp gộp (pooling) để tổng hợp đặc trưng từ các nút đồ thị
    logger.info(f"Init Pooling layer")    
    self.pooling = MaxPooling()
    
    # Khởi tạo các lớp fully connected cho phân loại
    logger.info(f"Init Fully connected layers")    
    fc_hidden = [gnn_feats[-1]] + fc_feats + [outclass]
    fc_layers = []
    for i in range(len(fc_hidden) - 1):
      fc_layers.append((f"fc_{i + 1}", nn.Linear(in_features=fc_hidden[i],
                                                 out_features=fc_hidden[i + 1])))
      if i < len(fc_hidden) - 2:
        fc_layers.append((f"fc_relu_{i + 1}", nn.ReLU()))        
    self.fc = nn.Sequential(OrderedDict(fc_layers))
    
    # Lớp softmax để chuẩn hóa xác suất đầu ra
    logger.info(f"Init Softmax layer")        
    self.softmax = nn.Softmax(dim=1)
    
  def forward(self, g):
    """
    # Hàm truyền xuôi (forward pass) của mô hình
    # Đầu vào: g - đồ thị đầu vào
    # Đầu ra: ans - xác suất dự đoán, h - biểu diễn đồ thị
    """
    # Lấy đặc trưng từ các nút trong đồ thị
    infeats = g.ndata[self.features]
        
    # Áp dụng các lớp GNN để học biểu diễn nút
    h = self.gnn(g, infeats)    
    
    # Cập nhật đặc trưng nút và thực hiện pooling
    g.ndata[self.features] = h
    h = self.pooling(g, g.ndata[self.features])

    # Đưa đặc trưng đồ thị qua các lớp fully connected
    logit = self.fc(h)
    
    # Áp dụng softmax để nhận xác suất phân loại
    ans = self.softmax(logit)
    return ans, h

  def forward_contrastive_learning(self, g):
    """
    # Hàm truyền xuôi (forward pass) của mô hình
    # Đầu vào: g - đồ thị đầu vào
    # Đầu ra: h - biểu diễn đồ thị
    """
    # Lấy đặc trưng từ các nút trong đồ thị
    infeats = g.ndata[self.features]
        
    # Áp dụng các lớp GNN để học biểu diễn nút
    h = self.gnn(g, infeats)    
    
    # Cập nhật đặc trưng nút và thực hiện pooling
    g.ndata[self.features] = h
    h = self.pooling(g, g.ndata[self.features])
    
    return h    
  
  def save_model(self, path):
    """
    # Lưu mô hình vào đường dẫn được chỉ định
    """
    torch.save(self.state_dict(), path)      

  def load_model(self, path):
    """
    # Tải mô hình từ đường dẫn được chỉ định
    """
    self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
