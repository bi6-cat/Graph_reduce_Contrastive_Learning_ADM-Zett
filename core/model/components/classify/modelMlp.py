import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    # Mô hình MLP cơ bản với số lớp ẩn có thể tùy chỉnh
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Xây dựng các lớp của mạng neural
        for i in range(len(layer_sizes) - 1):
            layers.append((f'fc{i+1}', nn.Linear(layer_sizes[i], layer_sizes[i+1])))
            if i < len(layer_sizes) - 2:  
                layers.append((f'relu{i+1}', nn.ReLU()))
        
        self.mlp = nn.Sequential(OrderedDict(layers))
        
    def forward(self, data):
        # Truyền dữ liệu qua mạng
        return self.mlp(data)
    
    def load(self, path):
        # Tải trọng số đã lưu
        self.load_state_dict(torch.load(path))
        
class MLPEngine(object):
    # Quản lý quá trình huấn luyện và dự đoán
    def __init__(self, epochs, lr, device, input_size, hidden_sizes, output_size):
        self.model = MLP(input_size, hidden_sizes, output_size)
        self.epochs = epochs
        self.lr = lr
        self.device = device
    
    def fit(self, X_train, y_train):        
        # Huấn luyện mô hình với dữ liệu đầu vào
        device = torch.device(self.device)
        self.model.to(device)            
        
        X_train, y_train = X_train.to(device), y_train.to(device)
        
        # Hàm mất mát và tối ưu hóa
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
    
    def predict(self, X_test):
        # Dự đoán kết quả từ dữ liệu kiểm tra
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test)
            y_pred = y_pred.argmax(dim=-1).detach().cpu().to(dtype=torch.long)
        return y_pred