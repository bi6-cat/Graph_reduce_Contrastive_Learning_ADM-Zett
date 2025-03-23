#model 1 layer
import torch.nn as nn
import torch
import torch.nn.functional as F
import dgl

from core.model.components.utils import get_convolution_layer

# Mô hình GNN 1 lớp
class ModelNN1Layer(nn.Module, object):
  def __init__(self, typemodel, infeats, h1feats, fc_layer1, fc_layer2, outclass) -> None:
    super(ModelNN1Layer, self).__init__()
    self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    self.conv1 = get_convolution_layer(infeats, h1feats, typemodel)
    self.fc1 = nn.Linear(h1feats, fc_layer1)
    self.fc2 = nn.Linear(fc_layer1, fc_layer2)
    self.fc3 = nn.Linear(fc_layer2, outclass)


    nn.init.xavier_uniform(self.fc1.weight)
    nn.init.xavier_uniform(self.fc2.weight)
    nn.init.xavier_uniform(self.fc3.weight)
    
  def g2v(self, g, infeats):
    # Thêm self-loop vào đồ thị
    g = dgl.add_self_loop(g)
    # Áp dụng lớp convolution
    h = self.conv1(g, infeats)
    h = F.relu(h)

    # Trích xuất đặc trưng bằng max pooling
    g.ndata['features'] = h
    h = dgl.max_nodes(g, 'features')

    # Đưa qua các lớp fully connected
    ans = self.fc1(h)
    ans = F.relu(ans)
    ans = self.fc2(ans)
    ans = F.relu(ans)
    ans = self.fc3(ans)
    return ans, h

  def out(self, gs):
    lg = []
    for g in gs:
      g = g.to(self.device)
      lg.append(self.g2v(g, g.ndata["features"])[1].float())
    ans = torch.cat(lg, dim=0)
    torch.cuda.empty_cache()
    return ans.squeeze()

  def forward(self, gs):
    lg = []
    for g in gs:
      g = g.to(self.device)
      lg.append(self.g2v(g, g.ndata["features"])[0].float())
    ans = torch.cat(lg, dim=0)
    torch.cuda.empty_cache()
    return ans.squeeze()

  def save(self, path):
    torch.save(self.state_dict(), path)

  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
# ------------------------------------------------------------------------------------------------------------------------------------
# Mô hình GNN 2 lớp
class ModelNN2Layer(nn.Module, object):
  def __init__(self, typemodel,  infeats, h1feats, h2feats, fc_layer1, fc_layer2, outclass) -> None:
    super(ModelNN2Layer, self).__init__()
    self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    self.conv1 = get_convolution_layer(infeats, h1feats, typemodel)
    self.conv2 = get_convolution_layer(h1feats, h2feats, typemodel)
    self.fc1 = nn.Linear(h2feats, fc_layer1)
    self.fc2 = nn.Linear(fc_layer1, fc_layer2)
    self.fc3 = nn.Linear(fc_layer2, outclass)


    nn.init.xavier_uniform(self.fc1.weight)
    nn.init.xavier_uniform(self.fc2.weight)
    nn.init.xavier_uniform(self.fc3.weight)


  def g2v(self, g, infeats):
    # Thêm self-loop vào đồ thị
    g = dgl.add_self_loop(g)
    # Áp dụng lớp convolution thứ nhất
    h = self.conv1(g, infeats)
    h = F.relu(h)
    # Áp dụng lớp convolution thứ hai
    h = self.conv2(g, h)
    h = F.relu(h)

    # Trích xuất đặc trưng bằng sum pooling
    g.ndata['features'] = h
    h = dgl.sum_nodes(g, 'features')
    # Đưa qua các lớp fully connected
    ans = self.fc1(h)
    ans = F.relu(ans)
    ans = self.fc2(ans)
    ans = F.relu(ans)
    ans = self.fc3(ans)
    return ans, h

  def out(self, gs):
    lg = []
    for g in gs:
      g = g.to(self.device)
      lg.append(self.g2v(g, g.ndata["features"])[1].float())
    ans = torch.cat(lg, dim=0)
    torch.cuda.empty_cache()
    return ans.squeeze()

  def forward(self, gs):
    lg = []
    for g in gs:
      g = g.to(self.device)
      lg.append(self.g2v(g, g.ndata["features"])[0].float())
    ans = torch.cat(lg, dim=0)
    torch.cuda.empty_cache()
    return ans.squeeze()

  def save(self, path):
    torch.save(self.state_dict(), path)

  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
# ------------------------------------------------------------------------------------------------------------------------------------
# Mô hình GNN 3 lớp
class ModelNN3Layer(nn.Module, object):
  def __init__(self, typemodel,  infeats, h1feats, h2feats, h3feats, fc_layer1, fc_layer2, outclass) -> None:
    super(ModelNN3Layer, self).__init__()
    self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    self.conv1 = get_convolution_layer(infeats, h1feats, typemodel)
    self.conv2 = get_convolution_layer(h1feats, h2feats, typemodel)
    self.conv3 = get_convolution_layer(h2feats, h3feats, typemodel)
    self.fc1 = nn.Linear(h3feats, fc_layer1)
    self.fc2 = nn.Linear(fc_layer1, fc_layer2)
    self.fc3 = nn.Linear(fc_layer2, outclass)


    nn.init.xavier_uniform(self.fc1.weight)
    nn.init.xavier_uniform(self.fc2.weight)
    nn.init.xavier_uniform(self.fc3.weight)


  def g2v(self, g, infeats):
    # Thêm self-loop vào đồ thị
    g = dgl.add_self_loop(g)
    # Áp dụng lớp convolution thứ nhất
    h = self.conv1(g, infeats)
    h = F.relu(h)
    # Áp dụng lớp convolution thứ hai
    h = self.conv2(g, h)
    h = F.relu(h)
    # Áp dụng lớp convolution thứ ba
    h = self.conv3(g, h)
    h = F.relu(h)

    # Trích xuất đặc trưng bằng max pooling
    g.ndata['features'] = h
    h = dgl.max_nodes(g, 'features')
    # Đưa qua các lớp fully connected
    ans = self.fc1(h)
    ans = F.relu(ans)
    ans = self.fc2(ans)
    ans = F.relu(ans)
    ans = self.fc3(ans)
    return ans, h

  def out(self, gs):
    lg = []
    for g in gs:
      g = g.to(self.device)
      lg.append(self.g2v(g, g.ndata["features"])[1].float())
    ans = torch.cat(lg, dim=0)
    torch.cuda.empty_cache()
    return ans.squeeze()

  def forward(self, gs):
    lg = []
    for g in gs:
      g = g.to(self.device)
      lg.append(self.g2v(g, g.ndata["features"])[0].float())
    ans = torch.cat(lg, dim=0)
    torch.cuda.empty_cache()
    return ans.squeeze()
    
  def save(self, path):
    torch.save(self.state_dict(), path)

  def load(self, path):
    self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))