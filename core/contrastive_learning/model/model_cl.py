from collections import OrderedDict
import os
import torch
import torch.nn as nn
import logging

from core.model.components.embedding.gnn import ModelModule

logger = logging.getLogger(__name__)

class ModelModuleCL(nn.Module):
    def __init__(self, config):
        super(ModelModuleCL, self).__init__()
        self.gnn = ModelModule(**config.model)
        
        logger.info(f"Init Fully connected layers")    
        fc_hidden = [config.contrastive_learing.project_layer.in_feat] + config.contrastive_learing.project_layer.fc_feats + [config.contrastive_learing.project_layer.outclass]
        fc_layers = []
        for i in range(len(fc_hidden) - 1):
            fc_layers.append((f"fc_{i + 1}", nn.Linear(in_features=fc_hidden[i],
                                                        out_features=fc_hidden[i + 1])))
            if i < len(fc_hidden) - 2:
                fc_layers.append((f"fc_relu_{i + 1}", nn.ReLU()))        
        self.fc = nn.Sequential(OrderedDict(fc_layers))
        
    def forward(self, g):        
        h = self.gnn.forward_contrastive_learning(g)
        h = self.fc(h)
        return h
    
    def save_model(self, path_folder, tp):
        os.makedirs(path_folder, exist_ok=True)
        path_cl = os.path.join(path_folder, "model_cl.pt")
        path_gnn = os.path.join(path_folder, "model_gnn.pt")
        if tp == "cl" or tp == "all":
            torch.save(self.state_dict(), path_cl)
        if tp == "gnn" or tp == "all":
            torch.save(self.gnn.state_dict(), path_gnn)
            
    def load_model(self, path):
        path_model = os.path.join(path)
        if not os.path.exists(path_model):
            logger.error(f"File not exists: {path_model}")
            return
        state_dict = torch.load(path_model, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)
        
        

        
        
            
        
        
        