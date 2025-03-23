import torch
import logging
logger = logging.getLogger(__name__)

def concat(tensor1: torch.Tensor, tensor2: torch.Tensor):
    # Chuyển đổi các tensor thành dạng vector 1D
    tensor1 = tensor1.reshape((1, -1))
    tensor2 = tensor2.reshape((1, -1))
    logger.info(f"Shape of gnn: {tensor1.shape}")    
    logger.info(f"Shape of permission: {tensor2.shape}")    
    # Ghép nối các vector và chuyển về dạng float32
    concat_vector = torch.concat(tensors=[tensor1, tensor2], dim=-1).to(dtype=torch.float32)
    return concat_vector