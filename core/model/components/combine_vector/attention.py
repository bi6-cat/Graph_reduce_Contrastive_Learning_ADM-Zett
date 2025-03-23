import torch
import torch.nn.functional as F

def attention_fusion(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """
    Hàm thực hiện kết hợp hai tensor bằng cơ chế attention.
    
    Tham số:
        tensor1: Tensor đầu vào thứ nhất
        tensor2: Tensor đầu vào thứ hai
        
    Trả về:
        attention_vector: Tensor kết quả sau khi kết hợp
    """
    # Chuyển đổi tensor1 thành vector cột
    tensor1 = tensor1.reshape((-1, 1))
    # Chuyển đổi tensor2 thành vector hàng
    tensor2 = tensor2.reshape((1, -1))
    # Tính toán ma trận trọng số bằng cách cộng hai tensor
    wei_broadcast = tensor1 + tensor2
    # Áp dụng hàm softmax để chuẩn hóa trọng số
    wei = F.softmax(wei_broadcast)
    # Tính toán tensor1 mới bằng cách nhân ma trận trọng số chuyển vị với tensor1
    tensor1 = (wei.T @ tensor1).reshape(1, -1)
    # Tính toán tensor2 mới bằng cách nhân tensor2 với ma trận trọng số chuyển vị
    tensor2 = (tensor2 @ wei.T)
    # Ghép nối hai tensor và chuyển đổi thành kiểu float32
    attention_vector = torch.concat([tensor1, tensor2], dim=-1).to(dtype=torch.float32)
    return attention_vector

# Ví dụ sử dụng:
# tensor1 = torch.ones(10)
# tensor2 = torch.ones(20)
# ans = attention_fusion(tensor1, tensor2)
# print(ans.shape)
# print(ans)