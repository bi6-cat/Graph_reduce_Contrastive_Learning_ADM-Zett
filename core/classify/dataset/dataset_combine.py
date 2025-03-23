import glob
import os
from pathlib import Path
import logging
from omegaconf import OmegaConf
import torch
import json

from core.model.api_module import get_algorithm_combine
from core.utils.write_read_file import read_embedding, read_json, write_embedding

logger = logging.getLogger(__name__)

class DataCombine(object):
    # Lớp quản lý việc kết hợp và xử lý dữ liệu từ nhiều nguồn khác nhau
    def __init__(self, config):
        # Khởi tạo đối tượng với cấu hình đầu vào
        super(DataCombine, self).__init__()
        logger.info(f"Config data classify: {json.dumps(OmegaConf.to_container(config), indent=4)}")
        self.config = config
        self.hashs, self.labels = self.get_hash_label()
        logger.info(f"Amount of hash code: {self.size}")
        if config.combine_algorithm is not None:
            # Khởi tạo thuật toán kết hợp dữ liệu
            logger.info(f"Init algorithm combine: {config.combine_algorithm}")
            self.combine_engine = get_algorithm_combine(config.combine_algorithm)
        
    def get_hash_label(self):
        # Lấy danh sách mã hash và nhãn từ dữ liệu
        logger.info("Getting... hash code of data")
        path_gnns = glob.glob(f"{self.config.path_data_gnn}/*.pt")
        labels = read_json(self.config.path_data_label_hash)
        return [Path(path).stem for path in path_gnns][:self.config.length], labels

    @property
    def size(self):
        # Trả về số lượng mã hash
        return len(self.hashs)    
    
    def start_compress(self):
        # Nén và kết hợp dữ liệu từ nhiều nguồn
        logger.info(f"Compress data with {self.config.embedding}")
        logger.info(f"Starting... compress data")
        ls_tensor = []
        ls_labels = []
        for idx, hash in enumerate(self.hashs):        
            logger.info(f"hash: {idx} - {hash}")    
            if "gnn" in self.config.embedding:
                # Đọc dữ liệu GNN nếu được yêu cầu
                path_gnn = f"{self.config.path_data_gnn}/{hash}.pt"
                tensor_gnn = read_embedding(path_gnn)
                tensor_compress = tensor_gnn.detach().clone()                                
            else:
                tensor_gnn = None
            if "permission" in self.config.embedding:
                # Đọc dữ liệu permission nếu được yêu cầu
                path_permission = f"{self.config.path_data_permission}/{hash}.pt"
                tensor_permission = read_embedding(path_permission)
                tensor_permission = torch.from_numpy(tensor_permission).to(dtype=torch.float32)
                tensor_compress = tensor_permission.detach().clone()                                
            else:
                tensor_permission = None
            if tensor_gnn is not None and tensor_permission is not None:                                
                # Kết hợp dữ liệu GNN và permission nếu cả hai tồn tại
                tensor_compress = self.combine_engine(tensor_gnn, tensor_permission)
            ls_tensor.append(tensor_compress.reshape(1, -1))
            try:
                ls_labels.append(self.labels[hash])
            except:
                logger.error(f"{hash} have'nt label")
                
        # Chia dữ liệu thành tập huấn luyện và kiểm tra
        train_dataset, train_labels, test_dataset, test_label = self.train_test_split(ls_tensor, ls_labels)
        self.train_zip, self.train_labels = self.zip_tensor(train_dataset, train_labels)
        self.test_zip, self.test_labels =  self.zip_tensor(test_dataset, test_label)    
        logger.info(f"Shape of train: {self.train_zip.shape} - labels: {self.train_labels.shape}")
        logger.info(f"Shape of test: {self.test_zip.shape} - labels: {self.test_labels.shape}")
        self.save_train_test()
        logger.info(f"\n\
                    Completed compress data {self.config.embedding}:\n\
                    Location of data:\n\
                        \tCompress data: {self.config.path_save_data_combine}")
    
    def train_test_split(self, ls_data, ls_labels):
        # Chia dữ liệu thành tập huấn luyện và kiểm tra theo tỷ lệ
        ratio = self.config.train_test_ratio
        logger.info(f"Split data with ratio: {ratio}")
        train_size = int(ratio * self.size)
        train_dataset, train_labels = ls_data[:train_size], ls_labels[:train_size]
        test_dataset, test_label = ls_data[train_size:], ls_labels[train_size:]
        return train_dataset, train_labels, test_dataset, test_label
        
    def zip_tensor(self, ls_tensor, ls_labels):
        # Gộp danh sách tensor thành một tensor duy nhất
        logger.info(f"Zip the data to one file")
        tensors = torch.concat(ls_tensor, dim=0).to(dtype=torch.float32)
        return tensors, torch.tensor(ls_labels, dtype=torch.long)
    
    def save_train_test(self):
        # Lưu dữ liệu đã xử lý vào đường dẫn chỉ định
        logger.info(f"Save data to folder {self.config.path_save_data_combine}")
        path_save_train = os.path.join(self.config.path_save_data_combine, "train.pt")
        path_save_test = os.path.join(self.config.path_save_data_combine, "test.pt")
        write_embedding({
            "X": self.train_zip,
            "y": self.train_labels
        }, path_save_train)
        write_embedding({
            "X": self.test_zip,
            "y": self.test_labels
        }, path_save_test)






