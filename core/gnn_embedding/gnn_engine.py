import glob
import json
import os
from pathlib import Path
import logging
from omegaconf import OmegaConf
import torch

from core.model.components.embedding.gnn import ModelModule
from core.data.components.dataset import create_batch_dataset
from core.utils.write_read_file import write_embedding

logger = logging.getLogger(__name__)

# Lớp quản lý quá trình nhúng đồ thị bằng GNN
class GnnEngine(object):
    # Khởi tạo engine với cấu hình
    def __init__(self, config_model, config_gnn):
        logger.info(f"Config gnn embedding: \n{json.dumps(OmegaConf.to_container(config_gnn), indent=4)}")
        self.config_model = config_model
        self.config_gnn = config_gnn
        self.device = torch.device(config_gnn.device)
        self.setup()
        
    # Cài đặt mô hình
    def setup(self):
        self.model = ModelModule(**self.config_model)
        self.model.to(self.device)
        
    # Thực hiện quá trình nhúng đồ thị
    def start_process(self):
        # Tạo thư mục lưu kết quả
        os.makedirs(Path(self.config_gnn.path_save_embedding), exist_ok=True)
        # Lấy danh sách file FCG cần xử lý
        paths = glob.glob(f"{self.config_gnn.path_data}/*.fcg")[:self.config_gnn.length]
        logger.info("Gnn Embedding...")
        for idx, path in enumerate(paths):
            logger.info(f"File: {idx} - {path}")
            # Tạo dữ liệu từ file đồ thị
            graph, _ = create_batch_dataset([path], [0])
            graph = graph.to(self.device)
            # Chế độ đánh giá và tắt gradient
            self.model.eval()
            with torch.no_grad():
                # Nhúng đồ thị thành vector
                _, h = self.model(graph)
            logger.info(f"Embedded file {path}")
            # Lưu vector nhúng
            path_save_file = os.path.join(self.config_gnn.path_save_embedding, f"{Path(path).stem}.pt")
            write_embedding(h, path_save_file)
            logger.info(f"Saved file {path_save_file}")
        # Báo cáo hoàn thành
        logger.info("Completed embedding gnn")
        logger.info(f"\n\
                    Completed embedding gnn of fcg data:\n\
                    Location of data:\n\
                        \tFCg folder: {self.config_gnn.path_data}\n\
                        \tEmnedding folder: {self.config_gnn.path_save_embedding}")