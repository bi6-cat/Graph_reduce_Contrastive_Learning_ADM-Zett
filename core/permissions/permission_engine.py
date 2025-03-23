import glob
import json
import os
from pathlib import Path
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
import logging

from core.permissions.model.llm import LLMEncoding
from core.permissions.model.onehot import OneHotEncoding
from core.permissions.model.w2v import W2VEncoding
from core.permissions.utils.extract_apk_text import get_sequence_permission
from core.utils.write_read_file import write_embedding

logger = logging.getLogger(__name__)

class PermissionEmbedding(object):
    def __init__(self, config):
        # Hiển thị cấu hình quyền
        logger.info(f"Config Permission: \n{json.dumps(OmegaConf.to_container(config), indent=4)}")
        self.config = config
        self.load_model()
        
    def load_model(self):
        # Tải mô hình nhúng văn bản
        logger.info(f"Encoding with type model: {self.config.type}")
        if self.config.type == "llm":
            self.model = LLMEncoding(cache_folder=self.config.model_embeding.cache_dir)
        elif self.config.type == "onehot":
            self.model = OneHotEncoding(vocab_file=self.config.model_embeding.vocab_file_onehot)
        elif self.config.type == "w2v":
            self.model = W2VEncoding(vocab_file=self.config.model_embeding.vocab_file_w2v,
                                     vector_size=self.config.model_embeding.vector_size,
                                     epochs=self.config.model_embeding.epochs)
        else:
            logger.error(f"Not support model embedidng type: {self.config.type}")
        
    def embedding_text(self, text):
        # Thực hiện nhúng văn bản thành vector
        logger.info("Embedding... ")
        embed = self.model.encode(text)
        return embed
        
    def start_process(self):
        # Bắt đầu quá trình nhúng quyền
        logger.info("Start embedding permission")
        
        # Lấy danh sách đường dẫn tới các file apk
        paths = glob.glob(self.config.path_data + "/*.apk")[:self.config.length]
        
        # Tạo các thư mục lưu trữ kết quả
        os.makedirs(self.config.path_save_text, exist_ok=True)
        os.makedirs(self.config.path_save_embedding, exist_ok=True)
        
        # Xử lý từng file apk
        for path in paths:
            hash_code = Path(path).stem
            data_file_text = os.path.join(self.config.path_save_text, f"{hash_code}.txt")
            data_file_embedding = os.path.join(self.config.path_save_embedding, f"{hash_code}.pt")
            permission_text = get_sequence_permission(path, data_file_text)
            vector_embedded = self.embedding_text(permission_text)
            write_embedding(vector_embedded, data_file_embedding)
        
        # Thông báo hoàn thành quá trình
        logger.info(f"\n\
                    Completed gen data text, embedding of the permission:\n\
                    Location of data:\n\
                        \tText folder: {self.config.path_save_text}\n\
                        \tEmnedding folder: {self.config.path_save_embedding}")
