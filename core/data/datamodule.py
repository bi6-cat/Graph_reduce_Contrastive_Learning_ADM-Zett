import pandas as pd
from dgl.dataloading import GraphDataLoader
import rootutils
from sklearn.model_selection import train_test_split
rootutils.setup_root(__file__,
                     indicator=(".project-root", "setup.cfg", "setup.py", ".git", "pyproject.toml"),
                     pythonpath=True)
from core.data.components.dataset import create_batch_dataset

class DataModule(object):
    # Lớp quản lý dữ liệu cho việc huấn luyện và kiểm tra mô hình
    def __init__(self,
                csv_file,
                train_test_ratio,
                batch_size,
                seed=1234):
        # Khởi tạo module dữ liệu với file CSV, tỷ lệ train/test và kích thước batch
        super(DataModule, self).__init__()
        self.csv_file = csv_file
        self.train_test_ratio = train_test_ratio
        self.batch_size = batch_size
        self.seed = seed
        
        self.prepare_dataset()
        self.setup()
        
    def prepare_dataset(self):
        # Đọc và trộn dữ liệu từ file CSV
        self.dataset = pd.read_csv(self.csv_file)
        self.dataset = self.dataset.sample(frac=1, random_state=self.seed).reset_index(drop=True)        
    
    def setup(self):
        # Chia dữ liệu thành tập huấn luyện và kiểm tra theo tỷ lệ được chỉ định
        self.train_dataset, self.test_dataset = train_test_split(self.dataset,
                                                                            train_size=self.train_test_ratio,
                                                                            shuffle=True,
                                                                            random_state=self.seed,
                                                                            stratify=self.dataset["label"])
                        
    def get_size_train_test(self):
        # Trả về kích thước của tập huấn luyện và kiểm tra
        return len(self.train_dataset), len(self.test_dataset) 
        
    def train_dataloader(self):
        # Tạo trình tải dữ liệu cho tập huấn luyện theo batch
        for i in range(0, len(self.train_dataset), self.batch_size):
            if i+self.batch_size <= len(self.train_dataset):
                data_batch = self.train_dataset[i:i+self.batch_size]
                paths = data_batch["path"].to_list()
                labels = data_batch["label"].to_list()
                yield create_batch_dataset(paths, labels)
        
    def test_dataloader(self):
        # Tạo trình tải dữ liệu cho tập kiểm tra theo batch
        for i in range(0, len(self.test_dataset), self.batch_size):
            if i+self.batch_size <= len(self.test_dataset):
                data_batch = self.test_dataset[i:i+self.batch_size]
                paths = data_batch["path"].to_list()
                labels = data_batch["label"].to_list()
                yield create_batch_dataset(paths, labels)
    
# Mã kiểm tra 
# data = DataModule(csv_file="data_storage/raw/csv/test.csv",
#                   batch_size=2,
#                   train_test_ratio=0)
# dataloader = data.test_dataloader()
# for i in dataloader:
#     print(i)