import pandas as pd
import logging
import rootutils
from sklearn.model_selection import train_test_split
rootutils.setup_root(__file__,
                     indicator=(".project-root", "setup.cfg", "setup.py", ".git", "pyproject.toml"),
                     pythonpath=True)
from core.contrastive_learning.data.components.datasetcl import create_batch_dataset_cl

logger = logging.getLogger(__name__)

class DataModuleCL(object):
    # Lớp quản lý dữ liệu cho việc huấn luyện và kiểm tra mô hình
    def __init__(self,
                csv_file,
                train_test_ratio,
                batch_size,
                seed=1234):
        # Khởi tạo module dữ liệu với file CSV, tỷ lệ train/test và kích thước batch
        super(DataModuleCL, self).__init__()
        logger.info(f"Init DataModule Contrastive learning")
        self.csv_file = csv_file
        self.train_test_ratio = train_test_ratio
        self.batch_size = batch_size
        self.seed = seed
        
        self.prepare_dataset()
        self.setup()
        
    def prepare_dataset(self):
        # Đọc và trộn dữ liệu từ file CSV
        logger.info(f"Prepare dataset for datamodule CL")
        self.dataset = pd.read_csv(self.csv_file)
        self.dataset = self.dataset.sample(frac=1, random_state=self.seed).reset_index(drop=True)        
    
    def __get_sample_positive(self, dataset, lb, idx):
        while True:
            sample = dataset.sample(n=1)
            index = sample.index.item()
            path = sample["path"].item()
            label = sample["label"].item()
            if index == idx: continue
            if label == lb: return (path, label)
    
    def __get_sample_negative(self, dataset, lb, idx):
        while True:
            sample = dataset.sample(n=1)
            index = sample.index.item()
            path = sample["path"].item()
            label = sample["label"].item()
            if index == idx: continue
            if label != lb: return (path, label)
            
    def __augment_dataset_cl(self, dataset):        
        augments_positive = []
        augments_negative = []
        for i in dataset.index:
            anchor_label = dataset.loc[i]["label"]
            augments_positive.append(self.__get_sample_positive(dataset, anchor_label, i))
            augments_negative.append(self.__get_sample_negative(dataset, anchor_label, i))
        return (pd.DataFrame(data=augments_positive, columns=["path", "label"]), 
                pd.DataFrame(data=augments_negative, columns=["path", "label"]))            
        
    def setup(self):
        # Chia dữ liệu thành tập huấn luyện và kiểm tra theo tỷ lệ được chỉ định
        self.train_dataset, self.test_dataset = train_test_split(self.dataset,
                                                                train_size=self.train_test_ratio,
                                                                shuffle=True,
                                                                random_state=self.seed,
                                                                stratify=self.dataset["label"])
        
        logger.info("Create dataset training: anchor, positive, negative")
        self.train_dataset_positive, self.train_dataset_negative = self.__augment_dataset_cl(self.train_dataset)

        logger.info("Create dataset testing: anchor, positive, negative")
        self.test_dataset_positive, self.test_dataset_negative = self.__augment_dataset_cl(self.test_dataset)
        
    def get_size_train_test(self):
        # Trả về kích thước của tập huấn luyện và kiểm tra
        return len(self.train_dataset), len(self.test_dataset) 
        
    def train_dataloader(self):
        # Tạo trình tải dữ liệu cho tập huấn luyện theo batch
        logger.info(f"Get train dataloader")
        for i in range(0, len(self.train_dataset), self.batch_size):
            if i+self.batch_size <= len(self.train_dataset):
                data_batch = self.train_dataset[i:i+self.batch_size]
                data_batch_positive = self.train_dataset_positive[i:i+self.batch_size]
                data_batch_negative = self.train_dataset_negative[i:i+self.batch_size]
                
                paths = data_batch["path"].to_list()
                paths_positive = data_batch_positive["path"].to_list()
                paths_negative = data_batch_negative["path"].to_list()
                
                labels = data_batch["label"].to_list()
                labels_positive = data_batch_positive["label"].to_list()
                labels_negative = data_batch_negative["label"].to_list()
                
                yield create_batch_dataset_cl(paths, paths_positive, paths_negative, labels)
        
    def test_dataloader(self):
        logger.info(f"Get test dataloader")
        # Tạo trình tải dữ liệu cho tập kiểm tra theo batch
        for i in range(0, len(self.test_dataset), self.batch_size):
            if i+self.batch_size <= len(self.test_dataset):
                data_batch = self.test_dataset[i:i+self.batch_size]
                data_batch_positive = self.test_dataset_positive[i:i+self.batch_size]
                data_batch_negative = self.test_dataset_negative[i:i+self.batch_size]
                
                paths = data_batch["path"].to_list()
                paths_positive = data_batch_positive["path"].to_list()
                paths_negative = data_batch_negative["path"].to_list()
                
                labels = data_batch["label"].to_list()
                labels_positive = data_batch_positive["label"].to_list()
                labels_negative = data_batch_negative["label"].to_list()
                                                
                yield create_batch_dataset_cl(paths, paths_positive, paths_negative, labels)
    
# Mã kiểm tra 
# data = DataModule(csv_file="data_storage/raw/csv/test.csv",
#                   batch_size=2,
#                   train_test_ratio=0)
# dataloader = data.test_dataloader()
# for i in dataloader:
#     print(i)