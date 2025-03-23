import json
import logging
import os
from pathlib import Path
import pickle

from omegaconf import OmegaConf
import torch

from core.classify.dataset.dataset_combine import DataCombine
from core.model.api_module import get_algorithm_classify
from core.output.output import ScoreOutput
from core.utils.write_read_file import read_embedding

logger = logging.getLogger(__name__)

class ClassifyEngine(object):
    def __init__(self, config):        
        self.config = config
    
    def start_combine_data(self):
        # Bắt đầu kết hợp dữ liệu
        datamodule = DataCombine(self.config.data)
        datamodule.start_compress()

    def load_data_after_combine(self):
        # Tải dữ liệu sau khi kết hợp
        logger.info(f"Loading train.pt, test.pt after combine from {self.config.data.path_save_data_combine}")
        path_train = os.path.join(self.config.data.path_save_data_combine, "train.pt")
        path_test = os.path.join(self.config.data.path_save_data_combine, "test.pt")
        train = torch.load(path_train)
        test = torch.load(path_test)
        X_train, y_train = train["X"], train["y"]
        X_test, y_test = test["X"], test["y"]
        logger.info(f"Loaded train.pt, test.pt after combine")
        return X_train, y_train, X_test, y_test
        
    def classify(self):
        # Phân loại
        logger.info(f"Config classify: {json.dumps(OmegaConf.to_container(self.config.classify_algorithm), indent=4)}")        
        classify_engines = []
        engine_labels = []
        logger.info(f"Loading object classify: {self.config.classify_algorithm.algorithms}")
        for alt in self.config.classify_algorithm.algorithms:
            if alt in self.config.classify_algorithm:
                ml_engines = get_algorithm_classify(alt)(**self.config.classify_algorithm[alt])
                logger.info(f"Loaded algorithm: {alt}")
                classify_engines.append(ml_engines)
                engine_labels.append(alt)
        logger.info(f"Loaded object classify: {self.config.classify_algorithm.algorithms}")
        
        X_train, y_train, X_test, y_test = self.load_data_after_combine()
        for label, engine in zip(engine_labels, classify_engines):
            logger.info(f"Classify with: {label} ----------------------------")
            engine.fit(X_train, y_train)
            logger.info(f"Trained model")
            logger.info(f"Testing model with: {label}")
            y_pred = engine.predict(X_test)
            y_test = self.standard_list(y_test)
            y_pred = self.standard_list(y_pred)
            self.save_experiment(y_pred, y_test, engine, label)
            
        logger.info(f"---------------------------------------\n\
            Ended classify with {self.config.classify_algorithm.algorithms}")
    
    def standard_list(self, ls):
        # Chuẩn hóa danh sách
        if isinstance(ls, torch.Tensor): return ls.tolist()
        return ls
    
    def save_experiment(self, y_preds, y_trues, engine, label):
        # Lưu kết quả thí nghiệm
        path_save_metric_engine = os.path.join(self.config.path_to_save_experiment, label)
        path_save_model_engine = os.path.join(self.config.path_to_save_experiment, label, "model.pt")
        os.makedirs(path_save_metric_engine, exist_ok=True)
        
        # Lưu điểm số của bài kiểm tra
        ScoreOutput.list_score(path_save_metrics=path_save_metric_engine,
                               types=self.config.metrics,
                               y_preds=y_preds,
                               y_trues=y_trues)
        
        # Lưu mô hình
        if label == "MLP": torch.save(engine.model.state_dict(), path_save_model_engine)
        else:
            with open(path_save_model_engine, "wb") as file:
                pickle.dump(engine, file)
        logger.info(f"Saved model `{label}` to file: {path_save_model_engine}")



