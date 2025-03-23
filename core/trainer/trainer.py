import json
import os
import pathlib
import time
import torch
import logging
from omegaconf import OmegaConf
from tqdm import tqdm

from core.data.components.dataset import create_batch_dataset
from core.data.datamodule import DataModule
from core.model.components.embedding.gnn import ModelModule
from core.output.output import ScoreOutput

logger = logging.getLogger(__name__)

class TrainEngine(object):    
    """Training engine manage train, test process"""
    @staticmethod
    def dataset_engine(config):
        # Tạo và chuẩn bị dữ liệu huấn luyện và kiểm thử
        # Dataset
        dataset = DataModule(
            csv_file=config.training.data.csv_file,
            train_test_ratio=config.training.data.train_test_ratio,
            batch_size=config.training.data.batch_size,
        )
        logger.info(f"Loaded dataset from {config.training.data.csv_file}")
        train_dataloader = dataset.train_dataloader()
        test_dataloader = dataset.test_dataloader()
        logger.info(f"Completed dataloader with batch size {config.training.data.batch_size}")
        # check data
        logger.info(f"Size of train - test: {dataset.get_size_train_test()}")
        # logger.info(f"train dataloader")
        # for item in train_dataloader:
        #     logger.info(f"{item}")
            
        # logger.info(f"test dataloader")
        # for item in test_dataloader:
        #     logger.info(f"{item}")
        return dataset, train_dataloader, test_dataloader
    
    @staticmethod
    def training_setup(config, model):
        # Cài đặt môi trường cho quá trình huấn luyện: thiết bị, tối ưu hoá, hàm mất mát
        device = torch.device(config.training.device)
        model.to(device)
        logger.info(f"Training device: {device}")
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        return device, optimizer, criterion
        
        
    @staticmethod
    def training_engine(config, train_dataloader): 
        # Thực hiện quá trình huấn luyện mô hình
        
        # show config
        logger.info(f"Config trainer: \n{json.dumps(OmegaConf.to_container(config.training), indent=4)}")
        
        # init config training
        logger.info(f"Loading... artchitecture model")       
        model = ModelModule(**config.training.model)
        
        # load model from checkpoint
        if config.training.path_model_checkpoint is not None:
            logger.info(f"Loading... from model checkpoint:\n {config.training.path_model_checkpoint}")
            model.load_model(config.training.path_model_checkpoint)

        # init, setup hyperparameter of model            
        device, optimizer, criterion = TrainEngine.training_setup(config, model)
        logger.info("Training...")
        epoch_losses = []
        epochs = config.training.epochs
        log_interval = config.training.log_interval

        # start training
        model.train()
        progress_epochs = tqdm(range(epochs), file=open(os.devnull, "w"))
        for epoch in progress_epochs:
            time.sleep(0.1)
            if progress_epochs.n % log_interval == 0:
                logger.info(str(progress_epochs))                
            epoch_loss = 0
            for iter, batch in enumerate(train_dataloader):
                graphs, labels = batch
                graphs = graphs.to(device)
                labels = labels.to(device)
                ans, h = model(graphs)
                loss = criterion(ans, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            logger.info('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss / (iter + 1)))
            epoch_losses.append(epoch_loss / (iter + 1))
        logger.info(f"Saving... model to {config.training.path_save_model}")
        os.makedirs(pathlib.Path(config.training.path_save_model).parent.absolute(), exist_ok=True)
        model.save_model(config.training.path_save_model)
        logger.info(f"Completed Training model with output:\n\
                        \tPath to model: {config.training.path_save_model}")
        
        
    @staticmethod
    def test_engine(config, test_dataloader):
        # Đánh giá mô hình trên tập dữ liệu kiểm thử
        logger.info("Testing...")
        device = torch.device(config.training.device)
        logger.info(f"Testing device: {device}")
        model = ModelModule(**config.training.model)
        model.to(device)
        logger.info(f"Loading... model from checkpoint: {config.training.path_save_model}")       
        model.load_model(config.training.path_save_model)
        logger.info(f"Loaded model:\n {model}")
        # testing
        progress_loader = tqdm(test_dataloader, file=open(os.devnull, "w"))
        preds = []
        trusts = []
        model.eval()
        with torch.no_grad():
            for batch in progress_loader:
                time.sleep(0.1)
                logger.info(str(progress_loader))
                graphs, labels = batch
                graphs = graphs.to(device)
                labels = labels.to(device)
                ans, h = model(graphs)
                ans_batch = ans.argmax(dim=1).long()
                preds.extend(ans_batch.cpu().tolist())
                trusts.extend(labels.cpu().tolist())
        logger.info(f"Ground trust: {trusts}")
        logger.info(f"Predictions: {preds}")
        logger.info(f"Saving... resulft to {config.training.path_save_metrics}")
        ScoreOutput.list_score(types=config.training.metrics,
                               y_preds=preds,
                               y_trues=trusts,
                               path_save_metrics=config.training.path_save_metrics)
        logger.info(f"Completed Testing model with output:\n\
                        \tPath to model: {config.training.path_save_metrics}")
    
    @staticmethod
    def predict(paths, model):
        # Dự đoán kết quả cho dữ liệu mới
        graphs, _ = create_batch_dataset(paths, [0]*len(paths))
        model.eval()
        ans, h = model(graphs)
        preds = ans.argmax(dim=1).long().cpu().tolist()
        return preds



