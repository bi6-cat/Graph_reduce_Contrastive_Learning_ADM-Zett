import json
import os
import pathlib
import time
import numpy as np
import torch
import logging
from omegaconf import OmegaConf
from tqdm import tqdm

from core.contrastive_learning.data.datamodulecl import DataModuleCL
from core.contrastive_learning.model.model_cl import ModelModuleCL
from core.output.output import ScoreOutput

logger = logging.getLogger(__name__)

class TrainEngineCL(object):    
    """Training engine manage train, test process"""
    @staticmethod
    def dataset_engine(config):
        # Tạo và chuẩn bị dữ liệu huấn luyện và kiểm thử
        # Dataset
        datamodule_cl = DataModuleCL(
            csv_file=config.training.contrastive_learing.data.csv_file,
            train_test_ratio=config.training.contrastive_learing.data.train_test_ratio,
            batch_size=config.training.contrastive_learing.data.batch_size,        
        )
        logger.info(f"Loaded dataset from {config.training.data.csv_file}")
        train_dataloader = datamodule_cl.train_dataloader()
        test_dataloader = datamodule_cl.test_dataloader()
        logger.info(f"Completed dataloader with batch size {config.training.data.batch_size}")
        # check data
        logger.info(f"Size of train - test: {datamodule_cl.get_size_train_test()}")
        # logger.info(f"train dataloader")
        # for item in train_dataloader:
        #     logger.info(f"{item}")
            
        # logger.info(f"test dataloader")
        # for item in test_dataloader:
        #     logger.info(f"{item}")
        return datamodule_cl, train_dataloader, test_dataloader
    
    @staticmethod
    def training_setup(config, model):
        # Cài đặt môi trường cho quá trình huấn luyện: thiết bị, tối ưu hoá, hàm mất mát
        device = torch.device(config.training.device)
        model.to(device)
        logger.info(f"Training device: {device}")
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.contrastive_learing.learning_rate)
        criterion = torch.nn.TripletMarginLoss(config.training.contrastive_learing.margin)
        return device, optimizer, criterion
        
        
    @staticmethod
    def training_engine(config, train_dataloader): 
        # Thực hiện quá trình huấn luyện mô hình
        
        # show config
        logger.info(f"Config trainer: \n{json.dumps(OmegaConf.to_container(config.training), indent=4)}")
        
        # init config training
        logger.info(f"Loading... artchitecture model")       
        model = ModelModuleCL(config.training)
        
        # load model from checkpoint
        if config.training.contrastive_learing.path_model_checkpoint is not None:
            logger.info(f"Loading... from model checkpoint:\n {config.training.contrastive_learing.path_model_checkpoint}")
            model.load_model(config.training.contrastive_learing.path_model_checkpoint)

        # init, setup hyperparameter of model            
        device, optimizer, criterion = TrainEngineCL.training_setup(config, model)
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
                
                anchors, positives, negatives, labels = batch
                anchors = anchors.to(device)
                positives = positives.to(device)
                negatives = negatives.to(device)
                
                anchor_emb = model(anchors)
                positive_emb = model(positives)
                negative_emb = model(negatives)
                loss = criterion(anchor_emb, positive_emb, negative_emb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            logger.info('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss / (iter + 1)))
            epoch_losses.append(epoch_loss / (iter + 1))
        logger.info(f"Saving... model to {config.training.contrastive_learing.path_save_model}")
        os.makedirs(pathlib.Path(config.training.contrastive_learing.path_save_model).parent.absolute(), exist_ok=True)
        model.save_model(config.training.contrastive_learing.path_save_model, config.training.contrastive_learing.type)
        logger.info(f"Completed Training model with output:\n\
                        \tPath to model: {config.training.contrastive_learing.path_save_model}")   
        
    @staticmethod
    def test_engine(config, test_dataloader):
        # Đánh giá mô hình trên tập dữ liệu kiểm thử
        logger.info("Testing...")
        device = torch.device(config.training.device)
        logger.info(f"Testing device: {device}")
        model = ModelModuleCL(config.training)
        model.to(device)
        path_model = os.path.join(config.training.contrastive_learing.path_save_model, "model_cl.pt")
        if not os.path.exists(path_model): logger.error(f"Not exist path to {path_model}")
        logger.info(f"Loading... model from checkpoint: {path_model}")  
        model.load_model(path_model)
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
                anchors, _, _, labels = batch
                anchors = anchors.to(device)
                anchor_emb = model(anchors)
                preds.extend(anchor_emb.cpu().tolist())
                trusts.extend(labels.cpu().tolist())
        preds = np.array(preds, dtype=np.float32)
        logger.info(f"Shape Predictions: {preds.shape}")
        ScoreOutput.data_vizualization(preds, trusts, config.training.contrastive_learing.path_save_metrics)
        logger.info(f"Completed Testing model with output:\n\
                        \tPath to metrics: {config.training.contrastive_learing.path_save_metrics}")



