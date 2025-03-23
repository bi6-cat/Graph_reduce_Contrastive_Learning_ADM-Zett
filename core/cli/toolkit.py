import typer
from typer import Argument
from typing import Annotated
import logging
import rootutils

rootutils.setup_root(__file__,
                     indicator=(".project-root", "setup.cfg", "setup.py", ".git", "pyproject.toml"),
                     pythonpath=True)
from core.logger.config_logger import logging_without
from core.config.load_config import load_config
from core.utils.utils import seed_everything

logger = logging.getLogger(__name__)
app = typer.Typer(name="GRC")

# test:>> python -m core.cli.toolkit dataset core/config/config.yaml
@app.command("dataset")
def processing_dataset(config_file: Annotated[str, Argument(help="path to config file")]="./config.yaml"):
    # import
    try:        
        from core.data_engine.data_engine import DataEngine
    except Exception as e:
        raise ImportError("Import error package DataEngine")
    
    # config
    config = load_config(config_file)
    
    # logging
    logging_without(config.dataset.path_logging)

    # Seed
    seed_everything(config.seed)
    
    # generate dataset
    DataEngine.generate_dataset(config)

# test:>> python -m core.cli.toolkit contastive_learning core/config/config.yaml
@app.command("contastive_learning")
def contastive_learning(config_file: Annotated[str, Argument(help="path to config file")]="./config.yaml"):
    # import
    try:        
        from core.contrastive_learning.trainer_cl import TrainEngineCL
    except Exception as e:
        raise ImportError("Import error package DataModuleCL")
    
    # config
    config = load_config(config_file)
    
    # logging
    logging_without(config.training.contrastive_learing.path_logging)
    
    # Dataset
    dataset, train_dataloader, test_dataloader = TrainEngineCL.dataset_engine(config)
        
    # Model training
    TrainEngineCL.training_engine(config=config,
                                train_dataloader=train_dataloader)   
    
    # Model testing
    TrainEngineCL.test_engine(config=config,
                            test_dataloader=test_dataloader)
    
    
# test:>> python -m core.cli.toolkit training core/config/config.yaml
@app.command("training")
def training_model(config_file: Annotated[str, Argument(help="path to config file")]="./config.yaml"):
    # import
    try:
        from core.trainer.trainer import TrainEngine
    except Exception as e:
        raise ImportError("Import error package TrainEngine")
    
    # Config
    config = load_config(config_file)
    
    # logging
    logging_without(config.training.path_logging)

    # Seed
    seed_everything(config.seed)
    
    # Dataset
    dataset, train_dataloader, test_dataloader = TrainEngine.dataset_engine(config)
        
    # Model training
    TrainEngine.training_engine(config=config,
                                train_dataloader=train_dataloader)   
    
    # Model testing
    TrainEngine.test_engine(config=config,
                            test_dataloader=test_dataloader)
    
# test:>> python -m core.cli.toolkit gnn_embedding core/config/config.yaml
@app.command("gnn_embedding")
def gnn_embedding(config_file: Annotated[str, Argument(help="path to config file")]="./config.yaml"):
    # import
    try:
        from core.gnn_embedding.gnn_engine import GnnEngine
    except Exception as e:
        raise ImportError("Import error package GnnEngine")
        
    # Config
    config = load_config(config_file)
    
    # logging
    logging_without(config.gnn_embedding.path_logging)   
    
    # Seed
    seed_everything(config.seed)
    
    # gnn embedding
    gnn_engine = GnnEngine(config_model=config.training.model,
                                  config_gnn=config.gnn_embedding)
    gnn_engine.start_process()        

# test:>> python -m core.cli.toolkit permission core/config/config.yaml
@app.command("permission")
def embedding_permission(config_file: Annotated[str, Argument(help="path to config file")]="./config.yaml"):    
    # import
    try:
        from core.permissions.permission_engine import PermissionEmbedding
    except Exception as e:
        raise ImportError("Import error package PermissionEmbedding")
    
    # Config
    config = load_config(config_file)
    
    # logging
    logging_without(config.permission.path_logging)    
    
    # Seed
    seed_everything(config.seed)
    
    # permission embedidng
    permission_engine = PermissionEmbedding(config=config.permission)
    permission_engine.start_process()

# test:>> python -m core.cli.toolkit classify core/config/config.yaml
@app.command("classify")
def classify(config_file: Annotated[str, Argument(help="path to config file")]="./config.yaml"):
    # import
    try:
        from core.classify.classify_engine import ClassifyEngine
    except Exception as e:
        raise ImportError("Import error package ClassifyEngine")
    
    # Config
    config = load_config(config_file)
    
    # logging
    logging_without(config.classify.path_logging)    
    
    # Seed
    seed_everything(config.seed)
    
    # classify engine
    classify_engine = ClassifyEngine(config.classify)
    
    # compress data
    if config.classify.data.reload:
        classify_engine.start_combine_data()
    
    # classify
    if config.classify.classify_algorithm.run:
        classify_engine.classify()
    
    # add attention of embedding permission
    # contrastive learning

if __name__=="__main__":
    app()