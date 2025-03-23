# Tệp chứa các fixture cho pytest
import pytest
from omegaconf import DictConfig
import logging

from core.config.load_config import load_config

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def dataset_config(config_file="core/config/config.yaml") -> DictConfig:
    # Tạo fixture để truy cập cấu hình dữ liệu
    config = load_config(config_file)
    return config.dataset

@pytest.fixture(scope="module")
def training_config(config_file="core/config/config.yaml") -> DictConfig:
    # Tạo fixture để truy cập cấu hình huấn luyện
    config = load_config(config_file)
    return config.training
    
@pytest.fixture(scope="module")
def gnn_embedding_config(config_file="core/config/config.yaml") -> DictConfig:
    # Tạo fixture để truy cập cấu hình embedding GNN
    config = load_config(config_file)
    return config.gnn_embedding

@pytest.fixture(scope="module")
def permission_config(config_file="core/config/config.yaml") -> DictConfig:
    # Tạo fixture để truy cập cấu hình quyền
    config = load_config(config_file)
    return config.permission

@pytest.fixture(scope="module")
def classify_config(config_file="core/config/config.yaml") -> DictConfig:
    # Tạo fixture để truy cập cấu hình phân loại
    config = load_config(config_file)
    return config.classify