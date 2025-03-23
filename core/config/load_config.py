from omegaconf import DictConfig, OmegaConf
import yaml
import logging
import re
import json
import logging

logger = logging.getLogger(__name__)

def dynamic_config(config):
    # Xử lý biến động trong file cấu hình
    logger.info("Automatic variable")
    pattern = r"\{\{([^}]+)\}\}"
    config_str = json.dumps(OmegaConf.to_container(config), indent=4)
    dynamic_varible = re.findall(pattern, config_str)
    
    logger.info(f"List variable dynamic: {list(set(dynamic_varible))}")
    
    # Thay thế các biến động bằng giá trị thực
    for key, value in config.dynamic.items():
        pat = f"{{{{{key}}}}}"
        config_str = config_str.replace(pat, value)
    config_dict = json.loads(config_str)
    dynamic_name = config_dict.pop("dynamic", None)
    config = DictConfig(config_dict)
    return config
    
def load_config(config_file: str):
    # Đọc và tải file cấu hình từ đường dẫn
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    logger.info(f"Load config file sucess from `{config_file}`")
    config = DictConfig(config) 
    # Xử lý các biến động trong cấu hình
    config = dynamic_config(config)
    return config

# test
# load_config("core/config/config.yaml")