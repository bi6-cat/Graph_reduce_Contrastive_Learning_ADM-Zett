import json
from pathlib import Path
import pandas as pd
import torch
import logging
logger = logging.getLogger(__name__)

# Đọc file text từ đường dẫn
def read_txt(path):
    with open(path, "r") as file:
        text = file.read()
    logger.info(f"completed read text: {Path(path).name}!!!!!")
    return text

# Ghi nội dung text vào file
def write_txt(path, text):
    with open(path, "w") as file:
        file.write(text)
    logger.info(f"write successfully text: {Path(path).name}")
        
# Đọc file CSV và trả về DataFrame
def read_csv(path):
    df = pd.read_csv(path)
    logger.info(f"completed read csv: {Path(path).name}!!!!!")
    return df

# Lưu DataFrame vào file CSV
def write_csv(df, path):
    df.to_csv(path, index=False)
    logger.info(f"write successfully csv: {Path(path).name}")

# Lưu embedding tensor vào file
def write_embedding(embed, path):
    torch.save(embed, path)
    logger.info(f"write successfully embed: {Path(path).name}")

# Đọc embedding tensor từ file
def read_embedding(path):
    embed = torch.load(path)
    logger.info(f"completed read embed: {Path(path).name}!!!!!")
    return embed

# Đọc file JSON và trả về dữ liệu
def read_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    logger.info(f"completed read json: {Path(path).name}!!!!!")
    return data

# Ghi dữ liệu vào file JSON
def write_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)
    logger.info(f"write successfully json: {Path(path).name}")