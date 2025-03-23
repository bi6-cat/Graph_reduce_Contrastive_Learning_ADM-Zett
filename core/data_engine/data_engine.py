import glob
import logging
import os
from pathlib import Path
from omegaconf import OmegaConf
import json
from core.data.apk_graph import apk_graph
from core.data.checkpoint_graph import save_graph
from core.prune.api_prune import get_algorithm_prune
from core.utils.capture_output import Capturing
from core.utils.write_read_file import write_txt

logger = logging.getLogger(__name__)

class DataEngine(object):
    @staticmethod
    def generate_dataset(config):
        # Tải module nhúng đồ thị
        try:
            with Capturing() as out:
                from core.embedding_data.api_embedd import embedd_graph
        except:
            logger.error(f"{ImportError('Error during loading embedd_graph')}")
        
        # Hiển thị thông tin cấu hình
        logger.info(f"Config dataset: \n{json.dumps(OmegaConf.to_container(config.dataset), indent=4)}")
        
        # Lấy danh sách file APK để xử lý
        path_apks = glob.glob(config.dataset.path_data + "/*.apk")[:config.dataset.length]
        n_apks = len(path_apks)
        
        # Khởi tạo thuật toán cắt tỉa đồ thị
        prune_engine = get_algorithm_prune(config.dataset.prune.algorithm)
        
        # Biến đếm số lượng node và cạnh
        nnode_original, nnode_prune, nedge_original, nedge_prune = 0, 0, 0, 0
        
        # Xử lý từng file APK
        for idx, apk in enumerate(path_apks):    
            logger.info(f"File: {idx}")
            
            # Chuyển APK thành đồ thị
            logger.info(f"Converting... apk to graph")
            G_original, dg = apk_graph(apk)
            
            # Cắt tỉa đồ thị
            logger.info(f"Pruning... graph")
            if config.dataset.prune.algorithm == "api":
                G_original_prune, dg_prune = prune_engine(G_original, config.dataset.prune.api_file, config.dataset.prune.order)
            else:
                G_original_prune, dg_prune = prune_engine(G_original, config.dataset.prune.target_ratio)
            
            # Nhúng đồ thị
            logger.info(f"Embedding... graph")
            G_original_embed, dg_embed = embedd_graph(G_original_prune, config.dataset.embedding_dim, config.dataset.cache_dir)        
            
            # Lưu kết quả
            logger.info("Saving... ")
            
            # Tính tổng số node và cạnh
            nnode_original += G_original.number_of_nodes()
            nnode_prune += G_original_prune.number_of_nodes()
            nedge_original += G_original.number_of_edges()
            nedge_prune += G_original_prune.number_of_edges()        
            
            # No saved because it's graph too large
            # save_graph_nx(str(dest_file), G_original)
            # Lưu đồ thị
            dest_fcg_file = Path(config.dataset.path_save_fcg) / f'{Path(apk).stem}.fcg'
            dest_fcg_file_prune = Path(config.dataset.path_save_fcg_prune) / f'{Path(apk).stem}.fcg'
            dest_fcg_file_embedding = Path(config.dataset.path_save_fcg_embedding) / f'{Path(apk).stem}.fcg'
            save_graph(str(dest_fcg_file), [dg])
            save_graph(str(dest_fcg_file_prune), [dg_prune])
            save_graph(str(dest_fcg_file_embedding), [dg_embed])
        
        # Tạo báo cáo tổng kết
        metrics = f"Avg number of nodes, edge:  \
                    \n\t Original: {round(nnode_original/n_apks, 4)} nodes - {round(nedge_original/n_apks, 4)} edges \
                    \n\t Pruned: {round(nnode_prune/n_apks, 4)} nodes - {round(nedge_prune/n_apks, 4)} edges"
        logger.info(metrics)
        logger.info(f"Completed process folder {config.dataset.path_data} to: \n\t fcg folder: {config.dataset.path_save_fcg} \n\t fcg_prune folder: {config.dataset.path_save_fcg_prune} \n\t fcg_embed folder: {config.dataset.path_save_fcg_embedding} ")
        
        # tạo thư mục và path file lưu
        os.makedirs(config.dataset.path_save_metrics, exist_ok=True)
        path_to_save_file = os.path.join(config.dataset.path_save_metrics, f"metrics_{config.dataset.prune.algorithm}.txt")
        
        # Lưu số liệu thống kê
        logger.info(f"Saving... metric to file {path_to_save_file}")
        write_txt(path_to_save_file, metrics)
        logger.info(f"Completed Generate Dataset")