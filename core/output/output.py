import os
from typing import List
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import logging
import rootutils
rootutils.setup_root(__file__,
                     indicator=(".project-root", "setup.cfg", "setup.py", ".git", "pyproject.toml"),
                     pythonpath=True)

from core.utils.write_read_file import write_txt
logger = logging.getLogger(__name__)
plt.ioff()

class ScoreOutput(object):
    # Lớp này chứa các phương thức tính toán và xuất điểm số đánh giá mô hình
    
    @staticmethod
    def accuracy_score(y_preds, y_labels):
        # Tính độ chính xác của mô hình
        score = accuracy_score(y_pred=y_preds, y_true=y_labels)
        return round(score, 4)
    
    @staticmethod
    def f1_score(y_preds, y_labels):
        # Tính điểm F1 (cân bằng giữa precision và recall)
        score = f1_score(y_pred=y_preds, y_true=y_labels)
        return round(score, 4)
    
    @staticmethod
    def recall_score(y_preds, y_labels):
        # Tính tỷ lệ recall (khả năng tìm ra các trường hợp dương tính)
        score = recall_score(y_pred=y_preds, y_true=y_labels)
        return round(score, 4)
    
    @staticmethod
    def precision_score(y_preds, y_labels):
        # Tính tỷ lệ precision (độ chính xác của các dự đoán dương tính)
        score = precision_score(y_pred=y_preds, y_true=y_labels)
        return round(score, 4)
    
    @staticmethod
    def confusion_matrix(y_preds, y_labels):
        # Tính ma trận nhầm lẫn để đánh giá chi tiết hiệu suất
        conf_matrix = confusion_matrix(y_pred=y_preds, y_true=y_labels)
        return conf_matrix
    
    @staticmethod
    def get_type_score(tp):
        # Trả về hàm tương ứng với loại đánh giá được yêu cầu
        if tp == "acc": return ScoreOutput.accuracy_score
        elif tp == "f1": return ScoreOutput.f1_score
        elif tp == "precision": return ScoreOutput.precision_score
        elif tp == "recall": return ScoreOutput.recall_score
        elif tp == "confusion": return ScoreOutput.confusion_matrix
        else: 
            logger.error("Not type of score supported")
    
    @staticmethod
    def auto_path_save_score(path, types):
        # Tự động tạo đường dẫn để lưu kết quả đánh giá
        paths = {}
        if "confusion" in types: paths["confusion"] = f"{path}/confusion.png"
        paths["metrics"] = f"{path}/metrics.txt"
        return paths
            
    @staticmethod
    def list_score(types: List[str], y_preds, y_trues, path_save_metrics):
        # Tính toán và lưu tất cả các chỉ số đánh giá được yêu cầu
        os.makedirs(path_save_metrics, exist_ok=True)
        paths = ScoreOutput.auto_path_save_score(path_save_metrics, types)
        metrics = ""
        for tp in types:
            metric_engine = ScoreOutput.get_type_score(tp)
            output = metric_engine(y_preds, y_trues)
            if tp == "confusion":
                # Vẽ và lưu ma trận nhầm lẫn dưới dạng hình ảnh
                dis = ConfusionMatrixDisplay(output, display_labels=["Benign", "Malware"])
                dis.plot(cmap=plt.cm.Blues,values_format='g')
                plt.savefig(paths["confusion"])
            metrics += f"{tp}: {output}\n"
        logger.info(f"Metrics: \n{metrics}")
        write_txt(paths["metrics"], metrics)
    
    @staticmethod
    def data_vizualization(preds, y_trues, path_save_metrics):
        
        os.makedirs(path_save_metrics, exist_ok=True)
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(preds)
        plt.figure(figsize=(8, 6))
        
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='blue', edgecolors='black')
        for i, (x, y) in enumerate(vectors_2d):
            plt.text(x, y, f"{y_trues[i]}", fontsize=8, color='red', ha='right')

        plt.axhline(0, color='gray', linewidth=0.5)
        plt.axvline(0, color='gray', linewidth=0.5)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title("2D Visualization of High-Dimensional Vectors")
        plt.grid(True)
        path_save_image = f"{path_save_metrics}/pca_cl.png"
        plt.savefig(path_save_image, dpi=300, bbox_inches='tight')
        
# Mã thử nghiệm
# pred = [1, 1, 0]
# true = [1, 1, 0]
# ls = ["acc", "confusion", "f1", "recall", "precision"]
# path = "experiments"
# ScoreOutput.list_score(ls, pred, true, path)





