from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import logging
logger = logging.getLogger(__name__)

# Hàm chọn mô hình học máy dựa vào loại và tham số
def selectML(typeML, param):
  # Random Forest - Rừng ngẫu nhiên
  if typeML == "RF":
    return RandomForestClassifier(max_depth=param, random_state=42)
  # K-Nearest Neighbors - K láng giềng gần nhất
  elif typeML == "KNN":
    return KNeighborsClassifier(n_neighbors=param)
  # Support Vector Machine - Máy vector hỗ trợ
  elif typeML == "SVM":
    return SVC(kernel=param)
  # Decision Tree - Cây quyết định
  elif typeML == "DT":
    return DecisionTreeClassifier(criterion=param)
  # Báo lỗi khi không hỗ trợ loại mô hình
  else: logger.error(f"Not support model classify {typeML}")