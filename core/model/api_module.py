# Import các thư viện phân loại từ sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Import các module tùy chỉnh từ dự án
from core.model.components.classify.modelMlp import MLPEngine
from core.model.components.combine_vector.attention import attention_fusion
from core.model.components.combine_vector.concat import concat
from core.model.components.embedding.gnn import ModelModule

# Hàm trả về thuật toán phân loại dựa trên tham số đầu vào
def get_algorithm_classify(tp):
    return {
        "MLP": MLPEngine,
        "RF": RandomForestClassifier,
        "KNN": KNeighborsClassifier,
        "SVM": SVC,
        "DT": DecisionTreeClassifier,
    }.get(tp, "Not Found Module Match")
    
# Hàm trả về thuật toán nhúng dựa trên tham số đầu vào
def get_algorithm_embedding(tp):
    return {
        "gnn": ModelModule,
    }.get(tp, "Not Found Module Match")
    
# Hàm trả về thuật toán kết hợp vector dựa trên tham số đầu vào
def get_algorithm_combine(tp):
    return {
        "attention": attention_fusion,
        "concat": concat,
    }.get(tp, "Not Found Module Match")