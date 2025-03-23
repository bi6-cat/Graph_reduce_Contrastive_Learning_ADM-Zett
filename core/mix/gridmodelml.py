# Import các thư viện cần thiết cho việc đánh giá mô hình
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os

# Hàm huấn luyện và đánh giá một mô hình cụ thể
def grid_model(model, X_train, y_train, X_test, y_test, pathgfig, pathscore, pathpred):
    # Huấn luyện mô hình
    model.fit(X_train, y_train)
    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)
    print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    dis = ConfusionMatrixDisplay(conf_matrix, display_labels=["Benign", "Malware"])
    dis.plot(cmap=plt.cm.Blues,values_format='g')
    plt.savefig(pathgfig)


    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Confusion Matrix:")
    print(conf_matrix)
    # Lưu kết quả đánh giá vào file
    score_answer = f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-score: {f1}"
    with open(pathscore, "w") as f:
        f.write(score_answer)
    # Lưu các dự đoán
    with open(pathpred, "wb") as f:
        np.save(f, y_pred)

# Hàm huấn luyện nhiều mô hình với các tham số khác nhau
def grid_models(estimator, param_grid, args, X_train, y_train, X_test, y_test, pathsave):
    # Duyệt qua các giá trị tham số
    for i in param_grid:
        print(f"Training model with {i} estimators................................................................")
        # Khởi tạo mô hình với tham số cụ thể
        model = estimator(n_estimators=i, **args)
        # Chuẩn bị đường dẫn để lưu kết quả
        folder = f"RF_N{i}"
        pathcommon = os.path.join(pathsave, folder)
        pathgfig = pathcommon + "/confusion.png"
        pathscore = pathcommon + "/score.txt"
        pathpred = pathcommon + "/pred.npy"

        # Tạo thư mục nếu chưa tồn tại và huấn luyện mô hình
        if os.path.exists(pathcommon):
            grid_model(model, X_train, y_train, X_test, y_test, pathgfig, pathscore, pathpred)
        else:
            os.makedirs(pathcommon)
            grid_model(model, X_train, y_train, X_test, y_test, pathgfig, pathscore, pathpred)