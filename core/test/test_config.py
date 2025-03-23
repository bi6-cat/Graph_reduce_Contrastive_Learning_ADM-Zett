import os
from omegaconf import DictConfig, ListConfig
import pytest

@pytest.mark.data
@pytest.mark.all
def test_dataset(dataset_config):
    """Hàm kiểm tra cấu hình dataset_config."""
    assert isinstance(dataset_config, DictConfig), "dataset_config phải là một dictionary"

    expected_types = {
        "path_logging": str,
        "length": int,
        "prune": DictConfig,
        "path_data": str,
        "path_save_fcg": str,
        "path_save_fcg_prune": str,
        "path_save_fcg_embedding": str,
        "cache_dir": str,
        "embedding_dim": int,
        "path_save_metrics": str
    }

    for key, expected_type in expected_types.items():
        assert key in dataset_config, f"Thiếu key: {key}"
        assert isinstance(dataset_config[key], expected_type), f"{key} phải là {expected_type}"

    assert dataset_config["length"] > 0, "length phải lớn hơn 0"
    assert dataset_config["embedding_dim"] > 0, "embedding_dim phải lớn hơn 0"

    prune_config = dataset_config["prune"]
    assert isinstance(prune_config["algorithm"], str), "algorithm phải là kiểu string"
    assert prune_config["algorithm"] in ["ld", "scan"], "algorithm phải là 'ld' hoặc 'scan'"
    
    assert isinstance(prune_config["target_ratio"], (float, int)), "target_ratio phải là số"
    assert 0 <= prune_config["target_ratio"] <= 1, "target_ratio phải nằm trong khoảng [0,1]"

    paths = [
        "path_logging", "path_data", "path_save_fcg", "path_save_fcg_prune",
        "path_save_fcg_embedding", "cache_dir", "path_save_metrics"
    ]
    for path_key in paths:
        assert os.path.exists(dataset_config[path_key]), f"Đường dẫn {dataset_config[path_key]} không tồn tại"

@pytest.mark.all
@pytest.mark.model
def test_training(training_config):
    """Hàm kiểm tra cấu hình training_config."""
    assert isinstance(training_config, DictConfig), "training_config phải là một dictionary"

    # Kiểm tra các key chính
    expected_keys = ["path_logging", "device", "learning_rate", "epochs", "log_interval",
                     "data", "model", "path_save_model", "metrics", "path_save_metrics"]
    for key in expected_keys:
        assert key in training_config, f"Thiếu key: {key}"

    # Kiểm tra kiểu dữ liệu của các key chính
    assert isinstance(training_config["path_logging"], str), "path_logging phải là string"
    assert training_config["device"] in ["cpu", "cuda"], "device phải là 'cpu' hoặc 'cuda'"
    assert isinstance(training_config["learning_rate"], float) and training_config["learning_rate"] > 0, "learning_rate phải là số thực dương"
    assert isinstance(training_config["epochs"], int) and training_config["epochs"] > 0, "epochs phải là số nguyên dương"
    assert isinstance(training_config["log_interval"], int) and training_config["log_interval"] > 0, "log_interval phải là số nguyên dương"

    # Kiểm tra dữ liệu `data`
    data_config = training_config["data"]
    assert isinstance(data_config["csv_file"], str), "csv_file phải là string"
    assert isinstance(data_config["batch_size"], int) and data_config["batch_size"] > 0, "batch_size phải là số nguyên dương"
    assert isinstance(data_config["train_test_ratio"], float) and 0 < data_config["train_test_ratio"] < 1, "train_test_ratio phải nằm trong khoảng (0,1)"

    # Kiểm tra `model`
    model_config = training_config["model"]
    valid_models = {"GCN", "SAGE", "GAT", "TAG"}
    assert model_config["typemodel"] in valid_models, f"typemodel phải là một trong {valid_models}"
    assert isinstance(model_config["features"], str), "features phải là string"
    assert isinstance(model_config["infeats"], int) and model_config["infeats"] > 0, "infeats phải là số nguyên dương"
    assert isinstance(model_config["gnn_feats"], ListConfig) and all(isinstance(i, int) and i > 0 for i in model_config["gnn_feats"]), "gnn_feats phải là danh sách số nguyên dương"
    assert isinstance(model_config["fc_feats"], ListConfig) and all(isinstance(i, int) and i > 0 for i in model_config["fc_feats"]), "fc_feats phải là danh sách số nguyên dương"
    assert isinstance(model_config["outclass"], int) and model_config["outclass"] > 0, "outclass phải là số nguyên dương"

    # Kiểm tra `metrics`
    valid_metrics = {"acc", "f1", "recall", "precision", "confusion"}
    assert set(training_config["metrics"]).issubset(valid_metrics), "metrics chứa giá trị không hợp lệ"

    # Kiểm tra các đường dẫn tồn tại (nếu chứa `{{name_experiment}}`, bỏ qua kiểm tra)
    paths = ["path_logging", "data.csv_file", "path_save_model", "path_save_metrics"]
    for path_key in paths:
        keys = path_key.split(".")
        path = training_config[keys[0]] if len(keys) == 1 else training_config[keys[0]][keys[1]]
        if "{{name_experiment}}" not in path:
            assert os.path.exists(path), f"Đường dẫn {path} không tồn tại"

@pytest.mark.all
@pytest.mark.gnn_embedding
def test_embedding_gnn(gnn_embedding_config):
    """Hàm kiểm tra cấu hình gnn_embedding_config."""
    assert isinstance(gnn_embedding_config, DictConfig), "gnn_embedding_config phải là một dictionary"

    expected_types = {
        "path_logging": str,
        "model_checkpoint": str,
        "path_data": str,
        "length": int,
        "path_save_embedding": str
    }

    for key, expected_type in expected_types.items():
        assert key in gnn_embedding_config, f"Thiếu key: {key}"
        assert isinstance(gnn_embedding_config[key], expected_type), f"{key} phải là {expected_type}"

    assert gnn_embedding_config["length"] > 0, "length phải lớn hơn 0"

    paths = ["path_logging", "model_checkpoint", "path_data", "path_save_embedding"]
    for path_key in paths:
        assert os.path.exists(gnn_embedding_config[path_key]), f"Đường dẫn {gnn_embedding_config[path_key]} không tồn tại"


# @pytest.mark.all
@pytest.mark.permission
def test_permission(permission_config):
    """Hàm kiểm tra cấu hình permission_config."""
    assert isinstance(permission_config, DictConfig), "permission_config phải là một dictionary"

    expected_types = {
        "path_logging": str,
        "cache_dir": str,
        "length": int,
        "path_data": str,
        "path_save_text": str,
        "path_save_embedding": str
    }

    for key, expected_type in expected_types.items():
        assert key in permission_config, f"Thiếu key: {key}"
        assert isinstance(permission_config[key], expected_type) or permission_config[key] is None, f"{key} phải là {expected_type} hoặc None"

    assert permission_config["length"] > 0, "length phải lớn hơn 0"

    paths = ["path_logging", "cache_dir", "path_data", "path_save_text", "path_save_embedding"]
    for path_key in paths:
        if permission_config[path_key]:
            assert os.path.exists(permission_config[path_key]), f"Đường dẫn {permission_config[path_key]} không tồn tại"

@pytest.mark.all
@pytest.mark.classify
def test_classify(classify_config):
    """Hàm kiểm tra cấu hình classify_config."""
    assert isinstance(classify_config, DictConfig), "classify_config phải là một dictionary"

    expected_keys = ["path_logging", "data", "classify_algorithm", "metrics", "path_to_save_experiment"]
    for key in expected_keys:
        assert key in classify_config, f"Thiếu key: {key}"

    assert isinstance(classify_config["path_logging"], str), "path_logging phải là string"
    assert isinstance(classify_config["data"], DictConfig), "data phải là dictionary"
    assert isinstance(classify_config["classify_algorithm"], DictConfig), "classify_algorithm phải là dictionary"
    assert isinstance(classify_config["metrics"], ListConfig), "metrics phải là danh sách"
    assert isinstance(classify_config["path_to_save_experiment"], str), "path_to_save_experiment phải là string"

    data_config = classify_config["data"]
    assert isinstance(data_config["reload"], bool), "reload phải là boolean"
    assert isinstance(data_config["embedding"], ListConfig), "embedding phải là danh sách"
    assert set(data_config["embedding"]).issubset({"gnn", "permission"}), "embedding chỉ được chứa 'gnn' và 'permission'"
    assert isinstance(data_config["path_data_label_hash"], str), "path_data_label_hash phải là string"
    assert isinstance(data_config["path_data_gnn"], str), "path_data_gnn phải là string"
    assert isinstance(data_config["path_data_permission"], str), "path_data_permission phải là string"
    assert data_config["combine_algorithm"] in ["concat"], "combine_algorithm phải là 'concat'"
    assert isinstance(data_config["length"], int) and data_config["length"] > 0, "length phải là số nguyên dương"
    assert isinstance(data_config["train_test_ratio"], float) and 0 < data_config["train_test_ratio"] < 1, "train_test_ratio phải nằm trong khoảng (0,1)"
    assert isinstance(data_config["path_save_data_combine"], str), "path_save_data_combine phải là string"

    classify_algo = classify_config["classify_algorithm"]
    assert isinstance(classify_algo["run"], bool), "run phải là boolean"
    assert isinstance(classify_algo["algorithms"], ListConfig), "algorithms phải là danh sách"
    valid_algorithms = {"MLP", "RF", "KNN", "SVM", "DT"}
    assert set(classify_algo["algorithms"]).issubset(valid_algorithms), "algorithms chỉ được chứa 'MLP', 'RF', 'KNN', 'SVM', 'DT'"

    if "MLP" in classify_algo["algorithms"]:
        mlp_config = classify_algo["MLP"]
        assert isinstance(mlp_config["input_size"], int) and mlp_config["input_size"] > 0, "MLP input_size phải là số nguyên dương"
        assert isinstance(mlp_config["hidden_sizes"], ListConfig), "MLP hidden_sizes phải là danh sách"
        assert isinstance(mlp_config["output_size"], int) and mlp_config["output_size"] > 0, "MLP output_size phải là số nguyên dương"
        assert mlp_config["device"] in ["cpu", "cuda"], "MLP device phải là 'cpu' hoặc 'cuda'"
        assert isinstance(mlp_config["lr"], float) and mlp_config["lr"] > 0, "MLP lr phải lớn hơn 0"
        assert isinstance(mlp_config["epochs"], int) and mlp_config["epochs"] > 0, "MLP epochs phải lớn hơn 0"

    if "RF" in classify_algo["algorithms"]:
        rf_config = classify_algo["RF"]
        assert isinstance(rf_config["max_depth"], int) and rf_config["max_depth"] > 0, "RF max_depth phải là số nguyên dương"
        assert isinstance(rf_config["random_state"], int), "RF random_state phải là số nguyên"

    if "KNN" in classify_algo["algorithms"]:
        knn_config = classify_algo["KNN"]
        assert isinstance(knn_config["n_neighbors"], int) and knn_config["n_neighbors"] > 0, "KNN n_neighbors phải là số nguyên dương"

    if "SVM" in classify_algo["algorithms"]:
        svm_config = classify_algo["SVM"]
        assert svm_config["kernel"] in ["linear", "poly", "rbf", "sigmoid"], "SVM kernel phải là 'linear', 'poly', 'rbf', hoặc 'sigmoid'"

    if "DT" in classify_algo["algorithms"]:
        dt_config = classify_algo["DT"]
        assert dt_config["criterion"] in ["gini", "entropy"], "DT criterion phải là 'gini' hoặc 'entropy'"

    valid_metrics = {"acc", "f1", "recall", "precision", "confusion"}
    assert set(classify_config["metrics"]).issubset(valid_metrics), "metrics chứa giá trị không hợp lệ"

    paths = [
        "path_logging",
        "data.path_data_label_hash",
        "data.path_data_gnn",
        "data.path_data_permission",
        "data.path_save_data_combine",
        "path_to_save_experiment"
    ]
    for path_key in paths:
        keys = path_key.split(".")
        path = classify_config[keys[0]] if len(keys) == 1 else classify_config[keys[0]][keys[1]]
        assert os.path.exists(path), f"Đường dẫn {path} không tồn tại"
