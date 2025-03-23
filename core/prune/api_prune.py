import logging
from core.prune.components.LD import prune_graph_local_degree
from core.prune.components.prune_api import prune_graph_api
from core.prune.components.scan import prune_graph_scan

logger = logging.getLogger(__name__)

# Hàm lấy thuật toán prune dựa vào tên
# Hỗ trợ 2 thuật toán: "ld" (local degree) và "scan"
def get_algorithm_prune(name: str="ld"):
    if name == "ld": return prune_graph_local_degree  # Trả về thuật toán local degree
    elif name == "scan": return prune_graph_scan      # Trả về thuật toán scan
    elif name == "api": return prune_graph_api      # Trả về thuật toán cắt tỉa API
    else: logger.error("Not type of prune")           # Báo lỗi nếu không tìm thấy thuật toán