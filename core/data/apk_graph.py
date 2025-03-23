import sys
import traceback
from pathlib import Path
import dgl
import networkx as nx
from androguard.misc import AnalyzeAPK
import traceback
from copy import deepcopy
import logging
import rootutils
rootutils.setup_root(__file__,
                     indicator=(".project-root", "setup.cfg", "setup.py", ".git", "pyproject.toml"),
                     pythonpath=True)

logger = logging.getLogger(__name__)
"""
    # Test a sample
    apk_graph("data_storage/raw/benign/001A76B481B894520582735D8375CF7C54049DA425508D3C68A198DCDB4EEC73.apk")
"""

def apk_graph(source_file: str):
    # Hàm này chuyển đổi tệp APK thành đồ thị 
    logger.info(f"Processing ... {source_file}")
    try:
        # Lấy tên tệp không có phần mở rộng
        file_name = Path(source_file).stem
        
        # Phân tích tệp APK bằng androguard
        _, _, dx = AnalyzeAPK(source_file)
        
        # Lấy đồ thị gọi hàm từ APK
        cg = dx.get_call_graph()
        
        # Chuyển đổi sang định dạng đồ thị có hướng của NetworkX
        G_original = nx.DiGraph(cg)
        
        # Tạo bản sao của đồ thị
        G = G_original.copy()
        
        # Chuyển đổi nhãn nút thành số nguyên
        G = nx.convert_node_labels_to_integers(G)
        
        # Chuyển đồ thị NetworkX sang đồ thị DGL
        dg = dgl.from_networkx(G)
        
        # Trả về cả đồ thị gốc và đồ thị DGL
        return G_original, dg
    except:
        # Xử lý lỗi nếu có
        logger.error(f"Error while processing {source_file}")
        traceback.logger.error_exception(*sys.exc_info())
        return None, None
    
# apk_graph("data_storage/raw/benign/001A76B481B894520582735D8375CF7C54049DA425508D3C68A198DCDB4EEC73.apk")
