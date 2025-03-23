from androguard.core.bytecodes.apk import APK
import logging

from core.utils.write_read_file import write_txt

logger = logging.getLogger(__name__)

def get_sequence_permission(path, path_save):
    """
        extract permissions from apk
        params:
        path: apk path
        return: str: sequence of permissions
    """
    # Khởi tạo biến chứa chuỗi quyền
    permission_text = None
    try:
        logger.info(f"Processing... file {path}")
        # Đọc file APK
        apk = APK(path)
        # Lấy danh sách quyền
        perms = apk.get_permissions()
        # Xử lý chỉ lấy tên quyền cuối cùng sau dấu chấm
        sqperms = [token.split(".")[-1] for token in perms]
        # Nối các quyền bằng dấu gạch dưới
        sen = "_".join(sqperms)
        # Chuyển về định dạng chuỗi với khoảng cách
        permission_text = " ".join(sen.split("_"))
    except:
        # Xử lý lỗi khi trích xuất quyền
        logger.info(f"Extract permission error in file {path}")
        permission_text = ""
    # Lưu kết quả vào file
    write_txt(path_save, permission_text)
    return permission_text

