# Mô-đun này giúp bắt đầu ra từ hệ thống để sử dụng sau này
from io import StringIO 
import sys

class Capturing(list):
    # Lớp này dùng để bắt đầu ra của stdout và lưu vào danh sách
    def __enter__(self):
        # Lưu stdout gốc và chuyển hướng sang StringIO
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        # Khôi phục stdout và lưu kết quả đã bắt được
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # giải phóng bộ nhớ
        sys.stdout = self._stdout