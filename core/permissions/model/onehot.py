import numpy as np
from sklearn.preprocessing import OneHotEncoder
import rootutils

# Thiết lập đường dẫn gốc cho dự án
rootutils.setup_root(__file__,
                     indicator=(".project-root", "setup.cfg", "setup.py", ".git", "pyproject.toml"),
                     pythonpath=True)
from core.utils.write_read_file import read_txt

class OneHotEncoding(object):
    """
    Lớp thực hiện mã hóa One-Hot cho văn bản dựa trên từ điển có sẵn
    """
    def __init__(self, vocab_file):
        """
        Khởi tạo đối tượng OneHotEncoding
        
        Tham số:
            vocab_file: Đường dẫn tới tệp chứa từ điển
        """
        self.vocab_file = vocab_file
        self.prepare_vocab()
        self.fit()
        
    def prepare_vocab(self):
        """
        Đọc và chuẩn bị từ điển từ tệp
        """
        text = read_txt(self.vocab_file)
        self.vocabs = text.strip().split("\n")            
    
    def fit(self):
        """
        Khởi tạo và huấn luyện mô hình OneHotEncoder với từ điển đã cho
        """
        self.model = OneHotEncoder(categories=[self.vocabs], sparse=False, handle_unknown="ignore")
        self.model.fit_transform(np.array(self.vocabs).reshape(-1, 1))
        
    def processing_text(self, text):
        """
        Tiền xử lý văn bản thành danh sách các token
        
        Tham số:
            text: Văn bản cần xử lý
            
        Trả về:
            Danh sách các token
        """
        return text.strip().split(" ")
    
    def encode(self, text):
        """
        Mã hóa văn bản thành vector one-hot
        
        Tham số:
            text: Văn bản cần mã hóa
            
        Trả về:
            Vector one-hot đại diện cho văn bản
        """
        tokens = self.processing_text(text)
        return self.model.transform(np.array(tokens).reshape(-1, 1)).sum(axis=0)



