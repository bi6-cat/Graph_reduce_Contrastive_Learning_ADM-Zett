from sentence_transformers import SentenceTransformer

# Lớp LLMEncoding dùng để mã hóa văn bản thành vector nhúng bằng mô hình ngôn ngữ
class LLMEncoding(object):
    # Khởi tạo đối tượng với thư mục lưu trữ cache
    def __init__(self, cache_folder):
        self.cache_folder = cache_folder
        self.setup()
        
    # Thiết lập mô hình SentenceTransformer
    def setup(self):
        # Sử dụng mô hình "all-MiniLM-L6-v2" để mã hóa câu thành vector
        self.model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=self.cache_folder)
    
    # Mã hóa văn bản đầu vào thành vector biểu diễn
    def encode(self, text):
        return self.model.encode(text)
