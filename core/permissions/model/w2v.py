from gensim.models import Word2Vec
import numpy as np

class W2VEncoding(object):
    """
    Lớp để mã hóa văn bản thành vector sử dụng mô hình Word2Vec
    """
    def __init__(self, vocab_file, vector_size=100, window=5, min_count=1, epochs=10):
        """
        Khởi tạo đối tượng W2VEncoding
        
        Tham số:
            vocab_file: Đường dẫn đến file từ vựng
            vector_size: Kích thước vector đầu ra
            window: Số từ xung quanh từ hiện tại để xem xét
            min_count: Số lần xuất hiện tối thiểu của từ để được xem xét
            epochs: Số lần lặp lại huấn luyện
        """
        self.vocab_file = vocab_file
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.read_corpus()
        self.fit()
        
    def read_corpus(self):
        """
        Đọc và xử lý dữ liệu từ file từ vựng để tạo các câu đầu vào
        """
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            self.sentences = [line.strip().split() for line in f.readlines()]
    
    def fit(self):
        """
        Huấn luyện mô hình Word2Vec với dữ liệu đầu vào
        """
        self.model = Word2Vec(self.sentences, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=4)
        self.model.train(self.sentences, total_examples=len(self.sentences), epochs=self.epochs)
        
    def encode(self, text):
        """
        Mã hóa đoạn văn bản thành vector
        
        Tham số:
            text: Văn bản đầu vào cần mã hóa
            
        Trả về:
            Vector biểu diễn của đoạn văn bản (trung bình của các vector từ)
        """
        words = text.strip().split()
        word_vectors = [self.model.wv[word] for word in words if word in self.model.wv]
        if word_vectors:
            return sum(word_vectors) / len(word_vectors)
        return np.zeros(self.vector_size)