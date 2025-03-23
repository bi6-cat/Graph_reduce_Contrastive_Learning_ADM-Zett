from core.embedding_data.components.codebert import EmbeddingCodeBert
from core.utils.capture_output import Capturing

def embedding_codebert(G, output_dim, cache_dir):
    # Khởi tạo mô hình CodeBERT để nhúng mã nguồn
    model_embedding = EmbeddingCodeBert(output_dim=output_dim, cache_dir=cache_dir)    
    mapping_embedd = {}
    with Capturing() as ouput:
        for node in G.nodes():
            try:
                # Lấy mã nguồn của phương thức từ nút
                func = str(node.method.source())     
            except:
                # Trường hợp không lấy được mã nguồn, tạo chuỗi thay thế
                func = "{} {}".format(str(node.class_name), str(node.method.name))
            # Chuyển đổi mã nguồn thành vector nhúng
            vector_embed = model_embedding.encode(func)
            # Lưu trữ vector nhúng cho từng nút
            mapping_embedd[node] = vector_embed
    return mapping_embedd
