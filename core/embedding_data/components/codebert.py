from transformers import AutoTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn

# Mô hình CodeBert tùy chỉnh kế thừa từ BertModel
class CustomCodeBertModel(BertModel):
    def __init__(self, config, n_layer=12, output_dim=768):
        super(CustomCodeBertModel, self).__init__(config)
        # Giới hạn số lớp transformer được sử dụng
        self.encoder.layer = self.encoder.layer[:n_layer]
        
        # Thêm lớp tuyến tính và pooling để xử lý đầu ra
        self.linear = nn.Linear(768, output_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self._init_weights(self.linear)

    # Hàm xử lý đầu vào và tạo embedding
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        last_hidden_state = outputs[0]
        last_hidden_state = self.linear(last_hidden_state)
        last_hidden_state = self.pooling(last_hidden_state.permute(0, 2, 1)).squeeze(-1)
        return last_hidden_state

# Lớp bao bọc cho việc tạo embedding từ mã nguồn sử dụng CodeBert
class EmbeddingCodeBert(object):
    def __init__(self, output_dim, cache_dir: str="./cache"):
        # Tải mô hình pretrained CodeBert từ Microsoft
        original_model = BertModel.from_pretrained("microsoft/codebert-base", cache_dir=cache_dir)
        config = BertConfig.from_pretrained("microsoft/codebert-base", cache_dir=cache_dir)

        self.output_dim = output_dim
        # Khởi tạo mô hình tùy chỉnh
        self.model = CustomCodeBertModel(config, n_layer=12, output_dim=768)
        # Tải tokenizer để xử lý văn bản đầu vào
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", cache_dir=cache_dir)

        # Sao chép trọng số từ mô hình pretrained sang mô hình tùy chỉnh
        with torch.no_grad():
            self.model.embeddings = original_model.embeddings
            for i, layer in enumerate(original_model.encoder.layer):
                if i < len(self.model.encoder.layer):
                    self.model.encoder.layer[i] = layer

    # Mã hóa chuỗi đầu vào thành vector embedding
    def encode(self, sequence_permission):
        in_encode = self.tokenizer(sequence_permission, return_tensors="pt")
        out_encode = self.model(**in_encode)
        return out_encode
