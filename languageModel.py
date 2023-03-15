from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import torch

class RoBERTa(torch.nn.Module):
    def __init__(self):
        super(RoBERTa, self).__init__()
        self.model = AutoModel.from_pretrained("xlm-roberta-base")

    def forward(self, textTensor, maskTensor):
        outputs = self.model(textTensor, token_type_ids=None, attention_mask=maskTensor)
        return outputs.last_hidden_state