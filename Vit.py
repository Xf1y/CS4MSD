import torch
from transformers import ViTFeatureExtractor, ViTModel

class VisionTransformer(torch.nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    def forward(self, visual_embeds):
        result = self.model(visual_embeds)
        return result.last_hidden_state 