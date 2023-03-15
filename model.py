import torch
import torch.nn.functional as F
from languageModel import Bert
from languageModel import RoBERTa
import clip
from sklearn.metrics.pairwise import cosine_distances
from PositionalEncoding import PositionalEncoding
from Vit import VisionTransformer


class Net(torch.nn.Module):
    def __init__(self, imgPath, clipModel):
        super(Net, self).__init__()
        self.imgPath = imgPath
        self.clipModel = clipModel
        self.fc1 = torch.nn.Linear(2560+512, 64) # 2560 1536 3584
        self.fc2 = torch.nn.Linear(64, 2)
        self.cliptext = torch.nn.Linear(512, 768)
        self.VisionTransformer = VisionTransformer()
        self.TextBert = RoBERTa()
        self.PositionalEncoding = PositionalEncoding(768, 0.1, 512)

        self.TextImageEncoderLayer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=4)
        self.TextImageTransformerEncoder = torch.nn.TransformerEncoder(self.TextImageEncoderLayer, num_layers=6)
        self.AttributesTransformerEncoder = torch.nn.TransformerEncoder(self.TextImageEncoderLayer, num_layers=6)
        self.multihead_attn = torch.nn.MultiheadAttention(512, 4)
        self.sentiImageFull = torch.nn.Linear(768, 512)

    def forward(self, image, text, ViTimageTensor, bertTextTensor, bertTextMaskTensor, attributesTensor, sentiment):
        a, clipImageSentence = self.clipModel.encode_image(image)  # batch * 50 * 768
        clipAttrSentence, b = self.clipModel.encode_text(text)  # batch * 77 * 512

        clipImageSentimentSentence = self.sentiImageFull(clipImageSentence.float())

        clipImageAndText = torch.cat((clipImageSentimentSentence, clipAttrSentence), dim=1)
        clipSentiment, _ = self.clipModel.encode_text(sentiment)
        query = clipSentiment.permute(1, 0, 2).float()
        key = clipImageAndText.permute(1, 0, 2).float()
        value = key
        attn_output, attn_output_weight = self.multihead_attn(query, key, value)
        attn_output = attn_output.permute(1, 0, 2)[:, -1, :]
        # print(attn_output.shape)

        ViTimageTensor = self.VisionTransformer(ViTimageTensor)  # batch*197*768

        bertTextTensor = self.TextBert(bertTextTensor, bertTextMaskTensor)  # batch*maxlen*768  128

        clipAttrSentence = self.cliptext(clipAttrSentence.float()) # batch * 77 * 768

        c = ViTimageTensor[:, 0, :]
        d = bertTextTensor[:, 0, :]

        ImageWithClip = torch.cat((clipImageSentence, ViTimageTensor), dim=1)
        TextWithClip = torch.cat((clipAttrSentence, bertTextTensor), dim=1)

        ImageWithClip = self.PositionalEncoding(ImageWithClip)
        TextWithClip = self.PositionalEncoding(TextWithClip)

        ImageWithClip = ImageWithClip.permute(1, 0, 2)
        TextWithClip = TextWithClip.permute(1, 0, 2)

        ImageWithClip = self.TextImageTransformerEncoder(ImageWithClip)
        TextWithClip = self.AttributesTransformerEncoder(TextWithClip)

        ImageWithClip = ImageWithClip.permute(1, 0, 2)[:, 0, :]  # 16*768
        TextWithClip = TextWithClip.permute(1, 0, 2)[:, 0, :]  # 16*768

        mergeTensor = torch.cat((a, b, ImageWithClip, TextWithClip, attn_output), dim=1)

        x = F.relu(self.fc1(mergeTensor))
        out = self.fc2(x)
        return out, c, d