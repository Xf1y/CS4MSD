from torch.utils.data import Dataset
import dataLoad
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from transformers import ViTFeatureExtractor, ViTModel
import torch

class MyDataset(Dataset):
    def __init__(self, dataPath, imgPath, imageAttributesPath, preprocess, clip, maxlen):
        data, label = dataLoad.GetData(dataPath, imgPath, imageAttributesPath)
        self.data = data
        self.label = label
        self.imgPath = imgPath
        self.preprocess = preprocess
        self.clip = clip
        self.maxlen = maxlen
        self.bertToken = AutoTokenizer.from_pretrained('xlm-roberta-base')
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    def __getitem__(self, index):
        id = self.data[index][0]
        text = self.data[index][1]
        imageAttributes = self.data[index][2]
        label = self.label[index]

        image = Image.open(self.imgPath + id + '.jpg').convert("RGB")
        imageTensor = self.preprocess(image)
        ViTimageTensor = self.feature_extractor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        textTensor = self.clip.tokenize([text], truncate=True).squeeze(0)

        sentimentTensor = self.clip.tokenize(["angry happy sad surprised bored"]).squeeze(0)

        bertTokenInput = self.bertToken(imageAttributes, max_length=self.maxlen, padding="max_length", return_tensors="pt", truncation=True)
        bertTextTensor = bertTokenInput["input_ids"].squeeze(0)
        bertTextMaskTensor = bertTokenInput["attention_mask"].squeeze(0)

        # newAttributesToken
        attributesTensor = self.clip.tokenize([imageAttributes], truncate=True).squeeze(0)
        newlabel = 1
        if label == 1:
            newlabel = -1

        return imageTensor, textTensor, ViTimageTensor, bertTextTensor, bertTextMaskTensor, attributesTensor, sentimentTensor, label, newlabel

    def __len__(self):
        return len(self.data)