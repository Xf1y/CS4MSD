import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from dataset import MyDataset
import clip
import numpy as np
import os
import random
from transformers import get_linear_schedule_with_warmup
from model import Net
import log
from FocalLoss import FocalLoss
import AComputeF1

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# setting random seed
seed = 20
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

batchSize = 16
trainPath = "./data/text/newTrain.pickle"
testPath = "./data/text/newTest.pickle"
imagePath = "./data/dataset_image/"
imageAttributesPath = "./data/text_preprocessing/imageAttributes.pickle"
maxlen = 128

learning_rate = 2e-5
# learning_rate = 1e-3
weight_decay = 1e-2
epsilon = 5e-3
epochs = 10

logPath = "./log.txt"

# Get data and prepare batchs

device = "cuda:0" if torch.cuda.is_available() else "cpu"
ClipModel, preprocess = clip.load("ViT-B/32", device=device)

trainData = MyDataset(trainPath, imagePath, imageAttributesPath, preprocess, clip, maxlen)
validData = MyDataset(testPath, imagePath, imageAttributesPath, preprocess, clip, maxlen)

trainLoader = DataLoader(trainData, batch_size=batchSize, num_workers=4, pin_memory=True, shuffle=True)
testLoader = DataLoader(validData, batch_size=batchSize, num_workers=4, pin_memory=True, shuffle=False)

model = Net(imagePath, ClipModel)
model.to(device)

criterion = FocalLoss(2)
criterion2 = torch.nn.CosineEmbeddingLoss()

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': weight_decay
     },
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0
     }
]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr = learning_rate, eps = epsilon)


# training steps: [number of batches] x [number of epochs].
total_steps = len(trainLoader) * epochs
# 设计 learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=total_steps)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(trainLoader, 0):
        img_tensor, text_tensor, ViTimageTensor, bertTextTensor, bertTextMaskTensor, attributesTensor, sentiment, target, newlabel = data 
        img_tensor, text_tensor, ViTimageTensor, bertTextTensor, bertTextMaskTensor, attributesTensor, sentiment = img_tensor.to(device), text_tensor.to(device), ViTimageTensor.to(device), bertTextTensor.to(
            device), bertTextMaskTensor.to(device), attributesTensor.to(device), sentiment.to(device)
        target = target.to(device)
        newlabel = newlabel.to(device)
        optimizer.zero_grad()

        outputs, img, txt = model(img_tensor, text_tensor, ViTimageTensor, bertTextTensor, bertTextMaskTensor, attributesTensor, sentiment)
        loss = criterion(outputs, target)
        loss2 = criterion2(img, txt, newlabel)

        loss = 0.85*loss + 0.15*loss2

        loss.backward()

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        if epoch + 1 < 5:
            if batch_idx % 123 == 122:
                log.xie('[%d, %5d] loss: %.4f' % (epoch + 1, batch_idx + 1, running_loss / 123), logPath)
                trial()
                model.train()
                
                running_loss = 0.0
        else:
            if batch_idx % 10 == 9:
                log.xie('[%d, %5d] loss: %.4f' % (epoch + 1, batch_idx + 1, running_loss / 10), logPath)
                trial()
                model.train()
                running_loss = 0.0


def trial():
    model.eval()
    correct = 0
    total = 0
    TP, FP, FN, TN = 0, 0, 0, 0
    with torch.no_grad():
        for data in testLoader:
            img_tensor, text_tensor, ViTimageTensor, bertTextTensor, bertTextMaskTensor, attributesTensor, sentiment, target, _ = data
            img_tensor, text_tensor, ViTimageTensor, bertTextTensor, bertTextMaskTensor, attributesTensor, sentiment = img_tensor.to(
                device), text_tensor.to(device), ViTimageTensor.to(device), bertTextTensor.to(
                device), bertTextMaskTensor.to(device), attributesTensor.to(device), sentiment.to(device)
            target = target.to(device)

            outputs, _, _ = model(img_tensor, text_tensor, ViTimageTensor, bertTextTensor, bertTextMaskTensor, attributesTensor, sentiment)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            TP1, FP1, FN1, TN1 = AComputeF1.Macro_F(predicted, target)
            TP += TP1
            FP += FP1
            FN += FN1
            TN += TN1
    Precision, Recall, F1 = AComputeF1.computer_F(TP, FP, FN, TN)
    log.xie('recall: %.4f, precision: %.4f, f1: %.4f, acc: %.4f' % (Recall, Precision, F1, (correct / total)), logPath)


if __name__ == '__main__':
    for epoch in range(epochs):
        train(epoch)
        trial()
        model.train()