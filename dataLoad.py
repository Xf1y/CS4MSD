import os
import pickle as pkl


def GetData(dataPath, imagePath, imageAttributesPath):
    imageAttributes = pkl.load(open(imageAttributesPath, "rb"))
    lines = pkl.load(open(dataPath, "rb"))
    train = pkl.load(open('./data/OCR/OCR_train2.pickle', "rb"))
    valid = pkl.load(open('./data/OCR/OCR_test2.pickle', "rb"))

    inputs = []
    labels = []
    for line in lines:
        id = line[0]
        if os.path.exists(imagePath + id + ".jpg"):
            imgAttr = imageAttributes[id]
            imgAttrSentence = ", ".join(imgAttr)
            imgAttrSentence += '.'

            if id in train:
                if len(train[id]) != 0:
                    imgAttrSentence += " ".join(train[id]) + '.'
                else:
                    imgAttrSentence = imgAttrSentence
            else:
                if len(valid[id]) != 0:
                    imgAttrSentence += " ".join(valid[id]) + '.'
                else:
                    imgAttrSentence = imgAttrSentence
            sentence = line[1]
            label = int(line[-1])
            inputs.append([id, sentence, imgAttrSentence])
            labels.append(label)

    return inputs, labels
