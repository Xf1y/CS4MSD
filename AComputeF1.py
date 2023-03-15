def Macro_F(prediction, target):
    TP, FP, FN, TN = 0, 0, 0, 0
    for pre, tar in zip(prediction, target):
        predic = int(pre)
        targ = int(tar)
        if targ == 1:
            if predic == 1:
                TP += 1
            else:
                FN += 1
        else:
            if predic == 1:
                FP += 1
            else:
                TN += 1

    return TP, FP, FN, TN


def computer_F(TP, FP, FN, TN):
    if TP + FP == 0:
        Precision = 0
    else:
        Precision = TP / (TP + FP)

    if TP + FN == 0:
        Recall = 0
    else:
        Recall = TP / (TP + FN)

    if Precision + Recall == 0:
        F1 = 0
    else:
        F1 = 2 * Precision * Recall / (Precision + Recall)
    return Precision, Recall, F1