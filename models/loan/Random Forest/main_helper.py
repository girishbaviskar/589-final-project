from collections import Counter
from unittest import result

import numpy as np
import pandas as pd
def splitArgumentsAndLabel(df_shuffled):
    X = df_shuffled.iloc[:, :-1]
    y = df_shuffled.iloc[:, -1:]
    return X,y

def stratified_k_fold(y):
    k = 10
    folds_pairs = []
    indices_per_class = [np.where(y == _class)[0] for _class in np.unique(y)]
    indices_per_class_shuffled = [np.random.permutation(c_idcs) for c_idcs in indices_per_class]
    class_indexes_split = [np.array_split(_class, k) for _class in indices_per_class_shuffled]
    folds = []
    for (c1, c2) in zip(*class_indexes_split):
        folds.append(np.concatenate((c1, c2)))

    for i in range(k):
        list1 = list(range(k))
        list1.remove(i)
        test_fold = folds[i]
        args = (folds[list1[0]], folds[list1[1]], folds[list1[2]], folds[list1[3]], folds[list1[4]], folds[list1[5]], folds[list1[6]], folds[list1[7]], folds[list1[8]])
        train_folds = np.concatenate(args)
        #train_folds = np.concatenate(np.delete(folds, i, axis=0))
        # train_folds = np.array([])
        # for i in list1:
        #     train_folds = np.concatenate(train_folds, folds[i])
        folds_pairs.append((train_folds, test_fold))

    return folds_pairs

def calculateAccuracy(listOfPredictedLabels, listOfActualLabels):
    if(len(listOfPredictedLabels) == len(listOfActualLabels)):
        correctCount = 0;
        for index, label in enumerate(listOfPredictedLabels):
            #print(label)
            if label == listOfActualLabels[index]:
                correctCount += 1
        return ((correctCount/len(listOfPredictedLabels)) )

def calculate_confusion_matrix(y_test, y_pred):
    tp = tn = fp = fn = 0
    for i in range(len(y_test)):
        if y_test[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_test[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y_test[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_test[i] == 1 and y_pred[i] == 0:
            fn += 1
    return tp, tn, fp, fn

def calculate_precision(tp, fp):
    return tp / (tp + fp)

def calculate_recall(tp, fn):
    return tp / (tp + fn)

def calculate_f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)

def plotGraph(xValues, yValues, title, xLable, yLabel, std_dev):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.xlabel(xLable)
    plt.ylabel(yLabel)
    #plt.plot(xValues, yValues)
    ax.errorbar(xValues, yValues,
            yerr=std_dev,
            fmt='-o', ecolor='black')
    plt.title(title)
    plt.show()
