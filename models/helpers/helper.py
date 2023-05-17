from sklearn.metrics import accuracy_score

from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import dataset
def read_data_file(path):
    df = pd.read_csv(path , delimiter=',')
    return df

def splitArgumentsAndLabel(df_shuffled):
    X = df_shuffled.iloc[:, :-1]
    y = df_shuffled.iloc[:, -1:]
    return X,y

def stratified_k_fold(y, n_classes):
    k = 10
    folds_pairs = []
    indices_per_class = [np.where(y == _class)[0] for _class in np.unique(y)]
    indices_per_class_shuffled = [np.random.permutation(c_idcs) for c_idcs in indices_per_class]
    class_indexes_split = [np.array_split(_class, k) for _class in indices_per_class_shuffled]
    folds = []

    if n_classes == 2:
        for (c1, c2) in zip(*class_indexes_split):
            folds.append(np.concatenate((c1, c2)))
    if n_classes == 3:
        for (c1, c2, c3) in zip(*class_indexes_split):
            folds.append(np.concatenate((c1, c2, c3)))
    for i in range(k):
        list1 = list(range(k))
        list1.remove(i)
        test_fold = folds[i]
        args = (folds[list1[0]], folds[list1[1]], folds[list1[2]], folds[list1[3]], folds[list1[4]], folds[list1[5]], folds[list1[6]], folds[list1[7]], folds[list1[8]])
        train_folds = np.concatenate(args)
        folds_pairs.append((train_folds, test_fold))

    return folds_pairs

def initialize_weights(neurons_per_layer):
    weights = {}
    for i in range(1, len(neurons_per_layer)):
        layer_input_size = neurons_per_layer[i-1]
        layer_output_size = neurons_per_layer[i]
        weights[f"theta{i}"] = np.random.uniform(low=-1, high=1, size=(layer_output_size, layer_input_size+1))
    return weights

def normalize_features(X, max_vals, min_vals):
    X = X - min_vals

    # Divide by the range of each feature
    range_vals = max_vals - min_vals
    X = X / range_vals

    return X


# def one_hot_encode(y , num_classes):
#     y_binary = np.zeros((len(y), num_classes))
#     for i in range(len(y)):
#         y_binary[i, y[i]] = 1
#
#     return y_binary

def one_hot_encode(y, num_classes):
    unique_labels = np.unique(y)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    y_encoded = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        label_index = label_to_index[y[i]]
        y_encoded[i, label_index] = 1

    return y_encoded

def calculate_precision(tp, fp):
    if tp + fp == 0:
        return 0.0
    else:
        return tp / (tp + fp)

def calculate_recall(tp, fn):
    if tp + fn == 0:
        return 0.0
    else:
        return tp / (tp + fn)

def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    else:
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