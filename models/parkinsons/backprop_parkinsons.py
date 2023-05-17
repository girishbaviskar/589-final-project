import numpy as np
import pandas as pd
from numpy import number
from pip._internal import network

from models.helpers.helper import initialize_weights, read_data_file, splitArgumentsAndLabel, stratified_k_fold, normalize_features, \
    one_hot_encode, calculate_precision, calculate_recall, calculate_f1_score, plotGraph
from models.helpers.backprop_helper import forward_propagation, back_propagation, calculate_reg_gradients, update_weights, \
    make_predictions, calculate_accuracy

# def read_data_file(path):
#     df = pd.read_csv(path, delimiter='\t')
#     return df

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

def test_back_prop_parkinsons_dataset():
    pddf = read_data_file('../../datasets/parkinsons.csv')
    X, y = splitArgumentsAndLabel(pddf)
    #one_hot_encoded_data_X = pd.get_dummies(X, columns=['Gender', 'Married','Education','Self_Employed','Property_Area', 'Dependents'], dtype=int)
    X = X.to_numpy()
    y = y.to_numpy().flatten()
    folds_indexes_list = stratified_k_fold(y, 2)
    return X, y, folds_indexes_list

if __name__ == "__main__":

    # To test House Votes dataset
    #All HyperParameters for the model
    reg_lambda = 0.01
    neurons_per_layer = [22, 4, 4, 4, 2] # bias terms are not included in this architecture
    iterations = 500
    mini_batch_size = 50
    alpha = 0.5


    X, y, folds_indexes_list  = test_back_prop_parkinsons_dataset()

    number_of_layers = len(neurons_per_layer)
    m = len(X)
    J = 0
    acc_grads = {}
    debug_mode = False

    c = 1
    list_of_accuracies_for_folds = []
    list_of_f1_for_folds = []
    list_of_j_for_folds = []
    for train, test in folds_indexes_list:
        print("Training fold ", c)
        list_of_J_for_each_batch = []
        final_weights = {}
        weights = initialize_weights(neurons_per_layer)
        J = 0
        cost = 0
        x_train = X[train]
        y_train = y[train]
        x_test = X[test]
        y_test = y[test]
        y_train_encode = one_hot_encode(y_train,2)
        y_test_encode = one_hot_encode(y_test, 2)

        min_vals = np.min(x_train, axis=0)
        max_vals = np.max(x_train, axis=0)
        #Normalize features
        x_train_reg = normalize_features(x_train, max_vals, min_vals)
        x_test_reg = normalize_features(x_test, max_vals, min_vals)
        for b in range(iterations):
            for i in range(0, m, mini_batch_size):
                J_b = 0
                # process mini-batch of training examples
                x_batch = x_train_reg[i:i + mini_batch_size]
                y_batch = y_train_encode[i:i + mini_batch_size]
                for i, x in  enumerate(x_batch):
                    J_i = 0
                    if debug_mode: print('Processing training instance: ', i + 1)

                    aL, A, Z = forward_propagation(x, weights, number_of_layers, debug_mode)
                    acc_grads = back_propagation(x, y_batch[i], A, weights, alpha, reg_lambda, number_of_layers, 1, acc_grads, debug_mode)

                    if debug_mode : print('Predicted output for instance ', i + 1, aL)
                    if debug_mode : print('Expected output for instance ', i + 1, y[i])

                    J_k = 0
                    if len(aL) > 1:
                        for j, k in enumerate(aL):
                            J_i += -y_batch[i][j] * np.log(k) - (1 - y_batch[i][j]) * np.log(1 - k)
                            #J_k += np.sum(J_class)
                    else:
                        J_i = -y_batch[i] * np.log(aL) - (1 - y_batch[i]) * np.log(1 - aL)
                        #J += np.sum(J_i)
                    if debug_mode : print('Cost, J, associated with instance ', i + 1, J_i)
                    J_b += J_i
                #J_i /= len(mini_batch_size)
                J += J_b
                list_of_J_for_each_batch.append(J_b[0] / mini_batch_size)


                #calculate reg gradients
                reg_grads = calculate_reg_gradients(acc_grads, weights, reg_lambda, number_of_layers, mini_batch_size, debug_mode)

                #update theta values
                final_weights = update_weights(weights, reg_grads, number_of_layers, alpha)
            J /= mini_batch_size
            # Regularization
            S = 0
            for key in weights:
                if len(weights[key].shape) == 2:
                    S += np.sum(np.square(weights[key][:, 1:]))
                else:
                    S += np.sum(np.square(weights[key]))
            S *= reg_lambda / (2 * mini_batch_size)
            cost = J + S
            if debug_mode: print('Final (regularized) cost, J, based on the batch training set:', cost)

    #predict

        y_pred = make_predictions(x_test_reg, final_weights, number_of_layers, False)
        acc = calculate_accuracy(y_pred, y_test)
        tp, tn, fp, fn = calculate_confusion_matrix(y_test, y_pred)
        precision = calculate_precision(tp, fp)
        recall = calculate_recall(tp, fn)
        f1_score = calculate_f1_score(precision, recall)
        print('Accuracy for fold k', c, acc)
        print('f1 score', c, f1_score)
        list_of_accuracies_for_folds.append(acc)
        list_of_f1_for_folds.append(f1_score)
        c += 1
        list_of_j_for_folds.append(list_of_J_for_each_batch)
    avg_acc = np.mean(list_of_accuracies_for_folds, axis=0)
    avg_f1 = np.mean(list_of_f1_for_folds, axis=0)
    mean_list = []
    for row in zip(*list_of_j_for_folds):
        mean = sum(row) / len(row)
        mean_list.append(mean)
    x_axis_val = list(range(len(mean_list)))
    plotGraph(x_axis_val, mean_list, 'Error(J) vs number batch for parkinsons dataset', "N_batches", "J", 0)
    print('Arch of NN: ', neurons_per_layer)
    print('reg lambda:', reg_lambda)
    print('alpha: ', alpha)
    print('mini_batch_size:', mini_batch_size)
    print('iterations:', iterations)
    print('average acc', avg_acc)
    print('average f1', avg_f1)


