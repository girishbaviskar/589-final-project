import numpy as np
from sklearn import datasets
from models.helpers.helper import initialize_weights, read_data_file, splitArgumentsAndLabel, stratified_k_fold, normalize_features, \
    one_hot_encode, calculate_precision, calculate_recall, calculate_f1_score
from models.helpers.backprop_helper import forward_propagation, back_propagation, calculate_reg_gradients, update_weights, \
    make_predictions, calculate_accuracy


def calculate_precision_recall(y_test, y_pred):
    classes = range(10)  # 10 possible values of y_pred and y_test
    num_classes = len(classes)
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

    for i in range(len(y_test)):
        true_label = y_test[i]
        predicted_label = y_pred[i]
        confusion_matrix[true_label][predicted_label] += 1

    precision = []
    recall = []
    for i in range(num_classes):
        TP = confusion_matrix[i][i]
        FP = sum(confusion_matrix[j][i] for j in range(num_classes) if j != i)
        FN = sum(confusion_matrix[i][j] for j in range(num_classes) if j != i)
        if TP + FP == 0:
            precision.append(0)
        else:
            precision.append(TP / (TP + FP))
        if TP + FN == 0:
            recall.append(0)
        else:
            recall.append(TP / (TP + FN))

    mean_precision = sum(precision) / num_classes
    mean_recall = sum(recall) / num_classes
    return mean_precision, mean_recall


def test_back_prop_parkinsons_dataset():

    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_X = digits[0]
    digits_dataset_y = digits[1]
    # N = len(digits_dataset_X)
    #
    # # Prints the 64 attributes of a random digit, its class,
    # # and then shows the digit on the screen
    # digit_to_show = np.random.choice(range(N), 1)[0]
    # print("Attributes:", digits_dataset_X[digit_to_show])
    # print("Class:", digits_dataset_y[digit_to_show])
    #
    # plt.imshow(np.reshape(digits_dataset_X[digit_to_show], (8, 8)))
    # plt.show()

    folds_indexes_list = stratified_k_fold(digits_dataset_y, 10)
    return digits_dataset_X, digits_dataset_y, folds_indexes_list

if __name__ == "__main__":

    # To test Handwritten Digits dataset
    #All HyperParameters for the model
    reg_lambda = 0.01
    neurons_per_layer = [64, 4, 4, 10] # bias terms are not included in this architecture
    iterations = 500
    mini_batch_size = 500
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
    list_of_J_for_each_instance = []
    for train, test in folds_indexes_list:
        print("Training fold ", c)
        final_weights = {}
        weights = initialize_weights(neurons_per_layer)
        J = 0
        cost = 0
        x_train = X[train]
        y_train = y[train]
        x_test = X[test]
        y_test = y[test]
        y_train_encode = one_hot_encode(y_train,10)
        y_test_encode = one_hot_encode(y_test, 10)

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
            list_of_J_for_each_instance.append(cost)
            print('Final (regularized) cost, J, based on the batch training set:', cost)

    #predict

        y_pred = make_predictions(x_test_reg, final_weights, number_of_layers, False)
        acc = calculate_accuracy(y_pred, y_test)
        precision, recall = calculate_precision_recall(y_test, y_pred)
        f1_score = calculate_f1_score(precision, recall)
        print('Accuracy for fold k', c, acc)
        print('f1 score', c, f1_score)
        list_of_accuracies_for_folds.append(acc)
        list_of_f1_for_folds.append(f1_score)
        c += 1
    avg_acc = np.mean(list_of_accuracies_for_folds, axis=0)
    avg_f1 = np.mean(list_of_f1_for_folds, axis=0)
    #plotGraph(range(m), list_of_J_for_each_instance, 'Error(J) vs n_instances for housevotes dataset', "J","N_instances", 0)
    print('Arch of NN: ', neurons_per_layer)
    print('reg lambda:', reg_lambda)
    print('alpha: ', alpha)
    print('mini_batch_size:', mini_batch_size)
    print('iterations:', iterations)
    print('average acc', avg_acc)
    print('average f1', avg_f1)


