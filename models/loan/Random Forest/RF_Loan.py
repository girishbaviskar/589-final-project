from sklearn.metrics import accuracy_score

from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import loan_decision_tree as dt_helper
import main_helper
#import dataset
def read_data_file():
    df = pd.read_csv('../../../datasets/loan.csv')
    return df

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

class RandomForest:
    def __init__(self, n_trees=100, maximal_depth=20, minimal_size_for_split=10, minimal_gain = 0.1, columns=None, attribute_type_dict = None ):
        self.n_trees = n_trees
        self.maximal_depth = maximal_depth
        self.minimal_size_for_split = minimal_size_for_split
        self.minimal_gain = minimal_gain
        self.columns = columns
        self.attribute_type_dict = attribute_type_dict
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        self.predictions_by_trees = []
        attribute_type = 'numerical'
        attribute_types = {key: attribute_type for key in self.columns}
        final_predictions = []
        for i in range(self.n_trees):
            tree = dt_helper.DecisionTree(all_features = columns, minimal_size_for_split = self.minimal_size_for_split, attribute_types = attribute_types, maximal_depth = self.maximal_depth, attribute_type_dict = self.attribute_type_dict)
            tree.ori_dataset_entropy = tree.get_entropy_of_original_data_set(y)
            #tree = DecisionTree(max_depth=self.max_depth)
            idxs = np.random.choice(len(X), size=len(X), replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            #predictions = tree.predict(X)
            #self.predictions_by_trees.append(predictions)

    def predict(self, X):
        tree_predictions = []
        for tree in self.trees:
            pred = tree.predict(X)
            tree_predictions.append(pred)

        #return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_predictions)
        result = []
        for i in range(len(tree_predictions[0])):
            counter = Counter()
            for sublist in tree_predictions:
                counter[sublist[i]] += 1
            # determine the most frequent value for this index and add it to the result list
            result.append(counter.most_common(1)[0][0])
        #print(result)
        return result


if __name__ == "__main__":
    pddf = read_data_file()
    X, y = splitArgumentsAndLabel(pddf)
    #drop loan_ID
    X = X.drop('Loan_ID', axis=1)


    minimal_size_for_split = 5
    minimal_gain = 0.1
    maximal_depth = 100
    columns = X.columns.values
    # create attribute type dictionary
    attributes_type_list = ['c', 'c', 'c', 'c', 'c', 'n', 'n', 'n', 'n', 'c', 'c']
    attribute_type_dict = dict(zip(columns, attributes_type_list))
    X = X.to_numpy()
    y = y.to_numpy().flatten()
    mapping = {'N': 0, 'Y': 1}
    y = np.array([mapping[label] for label in y])
    n_samples = X.shape[0]
    permutation = np.random.permutation(n_samples)  # generate a random permutation of the row indices

    X_shuffled = X[permutation]  # shuffle the rows of X
    y_shuffled = y[permutation]  # shuffle the rows of y in the same way
    folds_indexes_list = stratified_k_fold(y)
    n_trees_list = [1, 5, 10, 20, 30, 40, 50]
    accuracies_for_folds = []
    precision_for_folds = []
    recall_for_folds = []
    f1_score_for_folds = []
    mean_acc_values_list = []
    mean_precision_values_list = []
    mean_recall_values_list = []
    mean_f1_score_values_list = []
    for i in range(10):
        print("running iteration: ", i)
        k = 1
        for train, test in folds_indexes_list:
            print("running fold: ", k)
            x_train = X[train]
            y_train = y[train]
            x_test = X[test]
            y_test = y[test]
            accuracies_list = []
            precision_list = []
            recall_list = []
            f1_score_list = []
            for n_trees in n_trees_list:
                rf = RandomForest(n_trees=n_trees, maximal_depth=maximal_depth, minimal_size_for_split=minimal_size_for_split, columns=columns, attribute_type_dict = attribute_type_dict)

                rf.fit(x_train, y_train)
                y_pred = rf.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                tp, tn, fp, fn = main_helper.calculate_confusion_matrix(y_test, y_pred)
                precision = main_helper.calculate_precision(tp, fp)
                recall = main_helper.calculate_recall(tp, fn)
                f1_score = main_helper.calculate_f1_score(precision, recall)
                accuracies_list.append(accuracy)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_score_list.append(f1_score)
            accuracies_for_folds.append(accuracies_list)
            precision_for_folds.append(precision_list)
            recall_for_folds.append(recall_list)
            f1_score_for_folds.append(f1_score_list)
            k += 1
        mean_values = np.mean(accuracies_for_folds, axis=0)
        mean_acc_values_list.append(mean_values)
        prec_mean_values = np.mean(precision_for_folds, axis=0)
        mean_precision_values_list.append(prec_mean_values)
        rec_mean_values = np.mean(recall_for_folds, axis=0)
        mean_recall_values_list.append(rec_mean_values)
        f1_mean_values = np.mean(f1_score_for_folds, axis=0)
        mean_f1_score_values_list.append(f1_mean_values)

    final_acc_mean_values = np.mean(mean_acc_values_list, axis=0)
    final_prec_mean_values = np.mean(mean_precision_values_list, axis=0)
    final_rec_mean_values = np.mean(mean_recall_values_list, axis=0)
    final_f1_mean_values = np.mean(mean_f1_score_values_list, axis=0)
    for val in final_acc_mean_values:
        print('acc: ', val)
    main_helper.plotGraph(n_trees_list, final_acc_mean_values, 'Accuracy vs n_trees for cancer dataset(Info Gain)', "n_trees",
                          "Accuracy", 0)
    main_helper.plotGraph(n_trees_list, final_prec_mean_values, 'Precision vs n_trees for cancer dataset(Info Gain)', "n_trees",
                          "Precision", 0)
    main_helper.plotGraph(n_trees_list, final_rec_mean_values, 'Recall vs n_trees for cancer dataset(Info Gain)', "n_trees",
                          "Recall", 0)
    main_helper.plotGraph(n_trees_list, final_f1_mean_values, 'F1-score vs n_trees for cancer dataset(Info Gain)', "n_trees",
                          "F1-score", 0)