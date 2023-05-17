import numpy as np
import random
import math
import pandas as pd
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, all_features=None, minimal_size_for_split = 2, maximal_depth = 100, attribute_types = None, attribute_type_dict = None):
        self.all_features = all_features
        self.root = None
        self.attributes_to_test = all_features
        self.ori_dataset_entropy = None
        self.minimal_size_for_split = minimal_size_for_split
        self.maximal_depth = maximal_depth
        self.attribute_types = attribute_types
        self.attribute_type_dict = attribute_type_dict

    def fit(self, X, y):
        self.root = self.tree_builder(X, y)

    def tree_builder(self, X, y, depth=0):

        unique_label_list, number_of_unique_labels = np.unique(y, return_counts=True)
        if len(number_of_unique_labels) == 1:
            return Node(value=y[0])

        if len(X) < self.minimal_size_for_split or depth > self.maximal_depth :
            common_value = self.get_most_frequent_label(y)
            return Node(value=common_value)

        m = int(math.sqrt(len(self.all_features)))
        all_feature_list = self.all_features.tolist()
        self.attributes_to_test = random.sample(all_feature_list, m)
        feat_to_split, spit_threshold = self.calculate_best_split(X, y, self.attributes_to_test, self.attribute_types)
        feat_index = np.where(self.all_features == feat_to_split)[0][0]
        left_indices_list, right_indices_list = self.create_branches(X[:, feat_index], spit_threshold)
        len_of_left = len(X[left_indices_list, :])
        len_of_right = len(X[right_indices_list, :])
        #todo instead of creating parent as leaf make the branch as leaf and create other 2 branches
        # if (len_of_left == 0 or len_of_right == 0):
        #     common_value = self.get_most_frequent_label(y)
        #     #print('common value : ' + str(common_value))
        #     return Node(value=common_value)

        # left_tree = self.tree_builder(X[left_indices_list, :], y[left_indices_list], depth + 1)
        # right_tree = self.tree_builder(X[right_indices_list, :], y[right_indices_list], depth + 1)

        if len_of_left == 0:
            common_value = self.get_most_frequent_label(y)
            left_tree = Node(value=common_value)
        else:
            left_tree = self.tree_builder(X[left_indices_list, :], y[left_indices_list], depth + 1)

        if len_of_right == 0:
            common_value = self.get_most_frequent_label(y)
            right_tree = Node(value=common_value)
        else:
            right_tree = self.tree_builder(X[right_indices_list, :], y[right_indices_list], depth + 1)

        return Node(self.all_features[feat_index], spit_threshold, left_tree, right_tree)

    def get_most_frequent_label(self, y):
        # set_of_y = set(y)
        # return max(y, key=y.count)
        # print('y shape')
        # print(y.shape)
        #y = y[:, 0]
        #print('y shape')
        #print(y.shape)
        return np.bincount(y).argmax()

    def calculate_best_split(self, X, y, attributes_to_test, attribute_types):
        max_gain = -float("inf")
        split_fearure_name = None
        threshold_to_return = None
        for index, feature in enumerate(attributes_to_test) :
            feat_index = np.where(self.all_features == feature)[0]
            x_col = X[:, feat_index]
            if self.attribute_type_dict[feature] == 'n':
                means = []
                unique_x_col = np.unique(x_col)
                sorted_x_col = np.sort(unique_x_col)
                if len(sorted_x_col) == 1:
                    threshold_possible_values = sorted_x_col
                else:
                    for i in range(len(sorted_x_col) - 1):
                        mean = (sorted_x_col[i] + sorted_x_col[i + 1]) / 2
                        means.append(mean)
                    threshold_possible_values = means
                for threshold in threshold_possible_values:
                    info_gain = self.get_info_gain_numerical(x_col, y, threshold)
                    if (info_gain > max_gain):
                        max_gain = info_gain
                        split_fearure_name = feature
                        threshold_to_return = threshold
            else:
                info_gain = self.get_info_gain_categorical(x_col, y)
                if (info_gain > max_gain):
                    max_gain = info_gain
                    split_fearure_name = feature


        return split_fearure_name , threshold_to_return

    def get_info_gain_numerical(self, x_col, y, threshold):

        left, right = self.create_branches(x_col, threshold)
        left_length = len(left)
        right_length = len(right)
        y_length = len(y)

        if left_length == 0 or right_length == 0:
            return 0
        left_entropy = self.calculate_entropy(y[left])
        right_entropy = self.calculate_entropy(y[right])

        final_entropy = (left_length / y_length) * left_entropy +  (
                    right_length / y_length) * right_entropy

        return self.ori_dataset_entropy - final_entropy

    # def get_info_gain_numerical(self, x_col, y):
    #     # ori_entropy = self.get_entropy_of_original_data_set(y)
    #
    #     left, middle, right = self.create_branches(x_col)
    #     # print(y[right])
    #     # print(x_col.shape)
    #     # print(x_col)
    #     # print('left = ' + str(left) )
    #     left_length = len(left)
    #     middle_length = len(middle)
    #     right_length = len(right)
    #     y_length = len(y)
    #
    #     if left_length == 0 or middle_length == 0 or right_length == 0:
    #         return 0
    #     left_entropy = self.calculate_entropy(y[left])
    #     middle_entropy = self.calculate_entropy(y[middle])
    #     right_entropy = self.calculate_entropy(y[right])
    #
    #     final_entropy = (left_length / y_length) * left_entropy + (middle_length / y_length) * middle_entropy + (
    #             right_length / y_length) * right_entropy
    #
    #     return self.ori_dataset_entropy - final_entropy

    def get_info_gain_categorical(self, x_col, y):
        categories = set(y)
        y_length = len(y)

        entropies = []
        for category in categories:
            indices = [i for i, val in enumerate(y) if val == category]
            subset_entropy = self.calculate_entropy(y[indices])
            subset_length = len(indices)
            entropies.append((subset_length / y_length) * subset_entropy)

        final_entropy = sum(entropies)
        return self.ori_dataset_entropy - final_entropy

    def create_branches(self, x_col, threshold):
        left_indices_list = np.argwhere(x_col <=  threshold).flatten()
        right_indices_list = np.argwhere(x_col > threshold).flatten()
        return left_indices_list, right_indices_list

    def get_entropy_of_original_data_set(self, y):
        freq = np.bincount(y)
        probs = freq / len(y)
        ent = 0
        for prob in probs:
            if prob > 0:
                ent -= prob * np.log2(prob)
        return ent

    def calculate_entropy(self, label_values):
        # find unique labels and count
        entropy = 0
        unique_list, counts = np.unique(label_values, return_counts=True)
        #print(counts)
        # print(unique_list)
        for i in range(len(counts)):
            #print(str(i))
            prob_of_i = counts[i] / len(label_values)
            # print('prob_of_i ' + str(prob_of_i))
            entropy += -prob_of_i * np.log2(prob_of_i)
        # print('entropy ' + str(entropy))
        return entropy

    def run_tree(self, x, node):
        #print('node: ' + str(node))
        if node.value is not None:
            return node.value
        feat_index = np.where(self.all_features == node.feature)[0][0]
        # print('feat index')
        value_of_feature_for_x = x[feat_index]
        if value_of_feature_for_x <= node.threshold:
            return self.run_tree(x, node.left)
        else:
            return self.run_tree(x, node.right)

    def predict(self, X_data):
        predictions_list = []
        for x in X_data:
            pred = self.run_tree(x, self.root)
            predictions_list.append(pred)
        return predictions_list


