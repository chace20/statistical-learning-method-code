import numpy as np
import pandas as pd
from math import log2


class Node:
    def __init__(self):
        self.feature = None
        self.label = None
        self.child = {}  # key是特征值，value是Node


class DecisionTree:
    def __init__(self):
        self.root_node = None
        self.threshold = 0

    # ID3
    def compute_info_gain(self, data, feature):
        uni, counts = np.unique(data['label'], return_counts=True)
        entropy = - np.sum(counts[i]/len(data) * log2(counts[i]/len(data)) for i in range(len(uni)))

        # counts_dict: {feature_value: [feature_value_count, {label_value: label_value_count}]}
        counts_dict = {}
        for row in data.iterrows():
            row = row[1]
            # setdefault会有点难懂
            counts_dict.setdefault(row[feature], [0, {row['label']: 0}])[0] += 1
            counts_dict[row[feature]][1].setdefault(row['label'], 0)
            counts_dict[row[feature]][1][row['label']] += 1

        entropy_feature = 0
        for v in counts_dict.values():
            D_i = v[0]
            entropy_i = - np.sum(D_ik/D_i*log2(D_ik/D_i) for D_ik in v[1].values())
            entropy_feature += D_i / len(data) * entropy_i

        gain = entropy - entropy_feature
        return gain

    # C4.5
    def compute_info_gain_ratio(self, data, feature):
        uni, counts = np.unique(data['label'], return_counts=True)
        entropy = - np.sum(counts[i]/len(data) * log2(counts[i]/len(data)) for i in range(len(uni)))

        # counts_dict: {feature_value: [feature_value_count, {label_value: label_value_count}]}
        counts_dict = {}
        for row in data.iterrows():
            row = row[1]
            # setdefault会有点难懂
            counts_dict.setdefault(row[feature], [0, {row['label']: 0}])[0] += 1
            counts_dict[row[feature]][1].setdefault(row['label'], 0)
            counts_dict[row[feature]][1][row['label']] += 1

        entropy_feature = 0
        entropy_A = 0
        for v in counts_dict.values():
            D_i = v[0]
            entropy_i = - np.sum(D_ik/D_i*log2(D_ik/D_i) for D_ik in v[1].values())
            entropy_feature += D_i / len(data) * entropy_i
            entropy_A -= D_i / len(data) * log2(D_i/len(data))

        gain_ratio = (entropy - entropy_feature) / entropy_A

        return gain_ratio

    def get_node_label(self, D_i):
        labels, counts = np.unique(D_i['label'], return_counts=True)
        max_count = 0
        max_label = None
        for i in range(len(labels)):
            if counts[i] > max_count:
                max_count = counts[i]
                max_label = labels[i]
        return max_label

    def create(self, data):
        self.root_node = self._tree(data)

    def _tree(self, data):
        max_gain_ratio = 0
        max_feature = None

        for feature in data.columns:
            if feature == 'label':
                continue
            gain_ratio = self.compute_info_gain_ratio(data, feature)
            if gain_ratio > max_gain_ratio:
                max_gain_ratio = gain_ratio
                max_feature = feature

        current_node = Node()

        if max_gain_ratio <= self.threshold:
            print('reach leaf node')
            current_node.label = self.get_node_label(data)
            return current_node

        current_node.feature = max_feature

        for k, D_i in data.groupby(max_feature):
            current_node.label = self.get_node_label(D_i)
            # 递归生成子节点
            current_node.child[k] = self._tree(D_i.drop(columns=[max_feature]))

        return current_node


data = pd.read_csv('DT_data.csv')
tree = DecisionTree()
tree.create(data)

print(tree.root_node)
