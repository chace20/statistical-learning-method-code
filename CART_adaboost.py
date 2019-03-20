import numpy as np
import pandas as pd
import sys


class Node:
    def __init__(self):
        self.feature = None
        self.feature_value = None
        self.label = None
        self.left_child = None
        self.right_child = None


class CART_CLS:
    """
    用于adaboost的分类树
    """
    def __init__(self, Dm):
        self.root_node = None
        self.threshold = 0
        self.Dm = Dm

    def build(self, data):
        self.root_node = self.cut(data)

    def predict(self, X):
        return self.traverse(self.root_node, X)

    def compute_gini(self, D):
        labels, counts = np.unique(D['label'], return_counts=True)
        gini = 1 - np.sum((counts[k]/len(D))**2 for k in range(len(labels)))
        return gini

    def compute_min_gini(self, D):
        min_feature = None
        min_feature_value = None
        min_gini = sys.float_info.max
        for feature in D.columns:
            if feature == 'label':
                continue
                # 这里巧妙得利用了groupby来划分不同特征对应的子集
            for k, D_1 in D.groupby(feature):
                D_2 = D.loc[D.index.difference(D_1.index)]
                gini = 0
                for D_i in [D_1, D_2]:
                    labels, counts = np.unique(D_i['label'], return_counts=True)
                    # NOTICE Dm_Di_sum表示Di对应的Dm中的权值和。加上这个是为了支持训练数据集样本到的权重分布。
                    Dm_Di_sum = np.sum(self.Dm[i] for i in D_i.index)
                    gini += len(D_i)/len(D)*(1-np.sum((counts[k]/len(D_i))**2 for k in range(len(labels)))) * Dm_Di_sum
                print('feature: %s, value: %s, gini: %s' % (feature, k, gini))
                if gini < min_gini:
                    min_feature = feature
                    min_gini = gini
                    min_feature_value = k
        return min_feature, min_feature_value, min_gini

    def get_node_label(self, D_i):
        labels, counts = np.unique(D_i['label'], return_counts=True)
        max_count = 0
        max_label = None
        for i in range(len(labels)):
            if counts[i] > max_count:
                max_count = counts[i]
                max_label = labels[i]
        return max_label

    def cut(self, D_i):
        current_node = Node()
        # min_gini是在特征A的情况下，集合D的基尼指数
        # gini直接是D的基尼指数
        gini = self.compute_gini(D_i)
        if gini <= self.threshold:
            current_node.label = self.get_node_label(D_i)
            print('reach leaf node, label: %s' % current_node.label)
            return current_node

        min_feature, min_feature_value, min_gini = self.compute_min_gini(D_i)
        print('min_feature: %s, min_feature_value: %s, min_gini: %s\n' % (min_feature, min_feature_value, min_gini))

        current_node.feature = min_feature
        current_node.feature_value = min_feature_value
        current_node.label = self.get_node_label(D_i)

        D_1 = D_i[D_i[min_feature] == min_feature_value]
        D_2 = D_i.loc[D_i.index.difference(D_1.index)]

        current_node.left_child = self.cut(D_1.drop(columns=[min_feature]))
        current_node.right_child = self.cut(D_2.drop(columns=[min_feature]))

        return current_node

    def traverse(self, current_node, X):
        # 不存在出度为1的节点，所以可以直接判断是否为叶子节点
        if current_node.left_child is None and current_node.right_child is None:
            return current_node.label
        if X[current_node.feature] == current_node.feature_value:
            return self.traverse(current_node.left_child, X)
        else:
            return self.traverse(current_node.right_child, X)

    def print(self, node, space):
        if node is None:
            return
        print('%s(feature=%s, feature_value=%s, label=%s)' % (space, node.feature, node.feature_value, node.label))
        self.print(node.left_child, space+'\t')
        self.print(node.right_child, space+'\t')

