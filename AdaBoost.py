import pandas as pd
import numpy as np
from sklearn import metrics
from math import log
from math import exp

from CART_adaboost import CART_CLS


class AdaBoost:
    def __init__(self):
        # key: weight value: tree
        self.f_x = {}

    def fit(self, data):
        X = data[['body', 'business', 'potential']]
        Y = data['label']
        N = len(data)
        # 1. 初始化权值分布
        Dm = [1/N] * N
        # 2. 迭代更新权值
        while True:
            # (a) 得到基本分类器Gm(x)
            # TODO 训练数据集的权值感觉不能这样弄，这个权值必须反映在损失函数中
            # Dm_data = pd.DataFrame(columns=X.columns)
            # for i in range(N):
            #     Dm_data.loc[i] = X.iloc[i].apply(lambda x: x * Dm[i])
            Dm_data = data
            tree = CART_CLS(Dm)
            tree.build(Dm_data)
            # (b) 计算分类误差率
            em = np.sum(Dm[i] for i in range(N) if tree.predict(Dm_data.iloc[i]) != Y[i])
            # NOTICE 这里使用了一个CART二叉分类树，导致直接em=0，CART本身就是一个强分类器。
            # NOTICE 可能使用x>v这种决策树桩才能充分利用到AdaBoost的特性。那样每一个弱分类器都只是一个树桩决策树。
            if em == 0:
                self.f_x[1] = tree
                break
            # (c) 计算Gm(x)的系数alpha_m
            alpha_m = 0.5 * log((1-em)/em)
            # (d) 更新Dm
            Zm = np.sum(Dm[i] * exp(-alpha_m * Y[i] * tree.predict(Dm_data.iloc[i])) for i in range(N))
            Dm = list(Dm[i] * exp(-alpha_m * Y[i] * tree.predict(Dm_data.iloc[i])) / Zm for i in range(N))

            self.f_x[alpha_m] = tree
            # 判断是否达到终止条件
            if self.cal_f_x_precision(X, Y) > 0.9:
                break

    def cal_f_x_precision(self, X, Y):
        Y_pred = []
        for row in X.iterrows():
            Xi = row[1]
            Y_pred_i = np.sum(alpha_m * tree.predict(Xi) for alpha_m, tree in self.f_x.items())
            Y_pred.append(Y_pred_i)
        return metrics.precision_score(Y, Y_pred)


data = pd.read_csv('adaboost_data.csv')
adaboost = AdaBoost()
adaboost.fit(data)
