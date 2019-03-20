import pandas as pd
import numpy as np
from math import exp, log

# FIXME 有些问题


class IIS:
    def __init__(self):
        self.data = None
        self.X = None
        self.Y = set()
        self.w = []
        self.xy_num = {}
        self.x_num = {}
        self.xy2id = {}
        self.id2xy = {}
        self.n = 0
        self.N = 0
        self.M = 0
        self.EPS = 0.005

    def init_para(self, train_data):
        self.N = len(train_data)
        self.data = train_data
        self.X = train_data[['weather', 'desc', 'temp', 'feature']]
        self.Y = set(train_data['label'])

        for item in train_data.iterrows():
            row = item[1]
            X = row[0:4]
            y = row[-1]
            for x in X:
                self.xy_num.setdefault((x, y), 0)
                self.x_num.setdefault(x, 0)
                self.xy_num[(x, y)] += 1
                self.x_num[x] += 1

        self.n = len(self.xy_num)
        # 认为M就是不同(x,y)对的个数
        self.M = self.n
        self.w = [0] * self.n
        for i, xy in enumerate(self.xy_num):
            # 原来这里的xy2id映射是为了Pxy[(x,y)]计数方便，其实本质上对应的是i才对
            self.xy2id[xy] = i
            self.id2xy[i] = xy

    """
    特征函数
    """
    def feature(self, x, y):
        return 1 if (x, y) in self.xy_num else 0

    def Zx(self, X):    # 计算每个Z(x)值
        zx = 0
        for y in self.Y:
            ss = 0
            for x in X:
                if (x, y) in self.xy_num:
                    ss += self.w[self.xy2id[(x, y)]]
            zx += exp(ss)
        return zx

    def Pyx(self, y, Xi):
        # 书中的x其实是一个向量，所以这里用X代表
        # 求分母zx
        zx = self.Zx(Xi)
        # 求分子
        s = 0
        for x in Xi:
            if (x, y) in self.xy_num:
                s += self.w[self.xy2id[(x, y)]]
        pyx = exp(s)/zx
        return pyx

    def E_P_emp(self, i):
        # 姑且认为(x,y)的在输入集中的频度/N就是期望吧
        # 因为同时出现即f(x,y)=1，所以原来的公式可以转换为E~(f)=P~(x,y)
        return self.xy_num[self.id2xy[i]] / self.N

    def E_P_exp(self, i):
        x, y = self.id2xy[i]
        E = 0
        # 遍历X
        for row in self.X.iterrows():
            Xi = list(row[1])
            if x not in Xi:
                continue
            pyx = self.Pyx(y, Xi)
            # 认为x服从均匀分布，所以用P~(x)=1/N
            E += (1 / self.N) * pyx
        return E

    def convergence(self, last_w, w):
        for last, now in zip(last_w, w):
            if abs(last - now) >= self.EPS:
                return False
        return True

    def train(self, maxiter=1000):
        for loop in range(maxiter):
            print('loop: %s' % loop)
            last_w = self.w.copy()
            for i in range(self.n):
                ep_exp = self.E_P_exp(i)
                ep_emp = self.E_P_emp(i)
                self.w[i] += 1 / self.M * log(ep_emp/ep_exp)
            print('w: %s\n' % self.w)
            if self.convergence(last_w, self.w):
                break

    def predict(self, Xi):
        zx = self.Zx(Xi)
        result = {}
        for y in self.Y:
            ss = 0
            for x in Xi:
                if (x, y) in self.xy_num:
                    ss += self.w[self.xy2id[(x, y)]]
            pyx = exp(ss)/zx
            result[y] = pyx
        return result


"""
很重要的一点就是：训练集中的一个(x,y)就对应一个特征函数fi(x,y)，所以才会有i与(x,y)的映射
"""
df = pd.read_csv('max_entropy_data.csv')
iis = IIS()
iis.init_para(df)
iis.train(maxiter=1000)
result = iis.predict(['sunny', 'hot', 'high', 'FALSE'])
print(result)
