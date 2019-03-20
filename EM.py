import numpy as np
from math import sqrt, exp, pi


def cal_gaussian(y, mu, sigma):
    gaussian = 1/(sqrt(2 * pi) * sigma) * exp(-((y-mu)**2)/(2*(sigma**2)))
    if gaussian == 0.0:
        return 0.001
    return gaussian


Y = [-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75]
K = 2
N = len(Y)
threshold = 0.01

# 1. 初始化参数
np.random.seed(7)

mu = np.random.random(K)
mu2 = np.random.random(K)

sigma = np.random.random(K)
sigma2 = np.random.random(K)

alpha = np.array([0.5, 0.5])
alpha2 = np.array([0.5, 0.5])

while True:
    # 2. E步
    gamma = np.zeros([N, K])
    for k in range(K):
        for j in range(N):
            sum_j = np.sum(alpha[kk] * cal_gaussian(Y[j], mu[kk], sigma[kk]) for kk in range(K))
            gamma[j][k] = alpha[k] * cal_gaussian(Y[j], mu[k], sigma[k]) / sum_j

    # 3. M步
    for k in range(K):
        mu2[k] = np.dot(gamma[:, k], Y) / np.sum(gamma[:, k])
        sigma2[k] = sqrt(np.dot(gamma[:, k], list(map(lambda yj: (yj-mu2[k])**2, Y))) / np.sum(gamma[:, k]))
        alpha2[k] = np.sum(gamma[:, k]) / N

    # 4. 判断收敛，分别计算两次迭代结果的差的平方和
    di = sqrt(np.sum((mu2 - mu) ** 2)) + sqrt(np.sum((sigma2 - sigma) ** 2)) + sqrt(np.sum((alpha2 - alpha) ** 2))
    if di < threshold:
        break

    mu = mu2.copy()
    sigma = sigma2.copy()
    alpha = alpha2.copy()

print('mu: %s' % mu2)
print('sigma: %s' % sigma2)
print('alpha: %s' % alpha2)
