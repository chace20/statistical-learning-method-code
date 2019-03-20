import numpy as np

X = np.array([[3, 3], [4, 3], [1, 1]])
Y = np.array([1, 1, -1])

theta = 1


# 原始形式
def fit(X, Y):
    w = np.zeros([2, 1])
    b = 0

    while True:
        true_point = 0
        for i in range(len(X)):
            if (Y[i] * (np.dot(w.T, X[i]) + b))[0] <= 0:
                w = w + theta * Y[i] * X[i].reshape(-1, 1)
                b = b + theta * Y[i]
                print('w: %s' % w)
                print('b: %s\n' % b)
            else:
                true_point += 1
        if true_point == len(X):
            break
    return w, b


# 对偶形式
def fit2(X, Y):
    alpha = np.zeros([len(X), 1])
    b = 0

    Gram = np.empty([len(X), len(X)])

    for i in range(len(X)):
        for j in range(len(X)):
            Gram[i, j] = np.dot(X[i], X[j])

    while True:
        true_point = 0
        for i in range(len(X)):
            # 这里的[0]是为了将array转换成数字
            if (Y[i] * (np.sum(alpha[j] * Y[j] * Gram[i, j] for j in range(len(X))) + b))[0] <= 0:
                alpha[i] += theta
                b += Y[i]
                print('alpha: %s' % alpha)
                print('b: %s\n' % b)
            else:
                true_point += 1
        if true_point == len(X):
            break

    w = np.sum(alpha[i] * Y[i] * X[i] for i in range(len(X)))
    return w, b


w, b = fit2(X, Y)
print('\n----result----')
print(w)
print(b)
