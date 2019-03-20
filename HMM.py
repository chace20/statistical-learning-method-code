import numpy as np


def forward(A, B, Pi, O):
    T = len(O)
    N = len(A)
    # (a)
    last_alpha = [Pi[i] * B[i, O[0]] for i in range(N)]
    print('alpha_%d = %s' % (1, last_alpha))
    alpha = [0] * N
    # (2) [0,...,T-2] len=T-1
    for t in range(0, T-1):
        for i in range(N):
            alpha[i] = np.dot(last_alpha, A[:, i]) * B[i, O[t+1]]
        last_alpha = alpha.copy()
        print('alpha_%d = %s' % (t+2, alpha))
    # (3)
    prob = np.sum(alpha)
    print('prob(O|lambda) = %s\n' % prob)
    return prob


def backward(A, B, Pi, O):
    T = len(O)
    N = len(A)
    # (1)
    last_beta = [1] * N
    print('beta_%d = %s' % (T, last_beta))
    beta = [0] * N
    # (2) [T-2,...,0] len=T-1
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[i] = np.sum(A[i, j] * B[j, O[t+1]] * last_beta[j] for j in range(N))
        last_beta = beta.copy()
        print('beta_%d = %s' % (t+1, beta))
    # (3)
    prob = np.sum(Pi[i] * B[i, O[0]] * beta[i] for i in range(N))
    print('prob(O|lambda) = %s\n' % prob)
    return prob


def viterbi(A, B, Pi, O):
    T = len(O)
    N = len(A)
    delta = np.empty([T, N])
    psi = np.empty([T, N]).astype(np.int32)
    I_star = np.empty(T).astype(np.int32)

    # (1)
    delta[0] = [Pi[i] * B[i, O[0]] for i in range(N)]
    psi[0] = np.zeros(N)
    # (2) len=T-1
    for t in range(1, T):
        for i in range(N):
            delta[t, i] = max(delta[t-1, j] * A[j, i] for j in range(N)) * B[i, O[t]]
            psi[t, i] = np.argmax(list(delta[t-1, j] * A[j, i] for j in range(N)))
    print('delta[TxN] = %s' % delta)
    print('psi[TxN] = %s' % psi)
    # (3)
    P_star = max(delta[T-1])
    I_star[T-1] = np.argmax(delta[T-1])
    print('P* = %s' % P_star)
    print('I*_%d = %d' % (T, I_star[T-1] + 1))
    # (4) [T-2,...,0] len=T-1
    for t in range(T-2, -1, -1):
        I_star[t] = psi[t+1, I_star[t+1]]
    print('I* = %s\n' % list(i+1 for i in I_star))
    return I_star


A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
Pi = [0.2, 0.4, 0.4]
O = [0, 1, 0, 1]

backward(A, B, Pi, O)
forward(A, B, Pi, O)

viterbi(A, B, Pi, O)
