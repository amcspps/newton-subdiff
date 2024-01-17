import numpy as np
from numpy.linalg import inv, norm
from intvalpy import Interval

def pos(x):
    return x if x > 0 else 0

def neg(x):
    return -x if x < 0 else 0

def sti(x):
    n = len(x)
    result = np.zeros(2 * n)
    for i in range(n):
        result[i] = -x[i].inf
        result[i + n] = x[i].sup
    return result

def sti_inv(x):
    n = len(x) // 2
    result = [Interval(-x[i].mid, x[i + n].mid) for i in range(n)]
    return result

def partx_pos(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0.5  # actually something ∈ [0, 1]
    else:
        return 0

def partx_neg(x):
    if x < 0:
        return -1
    elif x == 0:
        return -0.5  # actually something ∈ [-1, 0]
    else:
        return 0

def partmax_1(C, i, j, x):
    n = len(x) // 2
    prod_1 = pos(C[i, j].sup) * pos(x[j])
    prod_2 = neg(C[i, j].inf) * pos(x[j + n])
    if prod_1 > prod_2:
        return (pos(C[i, j].sup), 0)
    elif prod_1 == prod_2:
        return (0.5 * pos(C[i, j].sup), 0.5 * neg(C[i, j].inf))
    else:
        return (0, neg(C[i, j].inf))

def partmax_2(C, i, j, x):
    n = len(x) // 2
    prod_1 = pos(C[i, j].sup) * pos(x[j + n])
    prod_2 = neg(C[i, j].inf) * pos(x[j])
    if prod_1 > prod_2:
        return (0, pos(C[i, j].sup))
    elif prod_1 == prod_2:
        return (0.5 * neg(C[i, j].inf), 0.5 * pos(C[i, j].sup))
    else:
        return (neg(C[i, j].inf), 0)

def partF(C, i, x):
    n = len(x) // 2
    res = np.zeros(2 * n)
    if 1 <= i <= n:
        for j in range(n):
            temp = partmax_1(C, i, j, x)
            res_1 = pos(C[i, j].inf) * partx_neg(x[j]) + neg(C[i, j].sup) * partx_neg(x[j + n]) - temp[0]
            res_2 = pos(C[i, j].inf) * partx_neg(x[j]) + neg(C[i, j].sup) * partx_neg(x[j + n]) - temp[1]
            res[j] -= res_1
            res[j + n] -= res_2
    else:
        i -= n
        for j in range(n):
            temp = partmax_2(C, i, j, x)
            res_1 = temp[0] - pos(C[i, j].inf) * partx_neg(x[j + n]) - neg(C[i, j].sup) * partx_neg(x[j])
            res_2 = temp[1] - pos(C[i, j].inf) * partx_neg(x[j + n]) - neg(C[i, j].sup) * partx_neg(x[j])
            res[j] += res_1
            res[j + n] += res_2
    return res

def D(C, x):
    n = len(x)
    D_matrix = np.zeros((n, n))
    for i in range(n):
        D_matrix[i, :] = partF(C, i + 1, x)
    return D_matrix

def init_point(C, d):
    midC = np.array([[interval.mid() for interval in row] for row in C])
    C̃ = np.array([[pos(midC[i, j]) if i == j else neg(midC[i, j]) for j in range(len(midC[i]))] for i in range(len(midC))])
    return np.linalg.solve(C̃, sti(d))

def sub_diff(C, d, x0, eps):
    def 𝒢(x):
        x_inv = sti_inv(x)
        C_x = np.dot(C, sti(x_inv))
        return sti(C_x) - sti(d)

    x = x0
    𝒢_val = 𝒢(x)
    count = 0

    while norm(𝒢_val) >= eps:
        print("x", x)
        try:
            x -= inv(D(C, x)) @ 𝒢_val
        except:
            print("Subgradient D is singular")
            break
        𝒢_val = 𝒢(x)
        count += 1

    return (sti_inv(x), count)

C = np.array([[Interval(2, 4), Interval(-2, 1)],
               [Interval(-1, 2), Interval(2, 4)]])
d = np.array([Interval(2, 2), Interval(2, 2)])
x0 = np.array([Interval(0, 1), Interval(0, 1)])
eps = 1e-6
result = sub_diff(C, d, x0, eps)
print(result)
