import numpy as np
from numba import njit, prange
from math import lgamma, exp, log, pow

@njit(parallel = True)
def weight(n, s):
    weight_array = np.zeros(n)
    for k in prange(1, n-s+2):
        weight_array[k-1] = exp(lgamma(n-k+1) - lgamma(n-k-s+2)
                + lgamma(n-s+1) - lgamma(n+1) + log(s))
    return weight_array

@njit(parallel = True)
def weight_2scale(n, p, s):
    s1, s2 = s, 2*s
    a, b = pow(s1, -2/p), pow(s2, -2/p)
    weight_1, weight_2  = np.zeros(n), np.zeros(n)
    
    for k in prange(1, n-s1+2):
        weight_1[k-1] = exp(lgamma(n-k+1) - lgamma(n-k-s1+2)
                + lgamma(n-s1+1) - lgamma(n+1) + log(s1))
    
    for k in prange(1, n-s2+2):
        weight_2[k-1] = exp(lgamma(n-k+1) - lgamma(n-k-s2+2)
                + lgamma(n-s2+1) - lgamma(n+1) + log(s2))
    return ((-b/(a - b))*weight_1 + (a/(a - b))*weight_2)

@njit(parallel = True)
def distance(X, x):
    n, p = X.shape
    dist = np.zeros(n)
    for i in prange(n):
        for j in prange(p):
            dist[i] += (X[i, j] - x[j]) ** 2
    return dist

@njit(parallel = True)
def weighted_sum(dist, y, w):
    y_new = y[np.argsort(dist)]
    sum = 0
    for i in prange(len(y_new)):
        sum += y_new[i]*w[i]
    return sum #np.vdot(y[np.argsort(dist)], w)

@njit(parallel = True)
def dpdz(p1, z, xz):
    n = p1.shape[0]
    Z = np.column_stack((z, z ** 2))
    Z_1 = np.hstack((np.ones((n, 1)).astype(np.float32), Z))
    coeff = np.linalg.lstsq(Z_1, p1)[0]
    return coeff[1] + 2*coeff[2]*xz[-1]

@njit
def bnn(y, X, x, s):
    n = X.shape[0]
    w = weight(n, s)
    dist = distance(X, x)
    pred = weighted_sum(dist, y, w)
    return pred

@njit(parallel = True)
def bnn_boot(y, X, x, s, boot = 100):
    n, p = X.shape
    w = weight_2scale(n, p, s)

    index = np.arange(n)
    b_results = np.zeros(boot)

    dist = distance(X, x)
    for j in prange(boot):
        idx = np.random.choice(index, size = n, replace=True)
        y_b, dist_b = y[idx], dist[idx]
        b_results[j] = weighted_sum(dist_b, y_b, w)
    return np.var(b_results)

@njit
def bnn_2(y, X, x, s):
    n, p = X.shape
    w = weight_2scale(n, p, s)
    y = y.copy()
    dist = distance(X, x)
    pred = weighted_sum(dist, y, w)
    return pred

@njit(parallel = True)
def bnn_vec(y, X, X_target, s):
    n, p = X.shape
    s1, s2 = s, 2*s
    a, b = pow(s1, -2/p), pow(s2, -2/p)
    w = (-b/(a - b))*weight(n, s1) + (a/(a - b))*weight(n, s2)
    y = y.copy()
    n_row = X_target.shape[0]
    pred = np.zeros(n_row)
    for i in prange(n_row):
        x = X_target[i, :]
        dist = distance(X, x)
        pred[i] = weighted_sum(dist, y, w)
    return pred

@njit(parallel = True)
def bnn_vec_boot(y, X, X_target, s, boot = 100):
    n, p = X.shape
    w = weight_2scale(n, p, s)

    index = np.arange(n)
    n_row = X_target.shape[0]
    b_results = np.zeros((n_row, boot))

    for i in prange(n_row):
        x = X_target[i, :]
        dist = distance(X, x)
        for j in prange(boot):
            idx = np.random.choice(index, size = n, replace=True)
            y_b, dist_b = y[idx], dist[idx]
            b_results[i, j] = weighted_sum(dist_b, y_b, w)
    return b_results

@njit
def p_elas(y, XZ, xz, p1, z, s, step = 0.2):
    n, p = XZ.shape
    w = weight(n, s)

    x_f, x_b = np.copy(xz), np.copy(xz)
    x_f[0], x_b[0] = x_f[0] + step/2, x_b[0] - step/2
    z_f, z_b = np.copy(xz), np.copy(xz)
    z_f[-1], z_b[-1] = z_f[-1] + step/2, z_b[-1] - step/2

    dist = distance(XZ, xz)
    dist_xf, dist_xb = distance(XZ, x_f), distance(XZ, x_b)
    dist_zf, dist_zb = distance(XZ, z_f), distance(XZ, z_b) 

    pred = weighted_sum(dist, y, w)
    pred_xf = weighted_sum(dist_xf, y, w)
    pred_xb = weighted_sum(dist_xb, y, w)
    pred_zf = weighted_sum(dist_zf, y, w)
    pred_zb = weighted_sum(dist_zb, y, w)

    pypp = (pred_xf - pred_xb)/step
    pypz = (pred_zf - pred_zb)/step

    p_elas = (pypp + pypz/dpdz(p1, z, xz))*xz[0]/pred
    return p_elas

@njit(parallel = True)
def p_elas_boot(y, XZ, xz, p1, z, s, step = 0.2, bsize = 100):
    n, p = XZ.shape
    w = weight(n, s)
    x_f, x_b = np.copy(xz), np.copy(xz)
    x_f[0], x_b[0] = x_f[0] + step/2, x_b[0] - step/2
    z_f, z_b = np.copy(xz), np.copy(xz)
    z_f[-1], z_b[-1] = z_f[-1] + step/2, z_b[-1] - step/2

    b_results = np.zeros(bsize)
    for i in prange(bsize):
        idx = np.random.choice(np.arange(n), size = n, replace=True)
        y_b, XZ_b = y[idx], XZ[idx, :]
        y_b = y_b.copy()

        dist_xf_b, dist_xb_b = distance(XZ_b, x_f), distance(XZ_b, x_b)
        dist_zf_b, dist_zb_b = distance(XZ_b, z_f), distance(XZ_b, z_b)

        pred_xf_b = weighted_sum(dist_xf_b, y_b, w)
        pred_xb_b = weighted_sum(dist_xb_b, y_b, w)
        pred_zf_b = weighted_sum(dist_zf_b, y_b, w)
        pred_zb_b = weighted_sum(dist_zb_b, y_b, w)

        pypp_b = (pred_xf_b - pred_xb_b)/step
        pypz_b = (pred_zf_b - pred_zb_b)/step

        dist_b = distance(XZ_b, xz)
        pred_b = weighted_sum(dist_b, y_b, w)
        
        b_results[i] = (pypp_b + pypz_b/dpdz(p1[idx], z[idx], xz))*xz[0]/pred_b
    return np.sqrt(np.var(b_results))