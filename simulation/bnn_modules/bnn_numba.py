import numpy as np
from numba import njit, prange
from math import exp, lgamma, log, pow

@njit
def weight(n, s):
    weight_array = np.zeros(n)
    for k in prange(1, n-s+2):
        weight_array[k-1] = exp(lgamma(n-k+1) - lgamma(n-k-s+2)
                + lgamma(n-s+1) - lgamma(n+1) + log(s))
    return weight_array

@njit
def distance(X, x):
    n, p = X.shape
    dist = np.zeros(n)
    for i in prange(n):
        for j in prange(p):
            dist[i] += (X[i, j] - x[j]) ** 2
    return dist

@njit(parallel = True)
def weighted_sum(dist, w, y):
    y_new = y[np.argsort(dist)]
    sum = 0
    for i in prange(len(y_new)):
        sum += y_new[i]*w[i]
    return sum #np.vdot(y[np.argsort(dist)], w)

@njit(parallel = True)
def dpdz(p1, z, xz):
    n = p1.shape[0]
    # Contiguous copy first: numba's reshape() rejects non-contiguous views such as z[:, 0],
    # which the complementarity and variety DGPs pass. Values are unchanged.
    z = np.ascontiguousarray(z).astype(np.float64).reshape(-1, 1)
    z_squared = (z ** 2)
    
    Z_1 = np.zeros((n, 3), dtype=np.float64)
    Z_1[:, 0] = 1.0
    Z_1[:, 1] = z.flatten()
    Z_1[:, 2] = z_squared.flatten()
    
    coeff = np.linalg.lstsq(Z_1, p1)[0]
    return coeff[1] + 2*coeff[2]*xz[-1]

@njit(parallel = True)
def p_elas(y, XZ, xz, s1, p1, z, step = 0.1):
    n, p = XZ.shape
    s2 = 2*s1
    a, b = pow(s1, -2/p), pow(s2, -2/p)
    w = ((-b/(a - b))*weight(n, s1) + (a/(a - b))*weight(n, s2)).copy()
    y = y.copy()

    x_f, x_b = np.copy(xz), np.copy(xz)
    x_f[0], x_b[0] = x_f[0] + step/2, x_b[0] - step/2
    z_f, z_b = np.copy(xz), np.copy(xz)
    z_f[-1], z_b[-1] = z_f[-1] + step/2, z_b[-1] - step/2

    dist_xf, dist_xb = distance(XZ, x_f), distance(XZ, x_b)
    dist_zf, dist_zb = distance(XZ, z_f), distance(XZ, z_b) 
    
    pred_xf = weighted_sum(dist_xf, w, y)
    pred_xb = weighted_sum(dist_xb, w, y)
    pred_zf = weighted_sum(dist_zf, w, y)
    pred_zb = weighted_sum(dist_zb, w, y)

    pypp = (pred_xf - pred_xb)/step
    pypz = (pred_zf - pred_zb)/step

    dist = distance(XZ, xz)
    pred = weighted_sum(dist, w, y)

    p_elas = (pypp + pypz/dpdz(p1, z, xz))*xz[0]/pred
    return p_elas

@njit(parallel = True)
def p_elas_boot(y, XZ, xz, s1, p1, z, step = 0.1, bsize = 100):
    n, p = XZ.shape
    s2 = 2*s1
    a, b = pow(s1, -2/p), pow(s2, -2/p)
    w = ((-b/(a - b))*weight(n, s1) + (a/(a - b))*weight(n, s2)).copy()
    y = y.copy()

    x_f, x_b = np.copy(xz), np.copy(xz)
    x_f[0], x_b[0] = x_f[0] + step/2, x_b[0] - step/2
    z_f, z_b = np.copy(xz), np.copy(xz)
    z_f[-1], z_b[-1] = z_f[-1] + step/2, z_b[-1] - step/2

    b_results = np.zeros(bsize)
    for i in prange(bsize):
        idx = np.random.choice(np.arange(n), size = n, replace=True)
        y_b, XZ_b, p1_b, z_bb = y[idx], XZ[idx, :], p1[idx], z[idx]
        y_b = y_b.copy()

        dist_xf_b, dist_xb_b = distance(XZ_b, x_f), distance(XZ_b, x_b)
        dist_zf_b, dist_zb_b = distance(XZ_b, z_f), distance(XZ_b, z_b)

        pred_xf_b = weighted_sum(dist_xf_b, w, y_b)
        pred_xb_b = weighted_sum(dist_xb_b, w, y_b)
        pred_zf_b = weighted_sum(dist_zf_b, w, y_b)
        pred_zb_b = weighted_sum(dist_zb_b, w, y_b)

        pypp_b = (pred_xf_b - pred_xb_b)/step
        pypz_b = (pred_zf_b - pred_zb_b)/step

        dist_b = distance(XZ_b, xz)
        pred_b = weighted_sum(dist_b, w, y_b)
        
        b_results[i] = (pypp_b + pypz_b/dpdz(p1_b, z_bb, xz))*xz[0]/pred_b
    return np.mean(b_results), np.var(b_results)

@njit(parallel = True)
def bnn_2scale(y, XZ, xz, s1):
    n, p = XZ.shape
    s2 = 2*s1
    a, b = pow(s1, -2/p), pow(s2, -2/p)
    w = ((-b/(a - b))*weight(n, s1) + (a/(a - b))*weight(n, s2)).copy()
    y = y.copy()

    dist = distance(XZ, xz)
    pred = weighted_sum(dist, w, y)
    return pred