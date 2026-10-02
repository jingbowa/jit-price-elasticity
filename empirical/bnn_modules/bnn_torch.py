"""Debiased two-scale bagged nearest neighbors (BNN) elasticity estimator.

PyTorch implementation used for the NYC ride-sharing application. The
estimator combines closed-form BNN weights at subsampling sizes (s, 2s)
to remove the leading-order bias term, and implements the control-function
identification result (equation (2) of the paper):

    d f / d p  =  d h / d p  +  (d g / d z)^{-1} * (d h / d z)
"""

import torch

def weight_2scale_torch(n, p, s, dtype=torch.float64, device=None):
    s1, s2 = s, 2 * s
    a, b = s1 ** (-2 / p), s2 ** (-2 / p)

    n_tensor = torch.tensor(n, dtype=dtype, device=device)
    s1_tensor = torch.tensor(s1, dtype=dtype, device=device)
    s2_tensor = torch.tensor(s2, dtype=dtype, device=device)

    k_values_1 = torch.arange(0, n - s1 + 1, dtype=dtype, device=device)
    k_values_2 = torch.arange(0, n - s2 + 1, dtype=dtype, device=device)

    weight_1 = torch.exp(torch.lgamma(n_tensor - k_values_1) - torch.lgamma(n_tensor - k_values_1 - s1_tensor + 1) + torch.lgamma(n_tensor - s1_tensor + 1) - torch.lgamma(n_tensor + 1) + torch.log(s1_tensor))
    weight_2 = torch.exp(torch.lgamma(n_tensor - k_values_2) - torch.lgamma(n_tensor - k_values_2 - s2_tensor + 1) + torch.lgamma(n_tensor - s2_tensor + 1) - torch.lgamma(n_tensor + 1) + torch.log(s2_tensor))

    weight_1 = torch.cat((weight_1, torch.zeros(n - len(weight_1), dtype=dtype, device=device)))
    weight_2 = torch.cat((weight_2, torch.zeros(n - len(weight_2), dtype=dtype, device=device)))

    result = ((-b / (a - b)) * weight_1 + (a / (a - b)) * weight_2)
    return result

def distance_torch(X, x):
    return torch.cdist(X, x.unsqueeze(0), p = 2).squeeze()

def weighted_sum_torch(dist, y, w):
    device = y.device
    sorted_indices = torch.argsort(dist)
    y_new = y[sorted_indices]
    w = w.to(device)
    return torch.sum(y_new * w)

def dpdz_torch(XZ, xz, p1_col_idx, z_col_idx, k=4, dtype=torch.float64):
    device = XZ.device
    # Get price column and reshape
    P = XZ[:, p1_col_idx].unsqueeze(1).to(dtype)

    # Create polynomial terms up to degree k
    z = XZ[:, z_col_idx].to(dtype)
    z_poly = torch.stack([z ** i for i in range(1, k + 1)], dim=1)
    Z = torch.cat((torch.ones(XZ.shape[0], 1, dtype=dtype, device=device), z_poly), dim=1)

    coeff = torch.linalg.lstsq(Z, P).solution
    return torch.sum(torch.stack([i * coeff[i] * (xz[z_col_idx] ** (i - 1)) for i in range(1, k + 1)]), dim=0)

def compute_derivatives(XZ, xz, y, w, p1_col_idx, z_col_idx, step=0.02, dtype=torch.float64):
    x_f, x_b = xz.clone().to(dtype), xz.clone().to(dtype)
    x_f[p1_col_idx], x_b[p1_col_idx] = x_f[p1_col_idx] + step / 2, x_b[p1_col_idx] - step / 2
    z_f, z_b = xz.clone().to(dtype), xz.clone().to(dtype)
    z_f[z_col_idx], z_b[z_col_idx] = z_f[z_col_idx] + step / 2, z_b[z_col_idx] - step / 2

    dist_xf, dist_xb = distance_torch(XZ, x_f), distance_torch(XZ, x_b)
    dist_zf, dist_zb = distance_torch(XZ, z_f), distance_torch(XZ, z_b)

    pred_xf = weighted_sum_torch(dist_xf, y, w)
    pred_xb = weighted_sum_torch(dist_xb, y, w)
    pred_zf = weighted_sum_torch(dist_zf, y, w)
    pred_zb = weighted_sum_torch(dist_zb, y, w)

    pypp = (pred_xf - pred_xb) / step
    pypz = (pred_zf - pred_zb) / step

    return pypp, pypz

def p_elas_2scale_torch(y, XZ, xz, p1_col_idx, z_col_idx, s, dtype=torch.float64):
    # Convert inputs to the specified dtype
    XZ = XZ.to(dtype)
    xz = xz.to(dtype)
    y = y.to(dtype)

    n, p = XZ.shape
    w = weight_2scale_torch(n, p, s, dtype=dtype)

    dist = distance_torch(XZ, xz)
    pred = weighted_sum_torch(dist, y, w)

    pypp, pypz = compute_derivatives(XZ, xz, y, w, p1_col_idx, z_col_idx, dtype=dtype)
    dpdz = dpdz_torch(XZ, xz, p1_col_idx, z_col_idx, dtype=dtype)

    p_elas = (pypp + pypz / dpdz) * xz[p1_col_idx] / pred
    return p_elas

def p_elas_2scale_torch_boot(y, XZ, xz, p1_col_idx, z_col_idx, s, boot_size=100,  dtype = torch.float64):
    device = y.device
    n = y.shape[0]
    p_elas_bootstraps = torch.zeros(boot_size, dtype=dtype, device=device)

    for b in range(boot_size):
        # Generate bootstrap sample by sampling with replacement
        bootstrap_indices = torch.randint(0, n, (n,), device=device)
        y_bootstrap = y[bootstrap_indices]
        XZ_bootstrap = XZ[bootstrap_indices]

        # Compute the p_elas for the bootstrap sample
        p_elas_bootstrap = p_elas_2scale_torch(y_bootstrap, XZ_bootstrap, xz, p1_col_idx, z_col_idx, s, dtype)
        p_elas_bootstraps[b] = p_elas_bootstrap

    return torch.mean(p_elas_bootstraps), torch.var(p_elas_bootstraps)
