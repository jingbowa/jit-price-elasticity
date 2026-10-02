import torch

def weight_torch(n, s, dtype=torch.float64, device=None):
    k_values = torch.arange(0, n-s+1, dtype=dtype, device=device)
    n_tensor = torch.tensor(n, dtype=dtype, device=device)
    s_tensor = torch.tensor(s, dtype=dtype, device=device)

    weight_values = torch.exp(
        torch.lgamma(n_tensor - k_values)
        - torch.lgamma(n_tensor - k_values - s_tensor + 1)
        + torch.lgamma(n_tensor - s_tensor + 1)
        - torch.lgamma(n_tensor + 1)
        + torch.log(s_tensor)
    )

    weight_array = torch.cat((weight_values, torch.zeros(n - len(weight_values), dtype=dtype, device=device)))
    return weight_array

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

def dpdz_torch(XZ, xz, p1_col_idx, z_col_idx, dtype=torch.float64):
    device = XZ.device
    # Use direct indexing + unsqueeze instead of slice (idx:idx+1 breaks for idx=-1).
    z = XZ[:, z_col_idx].unsqueeze(1)
    z_squared = z ** 2
    z_poly = torch.cat((z, z_squared), dim=1)
    ones = torch.ones(XZ.shape[0], 1, dtype=dtype, device=device)
    Z = torch.cat((ones, z_poly), dim=1)
    p1 = XZ[:, p1_col_idx].unsqueeze(1)
    # Normal equations instead of torch.linalg.lstsq — the latter's .solution
    # field is driver-dependent and unreliable across platforms (Windows gelsy vs Linux gelsd).
    coeff = torch.linalg.solve(Z.T @ Z, Z.T @ p1)
    return coeff[1] + 2*coeff[2]*xz[z_col_idx]

def compute_derivatives(XZ, xz, y, w, p1_col_idx, z_col_idx, step=0.1, dtype=torch.float64):
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

def bnn_torch(y, XZ, xz, s, dtype=torch.float64):
    device = y.device
    n, p = XZ.shape
    w = weight_torch(n, p, s, dtype=dtype, device=device)

    dist = distance_torch(XZ, xz)
    pred = weighted_sum_torch(dist, y, w)
    return pred

def bnn_2scale_torch(y, XZ, xz, s, dtype=torch.float64):
    device = y.device
    n, p = XZ.shape
    w = weight_2scale_torch(n, p, s, dtype=dtype, device=device)

    dist = distance_torch(XZ, xz)
    pred = weighted_sum_torch(dist, y, w)
    return pred

def p_elas_torch(y, XZ, xz, p1_col_idx, z_col_idx, s, dtype=torch.float64):
    n = XZ.shape[0]
    w = weight_torch(n, s, dtype=dtype)

    dist = distance_torch(XZ, xz)
    pred = weighted_sum_torch(dist, y, w)

    pypp, pypz = compute_derivatives(XZ, xz, y, w, p1_col_idx, z_col_idx, dtype=dtype)
    dpdz = dpdz_torch(XZ, xz, p1_col_idx, z_col_idx, dtype=dtype)

    p_elas = (pypp + pypz / dpdz) * xz[p1_col_idx] / pred
    return p_elas

def p_elas_2scale_torch(y, XZ, xz, p1_col_idx, z_col_idx, s, dtype=torch.float64):
    n, p = XZ.shape
    w = weight_2scale_torch(n, p, s, dtype=dtype)

    dist = distance_torch(XZ, xz)
    pred = weighted_sum_torch(dist, y, w)

    pypp, pypz = compute_derivatives(XZ, xz, y, w, p1_col_idx, z_col_idx, dtype=dtype)
    dpdz = dpdz_torch(XZ, xz, p1_col_idx, z_col_idx, dtype=dtype)

    p_elas = (pypp + pypz / dpdz) * xz[p1_col_idx] / pred
    return p_elas

def p_elas_torch_boot(y, XZ, xz, p1_col_idx, z_col_idx, s, boot_size=100, dtype = torch.float64):
    device = y.device
    n = y.shape[0]
    p_elas_bootstraps = torch.zeros(boot_size, dtype=dtype, device=device)

    for b in range(boot_size):
        # Generate bootstrap sample by sampling with replacement
        bootstrap_indices = torch.randint(n, (n,))
        y_bootstrap = y[bootstrap_indices]
        XZ_bootstrap = XZ[bootstrap_indices]

        # Compute the p_elas for the bootstrap sample
        p_elas_bootstrap = p_elas_torch(y_bootstrap, XZ_bootstrap, xz, p1_col_idx, z_col_idx, s, dtype)
        p_elas_bootstraps[b] = p_elas_bootstrap

    return torch.mean(p_elas_bootstraps), torch.var(p_elas_bootstraps)

def p_elas_2scale_torch_boot(y, XZ, xz, p1_col_idx, z_col_idx, s, boot_size=100,  dtype = torch.float64):
    device = y.device
    n = y.shape[0]
    p_elas_bootstraps = torch.zeros(boot_size, dtype=dtype, device=device)

    for b in range(boot_size):
        # Generate bootstrap sample by sampling with replacement
        bootstrap_indices = torch.randint(n, (n,))
        y_bootstrap = y[bootstrap_indices]
        XZ_bootstrap = XZ[bootstrap_indices]

        # Compute the p_elas for the bootstrap sample
        p_elas_bootstrap = p_elas_2scale_torch(y_bootstrap, XZ_bootstrap, xz, p1_col_idx, z_col_idx, s, dtype)
        p_elas_bootstraps[b] = p_elas_bootstrap

    return torch.mean(p_elas_bootstraps), torch.var(p_elas_bootstraps)