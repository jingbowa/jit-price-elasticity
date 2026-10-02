import torch

def Logit_data(num_obs=20000, num_products=3):
    m_price = torch.full((num_products,), 0.5)
    m, j = num_obs, num_products
    u = torch.rand(m, j, dtype=torch.float32) - 0.5
    z = torch.rand(m, j, dtype=torch.float32) - 0.5
    p_jt = m_price + 0.5 * (z + u + torch.rand(m, j, dtype=torch.float32) - 0.5)
    delta_i, alpha_i = 4, 2
    u_jt = delta_i - alpha_i * p_jt + u
    s_jt = torch.softmax(u_jt, dim=1)
    return s_jt[:, 0], s_jt[:, 1], torch.cat((p_jt, z[:, [0]]), dim=1), p_jt[:, 0], z[:, 0]

def Logit_stats(num_obs=20000, num_products=3):
    m_price = torch.full((num_products,), 0.5)
    m, j = num_obs, num_products
    p_jt = m_price.expand(m, j)
    u = torch.rand(m, j, dtype=torch.float32) - 0.5
    delta_i, alpha_i = 4, 2
    u_jt = delta_i - alpha_i * p_jt + u
    s_jt = torch.softmax(u_jt, dim=1)
    e_own = torch.mean(-alpha_i * p_jt * (1 - s_jt), dim=0)
    e_cross = torch.mean(alpha_i * p_jt * s_jt, dim=0)
    return e_own[0].item(), e_cross[0].item()

def BLP_corr_data(num_obs=20000, num_products=3, num_draws=200, m_price=None):
    if m_price is None:
        m_price = torch.full((num_products,), 0.5)
    m, j, n_i = num_obs, num_products, num_draws
    u = torch.rand(m, j, dtype=torch.float32) - 0.5
    z = torch.rand(m, j, dtype=torch.float32) - 0.5
    p_jt_2d = m_price + 0.5 * (z + u + torch.rand(m, j, dtype=torch.float32) - 0.5)
    p_jt = p_jt_2d.unsqueeze(-1)
    cfd = u.unsqueeze(-1)
    d_1, a_0, a_1, a_2 = 0.8, 3, 0.5, 0.5
    v = torch.randn(2, n_i, dtype=torch.float32)
    delta_i = (d_1 * v[0]).unsqueeze(0).unsqueeze(0)
    alpha_i = (a_0 + a_1 * v[0] + a_2 * v[1]).unsqueeze(0).unsqueeze(0)
    u_ijt = delta_i - alpha_i * p_jt + cfd
    s_ijt = torch.softmax(u_ijt, dim=1)
    s_jt = torch.mean(s_ijt, dim=2)
    return s_jt[:, 0], s_jt[:, 1], torch.cat((p_jt_2d, z[:, [0]]), dim=1), p_jt_2d[:, 0], z[:, 0]

def BLP_corr_stats(num_obs=20000, num_products=3, num_draws=200, m_price=None):
    if m_price is None:
        m_price = torch.full((num_products,), 0.5)
    m, j, n_i = num_obs, num_products, num_draws
    p_jt = m_price.unsqueeze(0).unsqueeze(-1).expand(m, j, 1)
    u = torch.rand(m, j, dtype=torch.float32) - 0.5
    cfd = u.unsqueeze(-1)
    d_1, a_0, a_1, a_2 = 0.8, 3, 0.5, 0.5
    v = torch.randn(2, n_i, dtype=torch.float32)
    delta_i = (d_1 * v[0]).unsqueeze(0).unsqueeze(0)
    alpha_i = (a_0 + a_1 * v[0] + a_2 * v[1]).unsqueeze(0).unsqueeze(0)
    u_ijt = delta_i - alpha_i * p_jt + cfd
    s_ijt = torch.softmax(u_ijt, dim=1)
    s_jt = torch.mean(s_ijt, dim=2, keepdim=True)
    aug = (s_ijt / s_jt)[:, 1, :].unsqueeze(1)
    e_own = torch.mean(-alpha_i * s_ijt * (1 - s_ijt) * p_jt / s_jt, dim=(0, 2))
    e_cross = torch.mean(alpha_i * s_ijt * p_jt * aug, dim=(0, 2))
    return e_own[0].item(), e_cross[0].item()

def BLP_ind_data(num_obs=20000, num_products=3, num_draws=200):
    m_price = torch.full((num_products,), 0.5)
    m, j, n_i = num_obs, num_products, num_draws
    u = torch.rand(m, j, dtype=torch.float32) - 0.5
    z = torch.rand(m, j, dtype=torch.float32) - 0.5
    p_jt_2d = m_price + 0.5 * (z + u + torch.rand(m, j, dtype=torch.float32) - 0.5)
    p_jt = p_jt_2d.unsqueeze(-1)
    cfd = u.unsqueeze(-1)
    d_1, a_0, a_2 = 0.8, 3, 0.5
    v = torch.randn(2, n_i, dtype=torch.float32)
    delta_i = (d_1 * v[0]).unsqueeze(0).unsqueeze(0)
    alpha_i = (a_0 + a_2 * v[1]).unsqueeze(0).unsqueeze(0)
    u_ijt = delta_i - alpha_i * p_jt + cfd
    s_ijt = torch.softmax(u_ijt, dim=1)
    s_jt = torch.mean(s_ijt, dim=2)
    return s_jt[:, 0], s_jt[:, 1], torch.cat((p_jt_2d, z[:, [0]]), dim=1), p_jt_2d[:, 0], z[:, 0]

def BLP_ind_stats(num_obs=20000, num_products=3, num_draws=200):
    m_price = torch.full((num_products,), 0.5)
    m, j, n_i = num_obs, num_products, num_draws
    p_jt = m_price.unsqueeze(0).unsqueeze(-1).expand(m, j, 1)
    u = torch.rand(m, j, dtype=torch.float32) - 0.5
    cfd = u.unsqueeze(-1)
    d_1, a_0, a_2 = 0.8, 3, 0.5
    v = torch.randn(2, n_i, dtype=torch.float32)
    delta_i = (d_1 * v[0]).unsqueeze(0).unsqueeze(0)
    alpha_i = (a_0 + a_2 * v[1]).unsqueeze(0).unsqueeze(0)
    u_ijt = delta_i - alpha_i * p_jt + cfd
    s_ijt = torch.softmax(u_ijt, dim=1)
    s_jt = torch.mean(s_ijt, dim=2, keepdim=True)
    aug = (s_ijt / s_jt)[:, 1, :].unsqueeze(1)
    e_own = torch.mean(-alpha_i * s_ijt * (1 - s_ijt) * p_jt / s_jt, dim=(0, 2))
    e_cross = torch.mean(alpha_i * s_ijt * p_jt * aug, dim=(0, 2))
    return e_own[0].item(), e_cross[0].item()