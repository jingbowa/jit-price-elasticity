"""Correlated random-coefficient logit DGP for the robustness tables (Online Appendix Tables A2 and A4).

Same parameterisation and sign convention as ``model_module_torch.BLP_corr_data``:

    u_ijt = d1 * v1_i - (a0 + a1 * v1_i + a2 * v2_i) * p_jt + xi_jt,
    p_jt  = 0.5 + 0.5 z_jt + 0.5 xi_jt + 0.5 e_jt,

with v1, v2 ~ N(0, 1) for ``num_draws`` consumers shared by all markets of a data set and
xi, z, e ~ U[-0.5, 0.5).  Two share normalisations are available:

* ``outside_good=True``:  s_ijt = exp(u_ijt) / (1 + sum_k exp(u_ikt)), the formula of Online
  Appendix C (Table A2).
* ``outside_good=False``: s_ijt = exp(u_ijt) / sum_k exp(u_ikt) over the J inside goods, the
  normalisation of ``model_module_torch.py`` (Table A4).

Differences from the modules it mirrors are computational only: every random number is drawn from a
CPU ``torch.Generator``, shares are computed in chunks of markets on ``device`` (so T = 5,120,000 fits in memory), and the arithmetic is
float64 throughout.
"""
import torch


def draw_consumers(gen, num_draws, dtype=torch.float64):
    v = torch.randn(2, num_draws, generator=gen, dtype=dtype)
    return v[0], v[1]


def _shares(u, outside_good):
    """Choice probabilities over products (dim 1) for utilities u of shape (markets, J, consumers)."""
    if outside_good:
        eu = torch.exp(u)
        return eu / (1 + eu.sum(dim=1, keepdim=True))
    return torch.softmax(u, dim=1)


def _coefficients(v1, v2, params, device):
    d1, a0, a1, a2 = params
    delta = (d1 * v1).to(device).view(1, 1, -1)
    alpha = (a0 + a1 * v1 + a2 * v2).to(device).view(1, 1, -1)
    return delta, alpha


def rc_corr_data(gen, num_obs, num_products, params, outside_good, num_draws=200,
                 dtype=torch.float64, device="cpu", chunk=20000):
    """One simulated data set.

    Returns (y1, y2, XZ): market shares of products 1 and 2 and the regressor matrix
    [p_1, ..., p_J, z_1] (CPU tensors, float64), as in ``BLP_corr_data``.
    """
    J = num_products
    xi = torch.rand(num_obs, J, generator=gen, dtype=dtype) - 0.5
    z = torch.rand(num_obs, J, generator=gen, dtype=dtype) - 0.5
    e = torch.rand(num_obs, J, generator=gen, dtype=dtype) - 0.5
    p = 0.5 + 0.5 * (z + xi + e)
    v1, v2 = draw_consumers(gen, num_draws, dtype)
    delta, alpha = _coefficients(v1, v2, params, device)
    y1 = torch.empty(num_obs, dtype=dtype)
    y2 = torch.empty(num_obs, dtype=dtype)
    for s in range(0, num_obs, chunk):
        pc = p[s:s + chunk].to(device).unsqueeze(-1)
        xc = xi[s:s + chunk].to(device).unsqueeze(-1)
        s_jt = _shares(delta - alpha * pc + xc, outside_good).mean(dim=2)
        y1[s:s + chunk] = s_jt[:, 0].cpu()
        y2[s:s + chunk] = s_jt[:, 1].cpu()
    XZ = torch.cat((p, z[:, [0]]), dim=1)
    return y1, y2, XZ


def rc_corr_truth(gen, num_obs, num_products, params, outside_good, num_draws=200, price=0.5,
                  dtype=torch.float64, device="cpu", chunk=20000):
    """True own- and cross-price elasticity (products 1 and 2 w.r.t. product 1's price) at
    p = price for every product, averaged over markets and consumers: the formula of
    ``BLP_corr_stats``, with its own fresh draws of xi and of the consumers."""
    J = num_products
    xi = torch.rand(num_obs, J, generator=gen, dtype=dtype) - 0.5
    v1, v2 = draw_consumers(gen, num_draws, dtype)
    delta, alpha = _coefficients(v1, v2, params, device)
    own_sum = torch.zeros((), dtype=dtype, device=device)
    cross_sum = torch.zeros((), dtype=dtype, device=device)
    for s in range(0, num_obs, chunk):
        xc = xi[s:s + chunk].to(device).unsqueeze(-1)
        sh = _shares(delta - alpha * price + xc, outside_good)            # (c, J, n)
        s_jt = sh.mean(dim=2, keepdim=True)                                # (c, J, 1)
        s0, s1 = sh[:, [0], :], sh[:, [1], :]
        own_sum += (-alpha * s0 * (1 - s0) * price / s_jt[:, [0], :]).sum()
        cross_sum += (alpha * s0 * price * s1 / s_jt[:, [1], :]).sum()
    n = num_obs * num_draws
    return (own_sum / n).item(), (cross_sum / n).item()
