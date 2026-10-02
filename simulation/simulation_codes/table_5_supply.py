import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch, functorch
from joblib import Parallel, delayed
from bnn_modules.bnn_module_supply import p_elas, p_elas_boot
from tqdm import tqdm
import json

def market_share(p_jt, cfd, eps, delta_i, alpha_i):
    u_ijt = delta_i[None, :] - alpha_i[None, :]*p_jt.unsqueeze(-1) + 0.5*cfd.unsqueeze(-1) + 0.0*eps.unsqueeze(-1)
    s_ijt = torch.exp(u_ijt)/(1 + torch.sum(torch.exp(u_ijt), dim=0, keepdim=True))
    s_jt = torch.mean(s_ijt, dim=1)
    partial = torch.mean(-alpha_i*s_ijt*(1-s_ijt), dim=1)
    return s_jt, partial

def loss_function(p_jt, cost, cfd, eps, delta_i, alpha_i):
    s_jt, partial = market_share(p_jt, cfd, eps, delta_i, alpha_i)
    y = partial*(p_jt - cost) + s_jt + 0.1*cfd
    loss = torch.sum(y**2)
    return loss

def equilibrium_price(cost, cfd, eps, delta_i, alpha_i, iterations = 1000):
    price = torch.zeros_like(cost, requires_grad= True)
    optimizer = torch.optim.Adam([price], lr=0.1)
    optimizer.zero_grad()
    # Solve the system of equations
    for _ in range(iterations):        
        optimizer.zero_grad()
        loss = loss_function(price, cost, cfd, eps, delta_i, alpha_i)
        loss.backward()
        optimizer.step()
        if loss < 0.0001:
            break
    return price.detach()

def BLP_supply_data(delta_i, alpha_i, num_obs, num_products):
    m, j = num_obs, num_products

    cfd = torch.rand(m, j) - 0.5
    eps = torch.rand(m, j) - 0.5
    supply_z = torch.rand(m, j) - 0.5
    cost = 0.5*torch.rand(m, j)

    price_star = Parallel(n_jobs=80)(delayed(equilibrium_price)(cost[i, :], cfd[i, :], eps[i, :], delta_i, alpha_i) for i in range(m))
    p_jt_2d = torch.stack(price_star) + 0.5*supply_z

    s_jt, _ = functorch.vmap(market_share, in_dims=(0, 0, 0, None, None))(p_jt_2d, cfd, eps, delta_i, alpha_i)
    return  s_jt[:,0].numpy(), s_jt[:,1].numpy(), torch.cat((p_jt_2d, supply_z[:,[0]]), dim=1).numpy(), p_jt_2d[:, 0].numpy(), supply_z[:, 0].numpy()

def BLP_supply_stats(m_price, delta_i, alpha_i, num_obs, num_products):
    m, j = num_obs, num_products
    p_jt = m_price[None, :, None]

    delta_i, alpha_i = delta_i[None, None, :], alpha_i[None, None, :]
    cfd = (torch.rand(m, j) - 0.5)[:, :, None]
    eps = (torch.rand(m, j) - 0.5)[:, :, None]

    u_ijt = delta_i - alpha_i*p_jt + 0.5*cfd + 0.0*eps
    s_ijt = torch.exp(u_ijt)/(1 + torch.sum(torch.exp(u_ijt), dim = 1, keepdim=True))
    s_jt = torch.mean(s_ijt, 2, keepdim = True)
    aug = (s_ijt/s_jt)[:, [1], :]
    e_own = torch.mean(-alpha_i*s_ijt*(1-s_ijt)*p_jt/s_jt, dim = (0, 2))
    e_cross = torch.mean(alpha_i*s_ijt*p_jt*aug, dim = (0, 2))
    return e_own[0].item(), e_cross[0].item()

def sim_corr(num_monte = 50):
    true_own, true_cross = np.zeros(num_monte), np.zeros(num_monte)
    est_own, est_cross = np.zeros(num_monte), np.zeros(num_monte)
    se_own, se_cross = np.zeros(num_monte), np.zeros(num_monte)
    
    for i in tqdm(range(num_monte)):
        num_obs, num_products, num_draws = 20000, 4, 1000
        s_own, s_cross = 20, 5
        d_1, a_0, a_1, a_2 = 0.8, 3.0, 0.5, 0.5
        v1, v2 = torch.randn(num_draws), torch.randn(num_draws)
        delta_i = (d_1*v1)
        alpha_i = (a_0 + a_1*v1 + a_2*v2)

        m_price = torch.tensor([1.0]*4)
        true_own[i], true_cross[i] = BLP_supply_stats(m_price, delta_i, alpha_i, num_obs, num_products)

        y1, y2, XZ, p1, z = BLP_supply_data(delta_i, alpha_i, num_obs, num_products)

        xz = np.array([1.0]*4 + [0])
        est_own[i], est_cross[i] = p_elas(y1, XZ, xz, p1, z, s_own), p_elas(y2, XZ, xz, p1, z, s_cross)

        se_own[i], se_cross[i] = p_elas_boot(y1, XZ, xz, p1, z, s_own), p_elas_boot(y2, XZ, xz, p1, z, s_cross)
    return {
        'true_own': true_own.tolist(),
        'true_cross': true_cross.tolist(),
        'est_own': est_own.tolist(),
        'est_cross': est_cross.tolist(),
        'se_own': se_own.tolist(),
        'se_cross': se_cross.tolist()
    }


def sim_ind(num_monte = 50):
    true_own, true_cross = np.zeros(num_monte), np.zeros(num_monte)
    est_own, est_cross = np.zeros(num_monte), np.zeros(num_monte)
    se_own, se_cross = np.zeros(num_monte), np.zeros(num_monte)
    
    for i in tqdm(range(num_monte)):
        num_obs, num_products, num_draws = 20000, 4, 1000
        s_own, s_cross = 20, 5
        d_1, a_0, a_1, a_2 = 0.8, 3.0, 0.5, 0.5
        v1, v2 = torch.randn(num_draws), torch.randn(num_draws)
        delta_i = (d_1*v1)
        alpha_i = (a_0 + 0.0*a_1*v1 + a_2*v2)

        m_price = torch.tensor([1.0]*4)
        true_own[i], true_cross[i] = BLP_supply_stats(m_price, delta_i, alpha_i, num_obs, num_products)

        y1, y2, XZ, p1, z = BLP_supply_data(delta_i, alpha_i, num_obs, num_products)

        xz = np.array([1.0]*4 + [0])
        est_own[i], est_cross[i] = p_elas(y1, XZ, xz, p1, z, s_own), p_elas(y2, XZ, xz, p1, z, s_cross)

        se_own[i], se_cross[i] = p_elas_boot(y1, XZ, xz, p1, z, s_own), p_elas_boot(y2, XZ, xz, p1, z, s_cross)
    return {
        'true_own': true_own.tolist(),
        'true_cross': true_cross.tolist(),
        'est_own': est_own.tolist(),
        'est_cross': est_cross.tolist(),
        'se_own': se_own.tolist(),
        'se_cross': se_cross.tolist()
    }


def sim_Logit(num_monte = 50):
    true_own, true_cross = np.zeros(num_monte), np.zeros(num_monte)
    est_own, est_cross = np.zeros(num_monte), np.zeros(num_monte)
    se_own, se_cross = np.zeros(num_monte), np.zeros(num_monte)
    
    for i in tqdm(range(num_monte)):
        num_obs, num_products= 20000, 4
        s_own, s_cross = 20, 5
        delta_i = torch.tensor((4.0,))
        alpha_i = torch.tensor((2.0,))

        m_price = torch.tensor([1.0]*4)
        true_own[i], true_cross[i] = BLP_supply_stats(m_price, delta_i, alpha_i, num_obs, num_products)

        y1, y2, XZ, p1, z = BLP_supply_data(delta_i, alpha_i, num_obs, num_products)

        xz = np.array([1.0]*4 + [0])
        est_own[i], est_cross[i] = p_elas(y1, XZ, xz, p1, z, s_own), p_elas(y2, XZ, xz, p1, z, s_cross)

        se_own[i], se_cross[i] = p_elas_boot(y1, XZ, xz, p1, z, s_own), p_elas_boot(y2, XZ, xz, p1, z, s_cross)
    return {
        'true_own': true_own.tolist(),
        'true_cross': true_cross.tolist(),
        'est_own': est_own.tolist(),
        'est_cross': est_cross.tolist(),
        'se_own': se_own.tolist(),
        'se_cross': se_cross.tolist()
    }

sim_corr_results = sim_corr()
sim_ind_results = sim_ind()
sim_logit_results = sim_Logit()

all_results = {
    "sim_corr": sim_corr_results,
    "sim_ind": sim_ind_results,
    "sim_logit": sim_logit_results
}

with open("./simulation_results/table_5_supply.json", "w") as f:
    json.dump(all_results, f)