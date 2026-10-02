"""Shared Monte Carlo loop for the robustness tables (A2, A4).

Point estimates come from ``bnn_modules.bnn_torch.p_elas_2scale_torch`` and standard errors from
``p_elas_2scale_torch_boot`` (100 bootstrap draws), exactly as in ``table_1_models.py``.  The only
change is computational: the (deterministic) jackknife weights are computed once per (n, p, m) on the
CPU, as the package computes them, and then kept on the estimation device.
"""
import functools
import json
import os
import platform
import time

import torch

import bnn_modules.bnn_torch as bt
from bnn_modules.bnn_torch import p_elas_2scale_torch, p_elas_2scale_torch_boot
from DGP_models.model_module_rc import rc_corr_data, rc_corr_truth

_weight_2scale_cpu = bt.weight_2scale_torch
_DEVICE = {"device": torch.device("cpu")}


@functools.lru_cache(maxsize=16)
def _cached_weights(n, p, s, dtype):
    return _weight_2scale_cpu(n, p, s, dtype=dtype).to(_DEVICE["device"])


def _weight_2scale_cached(n, p, s, dtype=torch.float64, device=None):
    return _cached_weights(n, p, int(s), dtype)


def use_device(device):
    _DEVICE["device"] = torch.device(device)
    _cached_weights.cache_clear()
    bt.weight_2scale_torch = _weight_2scale_cached   # looked up at call time by p_elas_2scale_torch


def environment():
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "platform": platform.platform(), "device": str(_DEVICE["device"])}
    if _DEVICE["device"].type == "cuda":
        prop = torch.cuda.get_device_properties(_DEVICE["device"])
        info.update(gpu=prop.name, gpu_multiprocessors=prop.multi_processor_count)
    return info


def run_spec(seed, T, J, m_own, m_cross, params, outside_good, mc, device, log=print):
    """Monte Carlo for one specification.  Seeds the data generator and the global CPU generator
    (used by the bootstrap) so each specification is reproducible on its own."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    xz = torch.tensor([0.5] * J + [0.0], dtype=torch.float64, device=device)
    out = []
    for it in range(mc):
        t0 = time.time()
        y1, y2, XZ = rc_corr_data(gen, T, J, params, outside_good, device=device)
        true_own, true_cross = rc_corr_truth(gen, T, J, params, outside_good, device=device)
        y1, y2, XZ = y1.to(device), y2.to(device), XZ.to(device)
        est_own = p_elas_2scale_torch(y1, XZ, xz, 0, J, m_own).item()
        _, var_own = p_elas_2scale_torch_boot(y1, XZ, xz, 0, J, m_own)
        est_cross = p_elas_2scale_torch(y2, XZ, xz, 0, J, m_cross).item()
        _, var_cross = p_elas_2scale_torch_boot(y2, XZ, xz, 0, J, m_cross)
        out.append({"iteration": it, "true_own": true_own, "true_cross": true_cross,
                    "est_own": est_own, "est_cross": est_cross,
                    "se_own": torch.sqrt(var_own).item(), "se_cross": torch.sqrt(var_cross).item()})
        log(f"    iteration {it + 1}/{mc}: own {est_own:.3f} (true {true_own:.3f}), "
            f"cross {est_cross:.3f} (true {true_cross:.3f}), {time.time() - t0:.1f}s")
        del y1, y2, XZ
    return out


def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"meta": {}, "results": {}}


def save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)
