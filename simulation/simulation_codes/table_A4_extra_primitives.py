"""Online Appendix Table A4: point-wise elasticities for extra primitive parameter values.

Correlated random-coefficient logit of DGP_models/model_module_torch.py (shares normalised over the
J inside goods), J = 4, T = 20,000, 50 Monte Carlo replications, elasticities at the mean price vector,
subsampling size m = 7.
Sign convention of the module: u_ijt = d1 v1 - (a0 + a1 v1 + a2 v2) p_jt + xi_jt.

    Panel A: d1 = 4,   a0 = 1.2, a1 = 0.3, a2 = 0.8   (appendix: b1 = 4,   a0 = -1.2, a1 = 0.3, a2 = 0.8)
    Panel B: d1 = 1.2, a0 = 4,   a1 = 0.8, a2 = 0.3   (appendix: b1 = 1.2, a0 = -4,   a1 = 0.8, a2 = 0.3)

Usage (from simulation/):
    python simulation_codes/table_A4_extra_primitives.py [--mc 50] [--seed 20260930] [--device cuda|cpu]
Writes simulation_results/table_A4_extra_primitives.json after every panel; panels already present
are skipped.  Runtime: about a minute on a GPU.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import torch

from simulation_codes._mc_common import use_device, environment, run_spec, load, save

PANELS = {"A": (4.0, 1.2, 0.3, 0.8), "B": (1.2, 4.0, 0.8, 0.3)}   # d1, a0, a1, a2
T, J, M = 20_000, 4, 7


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    use_device(args.device)
    path = "./simulation_results/table_A4_extra_primitives.json"
    data = load(path)
    data["meta"].update(table="Online Appendix Table A4", T=T, J=J, m=M, outside_good=False,
                        mc=args.mc, seed=args.seed, boot_draws=100,
                        num_consumer_draws=200, environment=environment())
    for k, (panel, params) in enumerate(PANELS.items()):
        key = f"{panel}|d1={params[0]}|a0={params[1]}|a1={params[2]}|a2={params[3]}"
        if key in data["results"] and len(data["results"][key]["iterations"]) >= args.mc:
            print(f"skip {key} (done)")
            continue
        seed = args.seed + 500 + k
        print(f"{key}: seed={seed} device={args.device}", flush=True)
        t0 = time.time()
        its = run_spec(seed, T, J, M, M, params, False, args.mc, args.device,
                       log=lambda s: print(s, flush=True))
        data["results"][key] = dict(panel=panel, params_d1_a0_a1_a2=params, seed=seed,
                                    minutes=round((time.time() - t0) / 60, 1), iterations=its)
        save(path, data)
        print(f"{key}: done in {(time.time() - t0) / 60:.1f} min -> {path}", flush=True)


if __name__ == "__main__":
    main()
