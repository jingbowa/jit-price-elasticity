"""Online Appendix Table A2: robustness to sample size and number of products.

Correlated random-coefficient logit (d1 = 0.8, a0 = 3, a1 = a2 = 0.5 in the sign convention of
DGP_models/model_module_torch.py), elasticities of products 1 and 2 with respect to product 1's price
at the mean price vector; 50 Monte Carlo replications per specification; the same subsampling size m
for own and cross elasticities.

    Panel A: J = 4 products, m = 7,  T in {5,000, 20,000, 80,000, 320,000}
    Panel B: T = 320,000,    m = 30,  J in {4, 8, 12, 16}
    Panel C: T = 5,120,000,  m = 480, J in {12, 16, 20, 24}

Market shares include an outside good, the formula of Online Appendix C.

Usage (from simulation/):
    python simulation_codes/table_A2_robustness.py [--panel A B C] [--mc 50] [--seed 20260930] [--device cuda|cpu]
Writes simulation_results/table_A2_robustness.json after every specification; specifications
already present are skipped, so an interrupted run resumes.  Runtime on one RTX 3090: Panel A a few
minutes, Panel B about 20 minutes, Panel C about 2 hours.  A GPU is strongly recommended for Panel C.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import torch

from simulation_codes._mc_common import use_device, environment, run_spec, load, save

PARAMS = (0.8, 3.0, 0.5, 0.5)            # d1, a0, a1, a2  (model_module_torch.BLP_corr_data)
PANELS = {
    "A": [dict(T=T, J=4, m=7) for T in (5_000, 20_000, 80_000, 320_000)],
    "B": [dict(T=320_000, J=J, m=30) for J in (4, 8, 12, 16)],
    "C": [dict(T=5_120_000, J=J, m=480) for J in (12, 16, 20, 24)],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", nargs="+", default=["A", "B", "C"], choices=list(PANELS))
    ap.add_argument("--mc", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default=None, help="run one specification, e.g. C|T=5120000|J=24|m=480")
    args = ap.parse_args()

    use_device(args.device)
    path = "./simulation_results/table_A2_robustness.json"
    data = load(path)
    data["meta"].update(table="Online Appendix Table A2", outside_good=True,
                        params_d1_a0_a1_a2=PARAMS, mc=args.mc, seed=args.seed, boot_draws=100,
                        num_consumer_draws=200, environment=environment())
    for pi, panel in enumerate(args.panel):
        for k, spec in enumerate(PANELS[panel]):
            key = f"{panel}|T={spec['T']}|J={spec['J']}|m={spec['m']}"
            if args.only and key != args.only:
                continue
            if key in data["results"] and len(data["results"][key]["iterations"]) >= args.mc:
                print(f"skip {key} (done)")
                continue
            seed = args.seed + 100 * "ABC".index(panel) + k
            print(f"{key}: seed={seed} device={args.device}", flush=True)
            t0 = time.time()
            its = run_spec(seed, spec["T"], spec["J"], spec["m"], spec["m"], PARAMS,
                           True, args.mc, args.device,
                           log=lambda s: print(s, flush=True))
            data["results"][key] = dict(panel=panel, seed=seed, minutes=round((time.time() - t0) / 60, 1),
                                        **spec, iterations=its)
            save(path, data)
            print(f"{key}: done in {(time.time() - t0) / 60:.1f} min -> {path}", flush=True)


if __name__ == "__main__":
    main()
