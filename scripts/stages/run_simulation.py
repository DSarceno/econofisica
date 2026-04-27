# --- FILE: scripts/stages/run_simulation.py ---
"""Stage independiente: simular trayectorias con un modelo previamente entrenado.

Para uso ad-hoc; el orquestador completo esta en scripts/run_pipeline.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.models.ornstein_uhlenbeck import OrnsteinUhlenbeck  # noqa: E402
from src.simulation.euler_maruyama import simulate_paths  # noqa: E402
from src.utils.io import save_array  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--D", type=float, default=0.5)
    parser.add_argument("--n-paths", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--burn", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--out", default="data/processed/sim_ou.npy")
    args = parser.parse_args()

    setup_logging()
    m = OrnsteinUhlenbeck(theta=args.theta, mu=args.mu, diffusion_coef=args.D)
    paths = simulate_paths(
        m,
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        dt=args.dt,
        burn_in=args.burn,
        parent_seed=args.seed,
    )
    p = save_array(paths.astype(np.float32), args.out)
    print(f"Simulacion guardada en: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
