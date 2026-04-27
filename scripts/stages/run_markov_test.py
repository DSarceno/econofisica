# --- FILE: scripts/stages/run_markov_test.py ---
"""Stage independiente: ejecutar test de Chapman-Kolmogorov."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_prices  # noqa: E402
from src.markov.chapman_kolmogorov import chapman_kolmogorov_test  # noqa: E402
from src.preprocessing.returns import log_returns, standardize  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    rng = seed_everything(cfg.project.random_seed)

    prices = load_prices(
        cfg.data.ticker, cfg.data.start, cfg.data.end,
        cache_dir=cfg.data.cache_dir, fallback_ticker=cfg.data.fallback_ticker,
        field=cfg.data.field,
    )
    r = log_returns(prices)
    if cfg.preprocessing.standardize.enabled:
        r = standardize(r, method=cfg.preprocessing.standardize.method)

    results = []
    for dt in cfg.markov_test.delta_t_steps:
        if dt < 2:
            continue
        res = chapman_kolmogorov_test(
            r,
            delta_t_steps=int(dt),
            n_bins=cfg.markov_test.n_bins_state,
            divergence=cfg.markov_test.divergence,
            bootstrap_samples=cfg.markov_test.bootstrap_samples,
            bootstrap_confidence=cfg.markov_test.bootstrap_confidence,
            rng=rng,
        )
        results.append(res.to_dict())

    save_json({"results": results}, Path(cfg.project.output_root) / "chapman_kolmogorov.json")
    for r in results:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
