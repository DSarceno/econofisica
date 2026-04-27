# --- FILE: scripts/stages/run_km.py ---
"""Stage independiente: estimar coeficientes de Kramers-Moyal solamente."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.loaders import load_prices  # noqa: E402
from src.kramers_moyal.estimators import estimate_km_coefficients  # noqa: E402
from src.preprocessing.returns import log_returns, standardize  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import save_dataframe, save_json  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    prices = load_prices(
        cfg.data.ticker, cfg.data.start, cfg.data.end,
        cache_dir=cfg.data.cache_dir, fallback_ticker=cfg.data.fallback_ticker,
        field=cfg.data.field,
    )
    r = log_returns(prices)
    if cfg.preprocessing.standardize.enabled:
        r = standardize(
            r,
            method=cfg.preprocessing.standardize.method,
            rolling_window=cfg.preprocessing.standardize.rolling_window,
        )

    km = estimate_km_coefficients(
        r,
        orders=[1, 2] + list(cfg.kramers_moyal.pawula.compute_orders),
        n_bins=cfg.kramers_moyal.n_bins,
        binning=cfg.kramers_moyal.binning,
        delta_t_step=cfg.kramers_moyal.delta_t_step,
        min_obs_per_bin=cfg.kramers_moyal.min_obs_per_bin,
    )
    df = pd.DataFrame({"x": km.x_centers})
    for k, v in km.coefficients.items():
        df[f"D{k}"] = v
    save_dataframe(df, Path(cfg.project.output_root) / "km_coefficients.parquet")
    save_json({"delta_t": km.delta_t, "n_bins": km.x_centers.size, "n_used": km.n_used},
              Path(cfg.project.output_root) / "km_summary.json")
    print(f"KM listo: {len(df)} bins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
