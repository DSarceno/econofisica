# --- FILE: scripts/stages/fetch_data.py ---
"""Stage independiente: solo descargar y cachear los datos crudos."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_prices  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    s = load_prices(
        cfg.data.ticker, cfg.data.start, cfg.data.end,
        cache_dir=cfg.data.cache_dir,
        fallback_ticker=cfg.data.fallback_ticker,
        field=cfg.data.field,
    )
    print(f"OK: {len(s)} observaciones en cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
