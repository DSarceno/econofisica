# --- FILE: tests/unit/test_config.py ---
from pathlib import Path

from src.utils.config import load_config


def test_load_pipeline_config():
    cfg = load_config(Path(__file__).parents[2] / "configs" / "pipeline.yaml")
    assert cfg.project.name
    assert cfg.data.ticker == "^GSPC"
    assert cfg.simulation.n_paths > 0
    assert cfg.kramers_moyal.n_bins > 0
    assert cfg.markov_test.delta_t_steps[0] == 1
