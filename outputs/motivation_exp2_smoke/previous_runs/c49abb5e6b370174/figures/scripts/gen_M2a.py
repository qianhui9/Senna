"""Reproduce M2a independently from measured E2 CSV/JSON artifacts."""
from pathlib import Path
from .plot_results import run
from .utils.data_utils import configuration, initialize, parser

if __name__ == "__main__":
    args = parser().parse_args()
    cfg = configuration(args)
    run(cfg, initialize(cfg, args), only="M2a")
