import logging
from pathlib import Path

from omegaconf import DictConfig

from utils.config_data_utils import ConfigData

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    """Creates csv file with all configs to pul from based on cutout config yaml."""
    # Using species proportions
    cc = ConfigData(cfg)
    cc.class_distribution()

    if cfg.cutouts.save_csv:
        Path(Path(cfg.data.csvpath).parent).mkdir(parents=True, exist_ok=True)
        cc.df.to_csv(cfg.data.csvpath, index=False)
