import logging

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

import config_data  # Do not remove
import convert_mask2yolo
import create_dataset
import crop_images

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_CURATEDATA(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    log.info(f"Starting task {cfg.general.task}")
    task = get_method(f"{cfg.general.task}.main")
    task(cfg)


if __name__ == "__main__":
    run_CURATEDATA()
