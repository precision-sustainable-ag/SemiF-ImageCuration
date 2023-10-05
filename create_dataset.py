import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from shutil import copy2

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

log = logging.getLogger(__name__)


class BaseDataset:
    """
    Base class for datasets that provides general utilities for reading and copying files.
    """

    def __init__(self, cfg):
        self.df = pd.read_csv(cfg.data.csvpath, low_memory=False)
        self.cpu_denominator = cfg.general.cpu_denominator

    @staticmethod
    def copy_file(src_path, dest_dir):
        """Copy a single file."""
        copy2(src_path, dest_dir)

    def copy_files(self, src_list, destination_dir):
        available_cpus = int(len(os.sched_getaffinity(0)) / self.cpu_denominator)
        with ProcessPoolExecutor(available_cpus) as executor:
            # Create a list of futures
            futures = [
                executor.submit(self.copy_file, path, destination_dir)
                for path in src_list
            ]

            # Use tqdm to display progress
            for _ in tqdm(
                as_completed(futures), total=len(src_list), desc="Copying files"
            ):
                pass


class CreateCutoutDataset(BaseDataset):
    """
    Dataset class specifically for handling cutout datasets.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.srccutoutdir = cfg.data.cutoutdir
        self.cutoutdstdir = Path(cfg.general.resultsdir, "cutouts")
        self.cutoutdstdir.mkdir(parents=True, exist_ok=True)

    def get_images(self):
        """Compute paths for cutout datasets."""
        self._compute_paths("cutout_path", ".png")
        self._compute_paths("cropout_path", ".jpg")
        self._compute_paths("cutout_masks_path", "_mask.png")
        self._compute_paths("cutout_metadata_path", ".json")
        return self.df

    def _compute_paths(self, column_name, file_extension):
        self.df[column_name] = (
            self.srccutoutdir
            + "/"
            + self.df["batch_id"]
            + "/"
            + (self.df["cutout_id"] + file_extension)
        )


class CreateDataset(BaseDataset):
    """
    Dataset class for handling full resolution datasets.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.srcimagedir = cfg.data.srcbatchdir
        self.dstdir = cfg.general.resultsdir
        self.dir_structure = {
            "images": Path(self.dstdir, "images"),
            "semantic_masks": Path(self.dstdir, "meta_masks", "semantic_masks"),
            "instance_masks": Path(self.dstdir, "meta_masks", "instance_masks"),
            "metadata": Path(self.dstdir, "metadata"),
        }
        for dir_path in self.dir_structure.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_images(self):
        """Compute paths for full resolution datasets."""
        self._compute_paths("image_path", "images", ".jpg")
        self._compute_paths("semantic_mask_path", "meta_masks/semantic_masks", ".png")
        self._compute_paths("instance_mask_path", "meta_masks/instance_masks", ".png")
        self._compute_paths("meta_path", "metadata", ".json")
        return self.df

    def _compute_paths(self, column_name, sub_directory, file_extension):
        self.df[column_name] = (
            self.srcimagedir
            + "/"
            + self.df["batch_id"]
            + "/"
            + sub_directory
            + "/"
            + (self.df["image_id"] + file_extension)
        )


def main(cfg: DictConfig) -> None:
    """
    Main function to put together dataset creation tasks.
    """
    cc = CreateDataset(cfg)
    co = CreateCutoutDataset(cfg)

    datasets = {
        cc: {
            "semantic_mask_path": cc.dir_structure["semantic_masks"],
            "instance_mask_path": cc.dir_structure["instance_masks"],
            "meta_path": cc.dir_structure["metadata"],
            "image_path": cc.dir_structure["images"],
        },
        co: {
            "cutout_path": co.cutoutdstdir,
            "cropout_path": co.cutoutdstdir,
            "cutout_masks_path": co.cutoutdstdir,
            "cutout_metadata_path": co.cutoutdstdir,
        },
    }

    for dataset, paths in datasets.items():
        df = dataset.get_images()
        for column, dest_dir in paths.items():
            src_list = [Path(x) for x in df[column]]
            dataset.copy_files(src_list, dest_dir)
