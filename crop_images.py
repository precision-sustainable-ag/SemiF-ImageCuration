import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
from omegaconf import DictConfig
from tqdm import tqdm

log = logging.getLogger(__name__)


class CropImages:
    """Handles operations related to cropping images and their corresponding masks."""

    def __init__(self, cfg) -> None:
        """
        Initialize the class with configuration settings.

        Args:
            cfg (DictConfig): Configuration object containing all the settings.
        """
        # yapf: disable
        self.src_imgs = sorted([x for x in Path(cfg.data.images).glob("*.jpg")])
        self.src_masks = sorted([x for x in Path(cfg.data.semantic_masks).glob("*.png")])

        # yapf: enable
        assert set([x.stem for x in self.src_imgs]) == set(
            [x.stem for x in self.src_masks]
        ), "Unequal image and mask filenane stems each other"

        self.split_imgs = Path(
            cfg.general.resultsdir,
            f"cropped_images_{cfg.crop_split.M}x{cfg.crop_split.N}",
        )
        self.split_imgs.mkdir(exist_ok=True, parents=True)
        self.split_masks = Path(
            cfg.general.resultsdir,
            f"cropped_masks_{cfg.crop_split.M}x{cfg.crop_split.N}",
        )
        self.split_masks.mkdir(exist_ok=True, parents=True)

        self.M = cfg.crop_split.M
        self.N = cfg.crop_split.N

        self.shape_flag = True

        self.exclude_empty_tiles = cfg.crop_split.exclude_empty_tiles


def tile(img, M, N):
    """Splits an image into smaller tiles of size MxN."""
    tiles = [
        img[x : x + M, y : y + N]
        for x in range(0, img.shape[0], M)
        for y in range(0, img.shape[1], N)
    ]

    return tiles


def save_imgs_masks(dst_img_dir, dst_mask_dir, img, mask, im_stem, count):
    """Saves the image and mask tiles to the specified directories."""
    cv2.imwrite(str(Path(dst_img_dir, im_stem + f"_{count}.jpg")), img)
    cv2.imwrite(str(Path(dst_mask_dir, im_stem + f"_{count}.png")), mask)


def save_tiles(
    src_img_path,
    src_mask_path,
    M,
    N,
    dst_img_dir,
    dst_mask_dir,
    exclude_empty_tiles=True,
):
    """
    Processes an image and its mask to generate and save tiles.

    Args:
        src_img_path (Path): Path to the source image.
        src_mask_path (Path): Path to the source mask.
        M (int): Tile height.
        N (int): Tile width.
        dst_img_dir (Path): Directory to save the image tiles.
        dst_mask_dir (Path): Directory to save the mask tiles.
        exclude_empty_tiles (bool, optional): Flag to exclude tiles without any content. Defaults to True.
    """
    im = Path(src_img_path)

    img = cv2.imread(str(src_img_path))
    mask = cv2.imread(str(src_mask_path))[:, :, 0]

    spl_imgs = tile(img, M, N)
    spl_masks = tile(mask, M, N)

    count = 0

    for i, m in zip(spl_imgs, spl_masks):
        if i.shape[:2] != (M, N):
            # shape_flag = False
            continue

        if exclude_empty_tiles:
            if m.max() == 0:
                continue

        save_imgs_masks(dst_img_dir, dst_mask_dir, i, m, im.stem, count)

        count += 1


def main(cfg: DictConfig) -> None:
    """
    Main function to start the cropping process.

    Reads configuration, prepares directories and initiates concurrent processing of images and masks.
    """
    log.info("Cropping images.")

    ci = CropImages(cfg)
    a_args = ci.src_imgs
    b_arg = ci.src_masks
    second_arg = ci.M
    third_arg = ci.N
    fourth_arg = ci.split_imgs
    fifth_arg = ci.split_masks
    available_cpus = int(len(os.sched_getaffinity(0)) / cfg.general.cpu_denominator)

    # Using ProcessPoolExecutor
    with ProcessPoolExecutor(available_cpus) as executor:
        futures = [
            executor.submit(
                save_tiles,
                img_path,
                mask_path,
                second_arg,
                third_arg,
                fourth_arg,
                fifth_arg,
            )
            for img_path, mask_path in zip(a_args, b_arg)
        ]

        # Display progress using tqdm
        for _ in tqdm(
            as_completed(futures), total=len(a_args), desc="Processing images"
        ):
            pass
