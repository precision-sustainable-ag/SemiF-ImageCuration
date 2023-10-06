import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig


class ImageConverter:
    """Class to handle the conversion of mask images to formatted text data."""

    @staticmethod
    def get_contours(masked_img):
        """Extract contours from a masked image."""
        contours, _ = cv2.findContours(masked_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        return contours

    def convert_to_txt(self, img_path):
        """Convert mask image to formatted text data."""
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        unique_labels = set(np.unique(image)) - {0}

        result = []
        for label in unique_labels:
            masked_img = np.where(image == label, 255, 0).astype(np.uint8)
            contours = self.get_contours(masked_img)
            for contour in contours:
                simplified_contour = cv2.approxPolyDP(
                    contour, epsilon=1e-2, closed=False
                )
                coords = [
                    (point[0][0] / image.shape[1], point[0][1] / image.shape[0])
                    for point in simplified_contour
                ]
                if len(coords) >= 3:
                    flattened_coords = [coord for pair in coords for coord in pair]
                    result.append(f"{label} " + " ".join(map(str, flattened_coords)))
        return "\n".join(result)


class MaskGenerator:
    """Class to handle the generation of masks from formatted text data."""

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def _parse_line(self, line):
        """Parse a line from the formatted text data to extract class index and coordinates."""
        parts = list(map(float, line.split()))
        class_index = int(parts[0])
        coords = [
            (int(parts[i] * self.width), int(parts[i + 1] * self.height))
            for i in range(1, len(parts), 2)
        ]
        return class_index, coords

    def generate_mask_from_txt(self, txt_data):
        """Generate a mask from formatted text data."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for line in txt_data.splitlines():
            label = int(line.split()[0])
            coords = list(map(float, line.split()[1:]))
            points = [
                (int(x * self.width), int(y * self.height))
                for x, y in zip(coords[::2], coords[1::2])
            ]
            contour = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
            cv2.drawContours(mask, [contour], -1, 255, thickness=1)

        # Clone the mask for processing
        temp_mask = mask.copy()

        # Identify all regions within the thin outlines
        contours, _ = cv2.findContours(
            temp_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            temp_fill = temp_mask.copy()
            cv2.drawContours(temp_fill, [contour], -1, 255, -1)
            if (
                np.any(temp_fill[0, :])
                or np.any(temp_fill[-1, :])
                or np.any(temp_fill[:, 0])
                or np.any(temp_fill[:, -1])
            ):
                continue
            else:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        return mask


def process_image(img_path, output_dir):
    """Process a mask image and save its converted data in the output directory."""
    converter = ImageConverter()
    txt_data = converter.convert_to_txt(img_path)

    # Save txt data
    txt_path = output_dir / f"{Path(img_path).stem}.txt"
    with open(txt_path, "w") as f:
        f.write(txt_data)


def mse(imageA, imageB):
    """Compute the mean squared error between two images."""
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def test_sample_conversion(img_path, output_dir, write_samples=False):
    """Test the conversion by regenerating the mask from the saved text data and calculating the MSE."""
    txt_path = output_dir / f"{Path(img_path).stem}.txt"

    # Read txt data
    with open(txt_path, "r") as f:
        txt_data = f.read()

    # Convert text back to mask image
    original_image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask_gen = MaskGenerator(original_image.shape[1], original_image.shape[0])
    regenerated_mask = mask_gen.generate_mask_from_txt(txt_data)

    # Compute MSE between original image and regenerated mask
    error = mse(original_image, regenerated_mask)
    if write_samples:
        sample_mask_dir = Path(output_dir.parent, "test_sample_mask")
        sample_mask_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample_mask_dir / Path(img_path).name), regenerated_mask)
        new_imgpath = f"{Path(sample_mask_dir, Path(img_path).stem + '.jpg')}"
        img_path = str(img_path).replace("masks", "images")
        img_path = str(img_path).replace("png", "jpg")
        shutil.copy2(img_path, new_imgpath)

    return img_path, error


def get_inputmask_and_outpout_dirs(cfg):
    # ... [Rest of the implementation for different configuration scenarios]
    if cfg.mask2yolo.cutouts_or_imgs.upper() == "CROPPED IMAGES":
        mask_dir = Path(
            cfg.general.resultsdir,
            f"cropped_masks_{cfg.crop_split.M}x{cfg.crop_split.N}",
        )
        output_dir = Path(
            cfg.general.resultsdir,
            f"cropped_yolo_masks_{cfg.crop_split.M}x{cfg.crop_split.N}",
        )
        output_dir.mkdir(exist_ok=True, parents=True)
    return mask_dir, output_dir


def test_yolo_masks(test_samples, write_samples, output_dir, mask_images, executor):
    test_images = random.sample(mask_images, test_samples)
    futures = [
        executor.submit(test_sample_conversion, img_path, output_dir, write_samples)
        for img_path in test_images
    ]
    errors = [future.result()[1] for future in as_completed(futures)]

    avg_mse = sum(errors) / len(errors)
    print(f"Avg MSE for Test (n={test_samples}): {avg_mse})")


def main(cfg: DictConfig):
    """Main function for converting masks to YOLO format and optionally test a sample of the conversions."""

    test_samples = cfg.mask2yolo.testing.sample
    write_samples = cfg.mask2yolo.testing.write_samples

    # Process according to the configuration
    mask_dir, output_dir = get_inputmask_and_outpout_dirs(cfg)

    mask_images = list(mask_dir.glob("*.png"))

    with ProcessPoolExecutor() as executor:
        # Convert all mask images
        futures = [
            executor.submit(process_image, img_path, output_dir)
            for img_path in mask_images
        ]

        for future in as_completed(futures):
            future.result()  # Raise any exceptions from the worker processes

        if cfg.mask2yolo.testing.test:
            test_yolo_masks(
                test_samples, write_samples, output_dir, mask_images, executor
            )
