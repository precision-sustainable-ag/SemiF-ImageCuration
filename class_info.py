import csv
import json
import logging
from ast import literal_eval
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class PresentClasses:
    def __init__(self, cfg) -> None:
        self.metadata = Path(cfg.data.metadata).glob("*.json")

        self.species_info_json = cfg.classes.speciesinfo_json
        self.present_class_info = cfg.classes.present_class_info
        self.all_class_info = cfg.classes.all_class_info
        self.true_mask_palette_path = cfg.classes.true_mask_palette
        self.common_names_file = cfg.classes.common_name_file
        self.USDA_symbols_file = cfg.classes.USDA_symbols_file
        self.datacsv = cfg.data.datacsv
        self.common_name_distribution = cfg.data.common_name_distribution
        self.batch_distribution = cfg.data.batch_distribution
        self.coco_categories = cfg.classes.coco_categories

    def read_metadata(self, path):
        with open(path, "r") as f:
            data = json.loads(f.read())
        return data

    def present_classes_csv(self, save=True):
        df = pd.read_csv(self.datacsv)
        df = df[
            ["class_id", "USDA_symbol", "species", "common_name", "hex", "r", "g", "b"]
        ]
        df = (
            df.drop_duplicates(subset=["class_id"])
            .sort_values("class_id")
            .reset_index(drop=True)
        )
        if save:
            df.to_csv(self.present_class_info, index=False)
        return df

    def all_classes_csv(self):
        meta = self.read_metadata(self.species_info_json)
        metas = [meta["species"][i] for i in meta["species"].keys()]
        df = pd.DataFrame.from_records(metas)
        df[["class_id", "USDA_symbol", "common_name", "rgb"]].to_csv(
            self.all_class_info, index=False
        )

    def true_mask_palette(self):
        meta = self.read_metadata(self.species_info_json)
        metas = [meta["species"][i] for i in meta["species"].keys()]
        df = pd.DataFrame.from_records(metas)
        df["true_mask_palette"] = None
        df["true_mask_palette"] = df["true_mask_palette"].astype("object")
        # df['class_id'] = df['class_id'].astype('int')
        for idx, row in df.iterrows():
            df.at[idx, "true_mask_palette"] = [
                int(row["class_id"]),
                int(row["class_id"]),
                int(row["class_id"]),
            ]

        f = df[["true_mask_palette", "class_id", "common_name"]]
        f.to_csv(self.true_mask_palette_path, index=False)

    def list_common_names(self):
        # meta = self.read_metadata(self.species_info_json)
        # metas = [meta["species"][i] for i in meta["species"].keys()]
        df = pd.read_csv(self.datacsv)
        # df = pd.DataFrame.from_records(metas)
        df = df.drop_duplicates(subset=["class_id"])
        df = df.sort_values("class_id").reset_index(drop=True)
        df["common_name"].to_csv(self.common_names_file, index=False)

    def list_USDA_symbols(self):
        # meta = self.read_metadata(self.species_info_json)
        # metas = [meta["species"][i] for i in meta["species"].keys()]
        df = pd.read_csv(self.datacsv)
        # df = pd.DataFrame.from_records(metas)
        df = df.drop_duplicates(subset=["class_id"])
        df = df.sort_values("class_id").reset_index(drop=True)

        df[["USDA_symbol", "common_name"]].to_csv(self.USDA_symbols_file, index=False)

    def class_distribution(self):
        df = pd.read_csv(self.datacsv)
        cn_dist = (
            df.groupby(["common_name"])["image_id"].count().sort_values(ascending=False)
        )
        batch_dist = (
            df.groupby(["batch_id"])["image_id"].count().sort_values(ascending=False)
        )

        cn_dist.to_csv(self.common_name_distribution)
        batch_dist.to_csv(self.batch_distribution)

    def categories_json(self):
        meta = self.read_metadata(self.species_info_json)
        metas = [meta["species"][i] for i in meta["species"].keys()]
        df = pd.DataFrame.from_records(metas)
        df["id"] = df["class_id"]
        df["name"] = df["common_name"]
        df["supercategory"] = df["group"]
        f = df[["id", "name", "supercategory"]]
        fdict = f.to_dict(orient="records")
        with open(self.coco_categories, "w") as fp:
            json.dump(fdict, fp)

    def binary_classes(self):
        meta = self.read_metadata(self.species_info_json)
        metas = [meta["species"][i] for i in meta["species"].keys()]
        df = pd.DataFrame.from_records(metas)
        mdf = df[["class_id", "group"]]
        mdf["binary_class"] = np.where(df["group"] == "monocot", 1, 2)
        bindf = mdf[["class_id", "binary_class"]]
        bindict = dict(zip(bindf["class_id"], bindf["binary_class"]))
        # bindict = bindf.to_dict(orient='records')
        print(bindict)


def main(cfg: DictConfig) -> None:
    # Create synth data container that holds all cutouts, backgrounds, and pots
    log.info("Writing class information.")
    pc = PresentClasses(cfg)
    # pc.categories_json()
    pc.present_classes_csv()
    # pc.all_classes_csv()
    # pc.true_mask_palette()
    # pc.list_common_names()
    # pc.list_USDA_symbols()
    # pc.class_distribution()
    # pc.binary_classes()
