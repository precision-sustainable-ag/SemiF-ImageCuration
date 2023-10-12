import json
import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from utils.datasets import Cutout

log = logging.getLogger(__name__)


class ConfigData:
    """
    A configuration-driven class used for curating images from the National Agricultural Image Repository.

    This class is built around the Hydra configuration tool, which allows users to specify image curation
    filters and properties through a structured configuration. The class reads cutouts, and applies a series
    of filters based on the properties specified in the configuration. These filters include filtering by
    species, green sum, cutout area, border extensions, and primary attribute.

    After filtering, the class can generate distributions based on common name and batch id. The results can
    be saved into CSV files for further use.

    Attributes:
    -----------
    cfg : DictConfig
        The Hydra configuration containing all the filtering and processing parameters.

    Methods:
    --------
    get_cutout_meta(path: str) -> Cutout:
        Retrieves cutout metadata from a given path.
    read_cutouts() -> DataFrame:
        Reads cutouts from the specified directory and returns them as a dataframe.
    get_images_count(df: DataFrame) -> None:
        Logs the count of unique images from the dataframe.
    filter_by_species(df: DataFrame) -> DataFrame:
        Applies species-based filtering on the dataframe.
    filter_by_green_sum(df: DataFrame) -> DataFrame:
        Filters the dataframe based on green sum criteria.
    filter_by_area(df: DataFrame) -> DataFrame:
        Filters the dataframe based on the area criteria.
    filter_by_properties(df: DataFrame) -> DataFrame:
        Filters the dataframe based on properties like border extension and primary attribute.
    sort_cutouts() -> DataFrame:
        Applies all the filters in sequence and returns the resultant dataframe.
    class_distribution() -> None:
        Calculates and saves the distribution based on common name and batch id.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.cutoutdir = cfg.data.cutoutdir
        self.species = cfg.cutouts.species

        # Main filters
        self.green_sum = cfg.cutouts.green_sum
        self.area = cfg.cutouts.area
        self.extends_border = cfg.cutouts.extends_border
        self.is_primary = cfg.cutouts.is_primary
        self.common_name_distribution = cfg.data.common_name_distribution
        self.batch_distribution = cfg.data.batch_distribution
        self.uniform_subsample = cfg.cutouts.uniform_subsample
        self.subsample_by_species = cfg.cutouts.subsample_by_species

        self.df = self.sort_cutouts()

    def get_cutout_meta(self, path):
        with open(path) as f:
            return Cutout(**json.load(f))

    def read_cutouts(self):
        batch_pref = ("MD", "TX", "NC")
        cutout_batches = [
            x for x in Path(self.cutoutdir).glob("*") if x.name.startswith(batch_pref)
        ]
        cutout_csvs = [x for batch in cutout_batches for x in batch.glob("*.csv")]

        return pd.concat(
            [pd.read_csv(x, low_memory=False) for x in cutout_csvs]
        ).reset_index(drop=True)

    def get_images_count(self, df):
        df = df.drop_duplicates(subset="image_id", keep="first")
        log.info(f"{len(df)} Images after removing duplicates")

    def filter_by_species(self, df):
        if self.species:
            df = df[df.USDA_symbol.isin(self.species)]
            log.info(f"{len(df)} cutouts after filter by species ({self.species})")
        return df

    def filter_by_green_sum(self, df):
        if self.green_sum:
            gsmax, gsmin = self.green_sum.max, self.green_sum.min
            df = df[(df.green_sum <= gsmax) & (df.green_sum >= gsmin)]
            log.info(f"{len(df)} cutouts after filter by green_sum ({self.green_sum})")
        return df

    def filter_by_area(self, df):
        if self.area:
            desc = df["area"].describe()
            bounds = {
                "mean": desc.iloc[1],
                25: desc.iloc[4],
                50: desc.iloc[5],
                75: desc.iloc[6],
            }
            min_bound, max_bound = bounds.get(self.area.min), bounds.get(self.area.max)

            if min_bound is not None:
                df = df[df["area"] > min_bound]
            if max_bound is not None:
                df = df[df["area"] < max_bound]

            log.info(f"{len(df)} cutouts after filter by area ({self.area})")

        return df

    def filter_by_properties(self, df):
        if self.extends_border != "None":
            df = df[df.extends_border == self.extends_border]
            log.info(
                f"{len(df)} cutouts after filter by extend border ({self.extends_border})"
            )
        if self.is_primary != "None":
            df = df[df["is_primary"] == self.is_primary]
            log.info(
                f"{len(df)} cutouts after filter by is_primary ({self.is_primary})"
            )
        return df

    def create_subsample_by_species(self, df):
        """
        Creates a subsample of the data based on the number of images specified for each class in the configuration.

        Returns:
        --------
        subsampled_df : DataFrame
            A dataframe containing the subsample of images.
        """
        assert not (
            self.uniform_subsample.status and self.subsample_by_species.status
        ), "Either specify 'uniform_subsample' or 'subsample_by_species', but not both."

        if not self.subsample_by_species.status:
            return df

        species_counts = self.subsample_by_species.species_counts

        # Create an empty dataframe to collect subsamples
        subsampled_dfs = []
        replace = self.subsample_by_species.replace
        for usda_symbol, count in species_counts.items():
            if usda_symbol not in df["USDA_symbol"].unique():
                log.warning(
                    f"Species USDA_symbol {usda_symbol} not found in configured df. Check cutouts.species in config. Skipping."
                )
                continue
            # Filter rows corresponding to the current common name
            filtered = df[df["USDA_symbol"] == usda_symbol]
            # If counts is specified and larger than the available data without replacement, adjust it
            if count is float and count <= 1:
                frac = count
                count = None
            else:
                frac = None
                if count and not replace and len(filtered) < count:
                    log.warning(
                        f"Sample size 'counts'({count}) smaller than population size {len(filtered)} for {usda_symbol}. Using population size as counts."
                    )
                    count = len(filtered)
            # Sample the required number of rows for the current common name
            subsample = filtered.sample(
                n=min(count, len(filtered)),
                frac=frac,
                random_state=self.subsample_by_species.random_state,
                replace=False,
            )

            # Append the subsample to the collection dataframe
            subsampled_dfs.append(subsample)

        subsampled_df = pd.concat(subsampled_dfs)
        log.info(f"Subsampled by species dataframe has {len(subsampled_df)} rows.")

        return subsampled_df

    def create_uniform_subsample(self, df):
        """
        Creates a subsample of the data based on the number of n_counts images specified in the configuration.

        Returns:
        --------
        subsampled_df : DataFrame
            A dataframe containing the subsample of images.
        """
        assert not (
            self.uniform_subsample.status and self.subsample_by_species.status
        ), "Either specify 'uniform_subsample' or 'subsample_by_species', but not both."

        if not self.uniform_subsample.status:
            return df

        common_names = df.common_name.unique()
        # Create an empty dataframe to collect subsamples
        subsampled_dfs = []

        n_counts = self.uniform_subsample.n_counts
        for common_name in common_names:
            # Filter rows corresponding to the current common name
            filtered = df[df["common_name"] == common_name]

            # Configure N and replace for sampling
            replace = self.uniform_subsample.replace
            if n_counts:
                if not replace and len(filtered) < n_counts:
                    log.warning(
                        f"Sample size 'n_counts'({n_counts}) smaller than population size {len(filtered)} for {common_name}. Using population size as n_counts."
                    )
                    n_counts = len(filtered)

            frac = self.uniform_subsample.frac if n_counts is None else None

            # Sample the required number of rows for the current common name
            subsample = filtered.sample(
                n=n_counts,
                frac=frac,
                replace=replace,
                random_state=self.uniform_subsample.random_state,
            )

            # Append the subsample to the collection dataframe
            subsampled_dfs.append(subsample)

        subsampled_df = pd.concat(subsampled_dfs)

        log.info(f"Uniformly subsampled dataframe has {len(subsampled_df)} rows.")

        return subsampled_df

    def sort_cutouts(self):
        df = self.read_cutouts()
        # Filtering
        df = self.filter_by_species(df)
        df = self.filter_by_green_sum(df)
        df = self.filter_by_area(df)
        df = self.filter_by_properties(df)
        # Subsampling
        df = self.create_uniform_subsample(df)
        df = self.create_subsample_by_species(df)

        if len(df) == 0:
            log.error("No cutouts. Exiting.")
            exit(1)

        self.get_images_count(df)
        return df.reset_index(drop=True)

    def class_distribution(self):
        cn_dist = (
            self.df.groupby(["common_name"])["image_id"]
            .count()
            .sort_values(ascending=False)
        )
        cn_dist.name = "count"

        batch_dist = (
            self.df.groupby(["batch_id"])["image_id"]
            .count()
            .sort_values(ascending=False)
        )
        batch_dist.name = "count"

        cn_dist.to_csv(self.common_name_distribution)
        batch_dist.to_csv(self.batch_distribution)
