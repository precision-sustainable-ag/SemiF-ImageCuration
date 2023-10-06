# SemiF-ImageCuration

## Overview

SemiF-ImageCuration is a tool designed for curating a subset of images from the National Agricultural Image Repository. It automates the process of selecting, filtering, and organizing large sets of plant images collected under semi-field nursery conditions, and is configured using Facebook's Hydra configuration tool.


## Installation

To install dependencies, ensure you have Conda installed and then create a new environment with the provided environment.yaml file:

```shell

conda env create -f environment.yaml
```


## Tasks
### 1. config_data

This task generates a CSV file containing metadata of selected images based on the filtering criteria specified in the configuration files, including species, green sum, area, and more.

### 2. create_dataset

After obtaining the curated CSV from config_data, this task retrieves the actual images and related data, organizing them into structured directories as per the project's needs.

### 3. crop_images

This task handles the operation of cropping the selected images and their respective masks. The images and masks are split into tiles, and each tile is saved separately, ensuring efficient and manageable datasets. This task operates on full-resolution images, not sub-images.

## Configuration

Configuration files in the conf/ directory allow users to customize the image selection criteria, filtering options, and other settings for both tasks. Use the cutout.yaml to specify image attributes and filtering parameters.

### Configuration for config_data

The config_data task is designed to produce a curated CSV file of metadata, pulled from json files, based on your specific filtering and selection criteria. Configuration details are as follows:

#### Species Filtering

Choose a specific list of plant species based on USDA symbols. Uncommenting a species includes it in the filtering process. The common names have been provided as comments next to the USDA symbols. Keep common names commented.

```yaml
species: 
  - AMPA    # Palmer amaranth
  # - AMAR2   # Common ragweed
  # ... [other species]
```

#### Primary Image Selection

Determine if only primary images should be included. If no preference, set it to None.
```yaml
is_primary: True # must use None if none
```

#### Image Border Extension

Configure the inclusion of images based on whether the cutout extends beyond the image frame. For no specific preference, set it to None.

```yaml
extends_border: None # must use None if none
```
#### Green Sum Filtering

Establish minimum and maximum thresholds for the green sum to filter images.

```yaml
green_sum:
  max: 1000000000
  min: 1000
```

### Image Area Filtering

Filter images based on their area by setting values for the minimum and maximum. You must use either mean, 25, 50, or 75 to represent percentiles.

```yaml
area: 
  max: 
  min: 25
```

#### Save CSV

Decide whether to generate and save a CSV file of all cutout metadata.

```yaml
save_csv: True 
```

#### Dataset Type

Specify whether to target "images" or "cutouts" for the dataset creation.

### Configuring Image and Cutout Cropping

For the crop_images task, use the crop_split section of the configuration file to define the specifics of how images and cutouts are cropped:

```yaml
crop_split: #for both images and cutouts
  M: 640  # Tile height
  N: 640  # Tile width
  exclude_empty_tiles: True  # Flag to exclude tiles without any content
```
Simply adjust the M and N values for tile dimensions and set exclude_empty_tiles according to whether you want to exclude tiles with empty masks.

## Execution

To execute a task, run the main script from the command line and specify the desired task in the configuration file:

```bash
python CURATEDATA.py general.task=config_data general.project_name=test_palmer
```

Replace config_data with create_dataset to execute the second task and test_palmer with your actual project name.

## Directory Structure

The tool creates a structured directory layout for organized data retrieval and storage:

* `data/projects/{project_name}`: Contains the main CSV file with metadata of selected images, a copy of the Hydra configs, and the log file.
* `data/projects/{project_name}/results`: Contains the "cutouts", "images", "meta_masks", and "metadata" directories housing the actual image data and related metadata.

## File Output

* `{project_name}.csv`: Main CSV containing metadata of selected images.

For detailed configurations and customizations, refer to the provided example in `data/projects/test_palmer` and update them as per your specific requirements.