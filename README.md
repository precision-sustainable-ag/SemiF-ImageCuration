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

## Configuration

Configuration files in the conf/ directory allow users to customize the image selection criteria, filtering options, and other settings for both tasks. Use the cutout.yaml to specify image attributes and filtering parameters.

## Execution

To execute a task, run the main script from the command line and specify the desired task in the configuration file:

```bash
python CURATEDATA.py general.task=config_data general.project_name=test_palmer
```

Replace config_data with create_dataset to execute the second task and my_project with your actual project name.

## Directory Structure

The tool creates a structured directory layout for organized data retrieval and storage:

* `data/projects/{project_name}`: Contains the main CSV file with metadata of selected images, a copy of the Hydra configs, and the log file.
* `data/projects/{project_name}/results`: Contains the "cutouts", "images", "meta_masks", and "metadata" directories housing the actual image data and related metadata.

## File Output

* `{project_name}.csv`: Main CSV containing metadata of selected images.

For detailed configurations and customizations, refer to the provided example in `data/projects/test_palmer` and update them as per your specific requirements.