# SSD300

The base code for this project is a Single Shot Detector (SSD) implementation based on the original authors.

## Install
Follow the installation instructions from previous assignments.
Then, install specific packages with

```
pip install -r requirements.txt
```

### Macos arm

If you are experiencing trouble when installing pycocotools, then the following command should work on Mac m1:

```
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' LDFLAGS='-L/usr/local/lib -lomp' pip install pycocotools==2.0.0  
```

## Code structure

The project structure can be seen below. Some of the directories and their contents are shortly explained below.

 * [configs](#configs)
 * [dataset_exploration](#dataset-exploration)
 * [notebooks](#notebooks)
 * [performance_assessment](#performance-assessment)
 * [scripts](#scripts)
 * [ssd](#ssd)
 * [tops](#tops)
 * [tutorials](#tutorials)
 * [other](#other)
    * [Benchmarking the data loader](#benchmarking-the-data-loader)
    * [Uploading results to the leaderboard](#uploading-results-to-the-leaderboard)
    * [Analayzing inference speed](#evaluating-inference-speed-(Quantitative-analysis))
* [Useful commands](#useful-commands)

The datasets are kept in ´data´ and ´datasets´.

## configs

This directory contains configs for anchors and models, as well as some files and utils for loading datasets. This is to easily maintain changes to different models and parameters as the project progresses.

### models

Model configs are created by creating a python file and instantiating the parameters using the `LazyCall` class. E.g. for the `backbone` of the network this can be done as follows:

```
from tops.config import LazyCall as L

backbone = L(backbones.BasicModel)(
    output_channels=[128, 256, 128, 128, 64, 64],
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)
```

Other parameters that can be set include `train`, `anchors`, `model`, `optimizer`, `schedulers`, `data_train`, `data_val`, `label_map`, and so on.

All of these parameters can also by adjusted in their own config files, see `test_anchors/base.py` or `change_lr.py` for more info.

### utils.py

Contains methods for getting dataset directory, `get_dataset_dir()`, and output directory, `get_output_dir()`.

## dataset exploration

This directory contains scripts and modules for exploratory data analysis. For instance, `dataset_statistics.py` contains methods for calculating certain statistics for a dataset, while `save_images_with_annotations` allows visualizing a subset of the dataset with annotations.

To run the scripts, do the following command from the SSD folder:

```
python -m dataset_exploration.dataset_statistics
```

Or to visualize images:

```
python -m dataset_exploration.save_images_with_annotations
```

By default, the script will print the 500 first train images in the dataset, but it is possible to change this by changing the parameters in the `main` function in the script.

## performance assessment

This folder contains scripts for assessing the qualitative performance of the different models.

### Qualitative performance assesment

To check how the model is performing on real images, check out the `performance assessment` folder. Run the test script by doing:

```
python -m performance_assessment.save_comparison_images <config_file>
```

If you for example want to use the config file `configs/tdt4265.py`, the command becomes:

```
python -m performance_assessment.save_comparison_images configs/tdt4265.py
```

This script comes with several extra flags. If you for example want to check the output on the 500 first train images, you can run:

```
python -m performance_assessment.save_comparison_images configs/tdt4265.py --train -n 1000
```

### Test on video

You can run your code on video with the following script:
```
python -m performance_assessment.demo_video configs/tdt4265.py input_path output_path
```
Example:
```
python3 -m performance_assessment.demo_video configs/tdt4265.py Video00010_combined.avi output.avi
```
You can download the validation videos from [OneDrive](https://studntnu-my.sharepoint.com/:f:/g/personal/haakohu_ntnu_no/EhTbLF7OIrZHuUAc2FWAxYoBpFJxfuMoLVxyo519fcSTlw?e=ujXUU7).
These are the videos that are used in the current TDT4265 validation dataset.

## scripts

Mainly contains code for using the updated tdt4265 dataset after annotations was added on April 20th, 2022. The extended dataset will be separated from the original one in your local data folder.

## ssd

sd

## tops

Model checkpointer and configs for creating lazy objects and instantiating them.

## tutorials

Contains tutorials for annotating videos, setting up the dataset, additional environment requirements, and how to use tensorboards.

- [Introduction to code](notebooks/code_introduction.ipynb).
- [Dataset setup](tutorials/dataset_setup.md) (Not required for TDT4265 computers).
- [Running tensorboard to visualize graphs](tutorials/tensorboard.md).

## Other

### Benchmarking the data loader
The file `benchmark_data_loading.py` will automatically load your training dataset and benchmark how fast it is. At the end, it will print out the number of images per second.

```
python benchmark_data_loading.py configs/tdt4265.py
```

### Uploading results to the leaderboard
Run the file:

```
python save_validation_results.py configs/tdt4265.py results.json
```

Remember to change the configuration file to the correct config.
The script will save a .json file to the second argument (results.json in this case), which you can upload to the leaderboard server.


### Evaluating inference speed (Quantitative analysis)

The file `runtime_analysis` lets you analyze the inference speed of models. This can be run by typing:

```
python3 runtime_analysis.py <config-file>
```

## Useful commands

#### Training and evaluation
To start training:
```
python train.py  configs/ssd300.py
```

To starting training VGG on VOC:
```
python train.py  configs/voc_vgg.py
```

To only run evaluation:
```
python train.py  configs/ssd300.py --evaluate-only
```

#### Demo.py
For VOC:
```
python demo.py configs/voc_vgg.py demo/voc demo/voc_output
```

For MNIST:
```
python demo.py configs/ssd300.py demo/mnist demo/mnist_output
```
