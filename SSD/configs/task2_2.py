# Inherit configs from task 2.1 (task2_1.py)
from task2_1 import *

from ssd.data.transforms import (
    ToTensor, Resize,
    GroundTruthBoxesToAnchors,
    RandomHorizontalFlip,
    RandomSampleCrop,
    PhotometricDistort
)

# New transformations added:
# * Random horizontal flip
# * Random sample crop
train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomSampleCrop)(),
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
    L(RandomHorizontalFlip)(),
    L(PhotometricDistort)(),
])

data_train.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022"),
    transform="${train_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022/train_annotations.json"))

data_val.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022"),
    transform="${val_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022/val_annotations.json"))
    
data_val.gpu_transform = gpu_transform
data_train.gpu_transform = gpu_transform
