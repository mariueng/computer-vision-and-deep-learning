# Inherit configs from task 2.1 (task2_1.py)
from .task2_1 import (
    train,
    anchors,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    label_map
)
from ssd.data import TDT4265Dataset
from ssd.data.transforms import (
    ToTensor,
    Resize,
    Normalize,
    GroundTruthBoxesToAnchors,
    RandomHorizontalFlip,
    RandomSampleCrop,
)
from tops.config import LazyCall as L
from .utils import get_dataset_dir
import torchvision


# Removed transformations from task2_2.py:
# * Photometric distort

# Notes for future work
# * Would most likely be beneficiary to use more subtle augmentations such as those found in
#   e.g. https://github.com/albumentations-team/albumentations
train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomSampleCrop)(),
    L(ToTensor)(),
    L(RandomHorizontalFlip)(),
    L(Resize)(imshape="${train.imshape}"),
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
])

val_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
])

gpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
])

data_val.gpu_transform = gpu_transform
data_train.gpu_transform = gpu_transform
