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
from ssd.data.transforms import (
    PhotometricDistort
)
from tops.config import LazyCall as L
from .utils import get_dataset_dir
import torchvision


# New transformations added:
# * Random horizontal flip
# * Random sample crop
# * Photometric distort
train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomSampleCrop)(),
    L(PhotometricDistort)(),
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

data_train.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022"),
    transform="${train_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022/train_annotations.json")
)

data_val.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022"),
    transform="${val_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022/val_annotations.json")
)

data_val.gpu_transform = gpu_transform
data_train.gpu_transform = gpu_transform