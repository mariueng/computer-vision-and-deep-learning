from .task2_3_1 import (
    train,
    optimizer,
    schedulers,
    model, 
    data_train,
    data_val,
    val_cpu_transform,
    train_cpu_transform, 
    gpu_transform,
    label_map,
    backbone,
    anchors,
    loss_objective
)
from tops.config import LazyCall as L
from ssd.modeling import SSD300DCIWI


model = L(SSD300DCIWI)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes= 8 + 1  # Add 1 for background
)