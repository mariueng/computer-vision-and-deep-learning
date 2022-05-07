import torch

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
    anchors
)
from tops.config import LazyCall as L
from ssd.modeling import SSD300, FocalLoss

loss_objective = L(FocalLoss)(
    anchors="${anchors}",
    alpha=torch.tensor([[0.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
    gamma=0.25,
)

model = L(SSD300)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes= 8 + 1  # Add 1 for background
)