from .task2_3_2 import (
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
from ssd.modeling import RetinaNet


# Task 2.3.3, shared conv heads but no improved weight init.
model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes= 8 + 1,
    improved_weight_init = False
)