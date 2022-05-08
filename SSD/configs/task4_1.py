from .task2_5 import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)

from tops.config import LazyCall as L
from ssd.modeling import RetinaNet
from ssd.modeling.backbones import BiFPN

# backbone = L(BiFPN)(
#     input_channels=[256, 512, 1024, 2048, 256, 256],
#     output_channels=[64, 64, 64, 64, 64, 64],
#     output_feature_sizes="${anchors.feature_sizes}",
# )

backbone = L(BiFPN)(
    input_channels=[256, 512, 1024, 2048, 512, 256], #[64, 128, 256, 512, 256, 64],
    feature_size=64,
    output_feature_sizes="${anchors.feature_sizes}"
)

model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes= 8 + 1,
    improved_weight_init = True
)