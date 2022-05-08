from .task2_3_4 import (
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
    loss_objective,
)
from tops.config import LazyCall as L
from ssd.modeling import AnchorBoxes

"""

Output from kmeans optimizer:

Recommended aspect ratios: (width/height)
 width     height     ratio     count
 39.32     46.20      1.17    2087.00
  9.68     11.00      1.14    2087.00
 22.81     24.99      1.10    2087.00
 12.72      9.85      0.77    2003.00
 16.00     12.95      0.81    2003.00
 22.63     17.10      0.76    1961.00
 19.09     13.41      0.70    1961.00
124.47     60.21      0.48    1856.00
 69.92     34.91      0.50    1856.00
  5.85      8.31      1.42    992.00
 15.43     21.57      1.40    992.00
 26.00     33.95      1.31    992.00
  5.41     14.23      2.63    729.00
 11.76     17.33      1.47    711.00
  4.18     12.75      3.05    706.00
  8.95     33.95      3.79    671.00
  2.36      8.81      3.73    671.00
 18.00     52.85      2.94    647.00
 13.44     33.31      2.48    592.00
  3.29     11.08      3.37    576.00
  6.51     22.08      3.39    576.00
  4.66     18.56      3.98    505.00
  6.81     13.01      1.91    473.00
  7.09     30.68      4.33    298.00
 10.70     48.26      4.51    298.00

"""

# Improved anchor boxes based on EDA
custom_aspect_ratios = [[1.1, .5], [1., .75, .5], [.75, .5], [.75, .5, 1.4], [3., 1.4, 4], [1.4, 3, 4]]

anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[24, 24], [40, 40], [48, 48], [128, 128], [172, 172], [256, 256], [256, 800]],
    aspect_ratios=custom_aspect_ratios,
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)

