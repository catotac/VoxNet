import os
import numpy as np

from code import dataparser
from code import cnn_model

# Change to current directory, so that we can use relative paths
os.chdir(os.path.dirname(os.path.realpath(__file__)))

#
# Data parsing and augmentation example
#
# Keyword arguments:
#   augment(bool): enables/disables data augmentation
#   level(int): data augmentation level, lower the number higher the augmentation
#   ops(func): Optional. List of functions for data augmentation. By default, it applies all possible methods.
#       Possible functions for 'ops' keyword argument:
#           * dataparser.flip_xy
#           * dataparser.flip_xz
#           * dataparser.flip_yz
#       Usage: ops=[dataparser.flip_xy, dataparser.flip_yz]
#
#
# Sample data parsing and augmentation code:
# >> data = dataparser.read("data/airplane", "train", 32, augment=True, level=1)
#
# If level = 1, len(data) = 2504
# If level = 2, len(data) = 1565
# If level = 3, len(data) = 1253
#

# 3D ConvNet


# 3D ConvNet with data augmentation
# cnn_model.trainmodel("data", 64)

pass
