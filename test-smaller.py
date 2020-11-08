import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random

architecture = '3,3,3,3,3'
filter_widths = [int(x) for x in architecture.split(',')]
out_joints = 6
model_pos_train = TemporalModelOptimized1f(17, 2, out_joints, filter_widths)
chk_filename = 'checkpoint/pretrained_h36m_detectron_coco.bin'
checkpoint = torch.load(
    chk_filename, map_location=lambda storage, loc: storage)

print(model_pos_train)
# print(checkpoint)
checkpoint['model_pos']['shrink.weight'] = torch.tensor(
    np.ones((out_joints * 3, 1024, 1)))
checkpoint['model_pos']['shrink.bias'] = torch.tensor(
    np.ones((out_joints * 3)))

model_pos_train.load_state_dict(checkpoint['model_pos'])
print('loaded')
