from common.custom_dataset import CustomDataset
import numpy as np

# from common.arguments import parse_args
import torch
from argparse import ArgumentParser

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


def evaluate(test_generator, model_pos, action=None, return_predictions=False, use_trajectory_model=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():

        model_pos.eval()

        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
            print(inputs_2d.size())
            predicted_3d_pos = model_pos(inputs_2d)
            print(predicted_3d_pos.size())

            return predicted_3d_pos.squeeze(0).cpu().numpy()


def fix_input(poses):

    dx = np.max(poses[:, :, 0]) - np.min(poses[:, :, 0])
    dy = np.max(poses[:, :, 1]) - np.min(poses[:, :, 1])
    norm_dist = np.max((dx, dy))
    # print(norm_dist)
    poses = 2 * poses / norm_dist - 1

    shift_y = (np.max(poses[:, :, 1]) + np.min(poses[:, :, 1])) / 2
    poses[..., 1] = poses[..., 1] - shift_y
    shift_x = (np.max(poses[:, :, 0]) + np.min(poses[:, :, 0])) / 2
    poses[..., 0] = poses[..., 0] - shift_x

    return poses


def main(args):
    big_data = np.load('/home/filipkr/Documents/xjob/pose-data/big_data.npz',
                       allow_pickle=True)['data'].item()

    in_data = big_data[args.subject][args.action]['positions_2d']
    print(in_data)
    in_data = fix_input(in_data[0])

    model_pos = TemporalModel(
        17, 2, 9, filter_widths=[3, 3, 3, 3, 3])
    checkpoint = torch.load(args.model_checkpoint,
                            map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'])
    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2

    gen = UnchunkedGenerator(None, None, [in_data], pad=pad, causal_shift=0)

    prediction = evaluate(gen, model_pos, return_predictions=True)
    # predicted_3d_pos = model_pos(inputs_2d)
    print(prediction)
    print(np.shape(prediction))

    np.save(args.save_path, prediction)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model_checkpoint')
    parser.add_argument('subject')
    parser.add_argument('action')
    parser.add_argument('save_path')
    args = parser.parse_args()
    main(args)
