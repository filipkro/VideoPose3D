# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from common.custom_dataset import CustomDataset
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

args = parse_args()
print(args)


print('Loading dataset...')
# dataset_path = 'data/data_3d_' + args.dataset + '.npz'
# dataset_path = '/home/filipkr/Documents/xjob/data_2d_custom_lol-take2.npz'
# pose_path = '/home/filipkr/Documents/xjob/custom_2d_training.npz'

dataset = CustomDataset(args.lol_path)


print('Loading 2D detections...')
# keypoints_metadata = np.load(dataset_path,
#                              allow_pickle=True)['metadata'].item()
keypoints_metadata = np.load(args.lol_path,
                             allow_pickle=True)['metadata'].item()
# keypoints = np.load('data/data_2d_' + args.dataset + '_' +
#                     args.keypoints + '.npz', allow_pickle=True)
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(
    dataset.skeleton().joints_right())


sets = {'train': {'sub': ['02', '05' '10', '12', '20'], 'act': ['SLS1R']},
        'test': {'sub': ['18'], 'act': ['SLS1R']}}


big_data = np.load(args.big_data, allow_pickle=True)

data = big_data['data'].item()

# fix 2d coordinates:

fix_one_of = ['10', '12', '20']

for subject in data.keys():
    for action in data[subject].keys():
        poses = data[subject][action]['positions_2d'][0]
        if subject in fix_one_of:
            poses3d = data[subject][action]['positions_3d']
            data[subject][action]['positions_3d'] = poses3d[:-1, ...]

        dx = np.max(poses[:, :, 0]) - np.min(poses[:, :, 0])
        dy = np.max(poses[:, :, 1]) - np.min(poses[:, :, 1])
        norm_dist = np.max((dx, dy))
        print(norm_dist)
        poses = 2 * poses / norm_dist - 1

        shift_y = (np.max(poses[:, :, 1]) + np.min(poses[:, :, 1])) / 2
        poses[..., 1] = poses[..., 1] - shift_y
        shift_x = (np.max(poses[:, :, 0]) + np.min(poses[:, :, 0])) / 2
        poses[..., 0] = poses[..., 0] - shift_x

        # fix thi, 'cause stupid now....

        data[subject][action]['positions_2d'] = poses


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        print(subject)
        if subject in data:
            for action in data[subject].keys():
                print(action)
                if action_filter is None or action in action_filter:

                    if ('positions_2d' in data[subject][action] and
                            'positions_3d' in data[subject][action]):
                        # print(data[subject][action]['positions_2d'])
                        # print(data[subject][action]['positions_3d'])
                        out_poses_2d.append(data[subject]
                                            [action]['positions_2d'])
                        out_poses_3d.append(data[subject]
                                            [action]['positions_3d'])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    return out_camera_params, out_poses_3d, out_poses_2d


cameras_valid, poses_valid, poses_valid_2d = fetch(sets['test']['sub'],
                                                   sets['test']['act'])

outjoints = 9

filter_widths = [int(x) for x in args.architecture.split(',')]

outjoints = 9
checkpoint = None
checkpoint = torch.load(args.chck_load,
                        map_location=lambda storage, loc: storage)
checkpoint['model_pos']['shrink.weight'] = torch.tensor(
    np.ones((outjoints * 3, 1024, 1)))
checkpoint['model_pos']['shrink.bias'] = torch.tensor(
    np.ones((outjoints * 3)))

model_pos_train = TemporalModelOptimized1f(17, 2, outjoints,
                                           filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels)
model_pos_train.load_state_dict(checkpoint['model_pos'])
print('loaded')


model_pos = TemporalModel(17, 2, outjoints, filter_widths=filter_widths,
                          causal=args.causal, dropout=args.dropout,
                          channels=args.channels, dense=args.dense)
model_pos.load_state_dict(checkpoint['model_pos'])
# model_pos_train.load_state_dict(loaded_weights['model_pos'])

receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2  # Padding on each side
print('PADDING', pad)
print('CAUSAL??', args.causal)
if args.causal:
    print('INFO: Using causal convolutions')
    causal_shift = pad
else:
    causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

for param in model_pos_train.layers_conv.parameters():
    param.requires_grad = False

for param in model_pos_train.expand_conv.parameters():
    param.requires_grad = False

for param in model_pos_train.layers_bn.parameters():
    param.requires_grad = False

for param in model_pos_train.expand_bn.parameters():
    param.requires_grad = False

model_params = 0
for parameter in model_pos_train.parameters():
    if parameter.requires_grad:
        model_params += parameter.numel()

print('Trainable after disabling first layers {}'.format(model_params))
print('MODEL', model_pos)

if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()


model_traj = None

test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

if not args.evaluate:
    print(sets)
    cameras_train, poses_train, poses_train_2d = fetch(
        sets['train']['sub'], sets['train']['act'], subset=args.subset)

    lr = args.learning_rate

    optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)

    lr_decay = args.lr_decay

    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    print('3d', poses_train)
    print('2d', poses_train_2d)
    print('3d shape', np.shape(poses_train))

    print('2d shape', np.shape(poses_train_2d))

    print(poses_train)
    train_generator = ChunkedGenerator(args.batch_size // args.stride, cameras_train, poses_train, poses_train_2d, args.stride,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    train_generator_eval = UnchunkedGenerator(cameras_train, poses_train, poses_train_2d,
                                              pad=pad, causal_shift=causal_shift, augment=False)
    print('INFO: Training on {} frames'.format(
        train_generator_eval.num_frames()))

    # if args.resume or checkpoint != None:
    #     epoch = checkpoint['epoch']
    #     if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         train_generator.set_random_state(checkpoint['random_state'])
    #     else:
    #         print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
    #
    #     lr = checkpoint['lr']

    print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
    print('** The final evaluation will be carried out after the last training epoch.')
    print(epoch)
    # Pos model only
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_traj_train = 0
        epoch_loss_2d_train_unlabeled = 0
        N = 0
        model_pos_train.train()
        print('batch', args.batch_size)
        # Regular supervised scenario
        for _, batch_3d, batch_2d in train_generator.next_epoch():
            print('N:', N)

            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
            inputs_3d[:, :, 0] = 0

            optimizer.zero_grad()

            # print(inputs_2d)
            # print(inputs_2d.size())

            # Predict 3D poses
            predicted_3d_pos = model_pos_train(inputs_2d)
            # print('predicted', predicted_3d_pos)
            # print('inputys_3d', inputs_3d)
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_train += inputs_3d.shape[0] * \
                inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            loss_total = loss_3d_pos
            loss_total.backward()

            optimizer.step()

        losses_3d_train.append(epoch_loss_3d_train / N)

        # End-of-epoch evaluation
        with torch.no_grad():
            model_pos.load_state_dict(model_pos_train.state_dict())
            model_pos.eval()

            epoch_loss_3d_valid = 0
            epoch_loss_traj_valid = 0
            epoch_loss_2d_valid = 0
            N = 0

            if not args.no_eval:
                # Evaluate on test set
                for cam, batch, batch_2d in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    inputs_traj = inputs_3d[:, :, :1].clone()
                    inputs_3d[:, :, 0] = 0

                    # Predict 3D poses
                    # print('2d size', inputs_2d.size())
                    # print(inputs_3d.size())

                    predicted_3d_pos = model_pos(inputs_2d)
                    # print(predicted_3d_pos)
                    # print(inputs_3d)
                    # print(predicted_3d_pos.size())
                    # print(inputs_3d.size())
                    inputs_3d = inputs_3d[:, :-1, :, :]
                    # print(inputs_3d.size())

                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_valid += inputs_3d.shape[0] * \
                        inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                N = 1 if N == 0 else N
                losses_3d_valid.append(epoch_loss_3d_valid / N)

                # Evaluate on training set, this time in evaluation mode
                epoch_loss_3d_train_eval = 0
                epoch_loss_traj_train_eval = 0
                epoch_loss_2d_train_labeled_eval = 0
                N = 0
                for cam, batch, batch_2d in train_generator_eval.next_epoch():
                    if batch_2d.shape[1] == 0:
                        # This can only happen when downsampling the dataset
                        continue

                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    inputs_traj = inputs_3d[:, :, :1].clone()
                    inputs_3d[:, :, 0] = 0

                    # Compute 3D poses
                    predicted_3d_pos = model_pos(inputs_2d)
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_train_eval += inputs_3d.shape[0] * \
                        inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                N = 1 if N == 0 else N
                losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)

            # Evaluate 2D loss on unlabeled training set (in evaluation mode)

        elapsed = (time() - start_time) / 60

        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000))
        else:
            print('[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000,
                losses_3d_train_eval[-1] * 1000,
                losses_3d_valid[-1] * 1000))

            # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        # Decay BatchNorm momentum
        momentum = initial_momentum * \
            np.exp(-epoch / args.epochs *
                   np.log(initial_momentum / final_momentum))
        model_pos_train.set_bn_momentum(momentum)

        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            # chk_path = os.path.join(
            #     args.checkpoint, 'epoch_{}.bin'.format(epoch))
            chk_path = os.path.join(
                args.save_folder, 'epoch_{}.bin'.format(epoch))

            print(chk_path)
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                'model_traj': None,
                'random_state_semi': None,
            }, chk_path)

            print('Saving checkpoint to', chk_path)
            if epoch % (3 * args.checkpoint_frequency) == 0:
                chk_path = os.path.join(
                    args.save_home, 'home-epoch_{}.bin'.format(epoch))

                torch.save({
                    'epoch': epoch,
                    'lr': lr,
                    'random_state': train_generator.random_state(),
                    'optimizer': optimizer.state_dict(),
                    'model_pos': model_pos_train.state_dict(),
                    'model_traj': None,
                    'random_state_semi': None,
                }, chk_path)

        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch > 3:
            if 'matplotlib' not in sys.modules:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(args.save_folder, 'loss_3d.png'))

            plt.close('all')
