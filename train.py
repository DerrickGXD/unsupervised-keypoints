import torch
import torch.nn as nn
from torch.nn.functional import softmax
import math
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from torch.utils.tensorboard import SummaryWriter
import lpips
from torch import optim
import copy
from absl import app
from absl import flags
from TrainDataset import channels_to_frame, frame_to_channels
from data import ytvis_final as yt_final
from data import tigdog_final as tf_final
from Models import UNet, UNet_Reconstruct, xy_outputs, get_2d_gaussian
from torch.utils.tensorboard import SummaryWriter
import torchvision
import utils
from data import frame_loader as tigdog_mf
from torch.utils.data import DataLoader
import os.path as osp
import os
import pickle as pkl

flags.DEFINE_integer('epochs', 20, '')
flags.DEFINE_integer('num_kps', 8, '')
flags.DEFINE_integer('vis_every', 50, '')
flags.DEFINE_integer('num_frames', 2, '')
flags.DEFINE_integer('batch_size', 2, '')
flags.DEFINE_float('std', 0.1, '')
flags.DEFINE_float('lr', 1e-3, '')
flags.DEFINE_string('exp_name', 'kp_tigers', 'tmp dir to extract dataset')
flags.DEFINE_string('root_dir_yt', '/home/filippos/data/youtube_vis/pkls/', 'tmp dir to extract dataset')
flags.DEFINE_string('root_dir', '/home/filippos/data/TigDog_new_wnrsfm/', 'tmp dir to extract dataset')
flags.DEFINE_string('tmp_dir', 'saved_frames/', 'tmp dir to extract dataset')
flags.DEFINE_string('category', 'tiger', 'tmp dir to extract dataset')
flags.DEFINE_string('tb_log_dir', 'logs/', 'tmp dir to extract dataset')
opts = flags.FLAGS


def change_range(img, to_01=False):
    if to_01:
        img_new = (img +1) / 2
        img_new = img_new.clamp(0, 1)
        return img_new
    else:
        img_new = (img - 0.5) * 2
        img_new = img_new.clamp(-1, 1)
        return img_new


def main(_):
    writer = SummaryWriter(log_dir=opts.tb_log_dir + '/' + opts.exp_name)

    torch.manual_seed(0)
    if opts.category in ['horse', 'tiger']:
        dataset = tf_final.TigDogDataset_Final(opts.root_dir, opts.category, transforms=None, normalize=False,
                                               max_length=None, remove_neck_kp=False, split='train',
                                               img_size=256, mirror=False, scale=False, crop=False)

        collate_fn = tf_final.TigDog_collate

    directory = opts.tmp_dir + '/' + opts.category + '/'
    if not osp.exists(directory):
        os.makedirs(directory)

    save_counter = 0
    sample_to_vid = {}
    samples_per_vid = {}
    print('Number of videos for ', opts.category, '-', len(dataset))
    i_sample = 0
    for i_sample, sample in enumerate(dataset):
        num_frames = sample['video'].shape[0]
        for i in range(num_frames):
            new_sample = {}
            for k in sample.keys():
                if k in ['video', 'sfm_poses', 'landmarks', 'segmentations', 'bboxes']:
                    new_sample[k] = sample[k][i]
            pkl.dump(new_sample, open(directory + str(save_counter) + '.pkl', 'wb'))
            sample_to_vid[save_counter] = i_sample
            if i_sample in samples_per_vid:
                samples_per_vid[i_sample].append(save_counter)
            else:
                samples_per_vid[i_sample] = [save_counter]
            save_counter += 1
        #     if i >= 5:  # 35:  # TODO:fix this
        #         break
        # if i_sample >= 3:  # TODO:fix this
        #     break

    training_samples = save_counter
    print('Training samples (frames):', training_samples)
    dataset = tigdog_mf.TigDogDataset_MultiFrame(opts.tmp_dir, opts.category, num_frames=opts.num_frames,
                                                 sample_to_vid=sample_to_vid,
                                                 samples_per_vid=samples_per_vid,
                                                 normalize=True, transforms=True,
                                                 remove_neck_kp=True, split='train', img_size=256,
                                                 mirror=True, scale=True, crop=True, v2_crop=True, tight_bboxes=True)
    collate_fn = tigdog_mf.TigDog_collate

    dataloader = DataLoader(dataset, opts.batch_size, drop_last=True, shuffle=True,
                            collate_fn=collate_fn, num_workers=2)
    print('Dataloader:', len(dataloader))

    keypoint_model = UNet(opts.num_kps).cuda()
    reconstruct_model = UNet_Reconstruct(3, opts.num_kps).cuda()
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    optimizer = optim.Adam(list(keypoint_model.parameters()) + list(reconstruct_model.parameters()), lr=opts.lr)
    std = opts.std
    n_iter = 0
    for epoch in range(opts.epochs):
        avg_loss = 0
        for sample in dataloader:
            input_img_tensor = sample['img'].type(torch.FloatTensor).clone().cuda()
            frame1 = input_img_tensor[:, 0]
            frame2 = input_img_tensor[:, 1]
            target = frame2
            source = frame1
            target_outputs = keypoint_model(target)
            result_x, result_y = xy_outputs(target_outputs, scaling=True, scale=frame2.shape[-1])
            result_kps = torch.cat([result_x, result_y], dim=1)
            result_kps_vis = torch.stack([result_x, result_y, torch.ones_like(result_y)], dim=-1)
            reconstruct = reconstruct_model(source, result_kps)
            loss = loss_fn_alex.forward(reconstruct, change_range(target)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if n_iter % opts.vis_every == 0:
                kp_img = utils.kp2im(result_kps_vis[0].detach().cpu().numpy(), target[0].cpu().numpy(), radius=2) / 255
                kp_img = torch.from_numpy(kp_img).permute(2, 0, 1)[None]
                kp_img = kp_img.to(source.device)
                grid = torch.cat([source[:1], target[:1], kp_img[:1], change_range(reconstruct[:1], to_01=True)], dim=3)[0]
                writer.add_image('images', grid, n_iter)
            avg_loss += loss.item()
            writer.add_scalar('Loss/train', loss, n_iter)
            n_iter += 1
        avg_loss = avg_loss / len(dataloader)
        print('Epoch ', epoch, ' average loss ', avg_loss)
    writer.close()


if __name__ == '__main__':
    app.run(main)
