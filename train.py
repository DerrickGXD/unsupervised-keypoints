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
from torchvision.transforms import RandomAffine

flags.DEFINE_integer('epochs', 30, '')
flags.DEFINE_integer('num_kps', 18, '')
flags.DEFINE_integer('vis_every', 2000, '')
flags.DEFINE_integer('num_frames', 2, '')
flags.DEFINE_integer('batch_size', 2, '')
flags.DEFINE_integer('img_size', 128, '')
flags.DEFINE_float('std', 0.1, '')
flags.DEFINE_float('lr', 1e-3, '')
flags.DEFINE_float('wd', 0, '')
flags.DEFINE_float('alpha', 0.01, '')
flags.DEFINE_float('beta', 0.01, '')
flags.DEFINE_string('exp_name', 'kp_tigers', 'tmp dir to extract dataset')
flags.DEFINE_string('root_dir_yt', '/home/xindeik/data/youtube_vis/pkls/', 'tmp dir to extract dataset')
flags.DEFINE_string('root_dir', '/home/xindeik/data/TigDog_new_wnrsfm_new/', 'tmp dir to extract dataset')
flags.DEFINE_string('tmp_dir', 'saved_frames/', 'tmp dir to extract dataset')
flags.DEFINE_string('category', 'tiger', 'tmp dir to extract dataset')
flags.DEFINE_string('tb_log_dir', 'logs_noaffine/', 'tmp dir to extract dataset')
flags.DEFINE_string('model_state_dir','state_dict_model_noaffine.pt', 'tmp dir to save model state')
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


def compute_dt(mask):
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(1-mask)
    return dist


def texture_dt_loss_v(texture_flow, dist_transf, reduce=True):
    V = texture_flow.size(1)
    flow_grid = texture_flow.view(-1, V, 1, 2)
    dist_transf = torch.nn.functional.grid_sample(dist_transf, flow_grid, align_corners=False, padding_mode='border')
    if reduce:
        return dist_transf.mean()
    else:
        dist_transf = dist_transf.mean(-1).mean(-1).squeeze(1)
        return dist_transf


def main(_):
    writer = SummaryWriter(log_dir=opts.tb_log_dir + str(opts.alpha)  + '/' + opts.exp_name)

    torch.manual_seed(0)
    if opts.category in ['horse', 'tiger']:
        dataset = tf_final.TigDogDataset_Final(opts.root_dir, opts.category, transforms=None, normalize=False,
                                               max_length=None, remove_neck_kp=False, split='train',
                                               img_size=opts.img_size, mirror=False, scale=False, crop=False)

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
           # if i >= 5:  # 35:  # TODO:fix this
               # break
        #if i_sample >= 3:  # TODO:fix this
           # break

    training_samples = save_counter
    print('Training samples (frames):', training_samples)
    dataset = tigdog_mf.TigDogDataset_MultiFrame(opts.tmp_dir, opts.category, num_frames=opts.num_frames,
                                                 sample_to_vid=sample_to_vid,
                                                 samples_per_vid=samples_per_vid,
                                                 normalize=True, transforms=True,
                                                 remove_neck_kp=True, split='train', img_size=opts.img_size,
                                                 mirror=True, scale=True, crop=True, v2_crop=True, tight_bboxes=True)
    collate_fn = tigdog_mf.TigDog_collate

    dataloader = DataLoader(dataset, opts.batch_size, drop_last=True, shuffle=True,
                            collate_fn=collate_fn, num_workers=2)
    print('Dataloader:', len(dataloader))

    keypoint_model = UNet(opts.num_kps).cuda()
    reconstruct_model = UNet_Reconstruct(3, opts.num_kps).cuda()
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    optimizer = optim.Adam(list(keypoint_model.parameters()) + list(reconstruct_model.parameters()), lr=opts.lr, weight_decay=opts.wd)
    std = opts.std
    n_iter = 0
    affine = RandomAffine(degrees=5, shear=(0.0,0.5))
    for epoch in range(opts.epochs):
        avg_loss = 0
        for sample in dataloader:
            input_img_tensor = sample['img'].type(torch.FloatTensor).clone().cuda()
            mask_3channels = torch.unsqueeze(sample['mask'], 2)
            mask_3channels = mask_3channels.repeat(1,1,3,1,1).clone().cuda()
            frame1 = input_img_tensor[:, 0] * mask_3channels[:,0]
            frame2 = input_img_tensor[:, 1] * mask_3channels[:,1]
            source = frame1
            target = frame2
            target_outputs = keypoint_model(target)
            result_x, result_y = xy_outputs(target_outputs, scaling=True, scale=16)
            result_kps = torch.cat([result_x, result_y], dim=1)
            result_kps_vis = torch.stack([result_x, result_y, torch.ones_like(result_y)], dim=-1)
            reconstruct = reconstruct_model(source, result_kps)
            target_mask = sample['mask'][:,1]
            mask_edt = np.stack([compute_dt(m) for m in target_mask])
            result_kps_xy = torch.dstack((result_x, result_y))
            edts_barrier = torch.tensor(mask_edt).float().unsqueeze(1).cuda()
            loss_mask = texture_dt_loss_v(result_kps_xy, edts_barrier)
            loss_reconstruction = loss_fn_alex.forward(reconstruct, change_range(target)).mean()
            loss = loss_reconstruction + (opts.alpha * loss_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if n_iter % opts.vis_every == 0:
                kp_img = utils.kp2im(result_kps_vis[0].detach().cpu().numpy(), target[0].cpu().numpy(), radius=2) / 255
                kp_img = torch.from_numpy(kp_img).permute(2, 0, 1)[None]
                kp_img = kp_img.to(source.device)
                kp_mask = utils.kp2im(result_kps_vis[0].detach().cpu().numpy(), mask_3channels[0,1].cpu().numpy())
                kp_mask = torch.from_numpy(kp_mask).permute(2, 0, 1)[None]
                kp_mask = kp_mask.to(source.device)
                grid = torch.cat([source[:1], target[:1], kp_img[:1], kp_mask, change_range(reconstruct[:1], to_01=True)], dim=3)[0]
                writer.add_image('iter {n} of image (reconstruction, mask, loss) = ({r},{m},{l})   '.format(r=loss_reconstruction,m=loss_mask,l=loss,n=str(n_iter)), grid, n_iter)
            avg_loss += loss.item()
            writer.add_scalar('Loss/train std : ' + str(opts.std), loss, n_iter)
            n_iter += 1
        avg_loss = avg_loss / len(dataloader)
        print('Epoch ', epoch, ' average loss ', avg_loss)
    torch.save({
        'keypoint_state_dict' : keypoint_model.state_dict(),
        'reconstruct_state_dict' : reconstruct_model.state_dict()
        }, opts.model_state_dir + str(opts.alpha))
    writer.close()


if __name__ == '__main__':
    app.run(main)
