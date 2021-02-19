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
from torchvision.transforms.functional import affine, _get_inverse_affine_matrix

flags.DEFINE_integer('epochs', 50, '')
flags.DEFINE_integer('num_kps', 18, '')
flags.DEFINE_integer('vis_every', 2000, '')
flags.DEFINE_integer('num_frames', 2, '')
flags.DEFINE_integer('batch_size', 2, '')
flags.DEFINE_integer('img_size', 128, '')
flags.DEFINE_float('std', 0.1, '')
flags.DEFINE_float('lr', 1e-3, '')
flags.DEFINE_float('wd', 0, '')
flags.DEFINE_float('alpha', 0.01, '')
flags.DEFINE_float('beta', 10, '')
flags.DEFINE_string('exp_name', 'kp_tigers', 'tmp dir to extract dataset')
flags.DEFINE_string('root_dir_yt', '/home/xindeik/data/youtube_vis/pkls/', 'tmp dir to extract dataset')
flags.DEFINE_string('root_dir', '/home/xindeik/data/TigDog_new_wnrsfm_new/', 'tmp dir to extract dataset')
flags.DEFINE_string('tmp_dir', 'saved_frames/', 'tmp dir to extract dataset')
flags.DEFINE_string('category', 'tiger', 'tmp dir to extract dataset')
flags.DEFINE_string('tb_log_dir', 'logs_affine_dlr/', 'tmp dir to extract dataset')
flags.DEFINE_string('model_state_dir','state_dict_model_affine_dlr.pt', 'tmp dir to save model state')
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
    loss_mse = torch.nn.MSELoss()
    optimizer = optim.Adam(list(keypoint_model.parameters()) + list(reconstruct_model.parameters()), lr=opts.lr, weight_decay=opts.wd)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=1e-7)
    std = opts.std
    n_iter = 0
    n_batch = 0
    #affine = RandomAffine(degrees=5, shear=(0.0,0.5))
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

            for i in range(4):
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

                rand_angle = np.random.uniform(0,50)
                rand_shear = np.random.uniform(0,50)
                target_affine = affine(target, rand_angle, [0.0,0.0],1.0,[0.0,rand_shear])
                affine_outputs = keypoint_model(target_affine)
                affine_x, affine_y = xy_outputs(affine_outputs, scaling=True, scale=16)
                matrix = _get_inverse_affine_matrix([0.0,0.0],rand_angle,[0.0,0.0],1.0,[0.0,rand_shear])
                transformation_matrix = torch.tensor([[matrix[0],matrix[1],matrix[2]],[matrix[3],matrix[4],matrix[5]],[0,0,1]])
                affine_kps_xy = torch.dstack((affine_x,affine_y))
                true_affine_kps = torch.zeros(opts.batch_size,opts.num_kps,2)
                for batch in range(opts.batch_size):
                    for n in range(opts.num_kps):
                        result_xyz = torch.tensor([result_kps_xy[batch,n,0], result_kps_xy[batch,n,1],1])
                        t = torch.matmul(torch.inverse(transformation_matrix),result_xyz)
                        true_affine_kps[batch,n] = t[:2]

                true_affine_kps_vis = torch.stack([true_affine_kps[:,:,0], true_affine_kps[:,:,1], torch.ones_like(true_affine_kps[:,:,1])], dim=-1)
                pred_affine_kps_vis = torch.stack([affine_x, affine_y, torch.ones_like(affine_y)], dim=-1)
                loss_affine = loss_mse(affine_kps_xy.detach().cpu(), true_affine_kps)
                if(epoch<20):
                    alpha = 0
                else:
                    alpha = opts.alpha
                loss = loss_reconstruction + (alpha * loss_mask) + (opts.beta * loss_affine)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if n_batch % opts.vis_every == 0:
                    kp_img = utils.kp2im(result_kps_vis[0].detach().cpu().numpy(), target[0].cpu().numpy(), radius=2) / 255
                    kp_img = torch.from_numpy(kp_img).permute(2, 0, 1)[None]
                    kp_img = kp_img.to(source.device)
                    kp_affine = utils.kp2im(true_affine_kps_vis[0].detach().cpu().numpy(), target_affine[0].cpu().numpy(), radius=2) / 255
                    kp_affine = torch.from_numpy(kp_affine).permute(2, 0, 1)[None]
                    kp_affine = kp_affine.to(source.device)
                    kp_affine_p = utils.kp2im(pred_affine_kps_vis[0].detach().cpu().numpy(), target_affine[0].cpu().numpy(), radius=2) /255
                    kp_affine_p = torch.from_numpy(kp_affine_p).permute(2, 0, 1)[None]
                    kp_affine_p = kp_affine_p.to(source.device)
                    kp_mask = utils.kp2im(result_kps_vis[0].detach().cpu().numpy(), mask_3channels[0,1].cpu().numpy(), radius=2) / 255
                    kp_mask = torch.from_numpy(kp_mask).permute(2, 0, 1)[None]
                    kp_mask = kp_mask.to(source.device)
                    grid = torch.cat([source[:1], target[:1], kp_img[:1], kp_affine[:1], kp_affine_p[:1], kp_mask, change_range(reconstruct[:1], to_01=True)], dim=3)[0]
                    writer.add_image('iter {n} of image {i} (reconstruction, mask, affine, loss) = ({r},{m},{a},{l}) '.format(r=loss_reconstruction,m=loss_mask,a=loss_affine,l=loss,i=i,n=str(n_iter)), grid, n_iter)
                avg_loss += loss.item()
                writer.add_scalar('Loss : ' , loss, n_iter)
                writer.add_scalar('Reconstruction : ', loss_reconstruction, n_iter)
                writer.add_scalar('Mask : ', loss_mask, n_iter)
                writer.add_scalar('Affine : ', loss_affine, n_iter)
                n_iter += 1
            n_batch += 1
        avg_loss = avg_loss / (len(dataloader)*4)
        lr_scheduler.step(avg_loss)
        print('Epoch ', epoch, ' average loss ', avg_loss)
    torch.save({
        'keypoint_state_dict' : keypoint_model.state_dict(),
        'reconstruct_state_dict' : reconstruct_model.state_dict()
        }, opts.model_state_dir + str(opts.alpha))
    writer.close()


if __name__ == '__main__':
    app.run(main)
