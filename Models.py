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

from TrainDataset import channels_to_frame, frame_to_channels


def loss_alex(list1, list2, loss_fn_alex):
    loss = 0
    for i in range(0, len(list1)):
        loss += loss_fn_alex(list1[i], list2[i])
    loss /= len(list1)

    return loss


def rgba_to_rgb(rgba):
    rgb = np.zeros((rgba.shape[0], rgba.shape[1], 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    rgb[:, :, 0] = r * a
    rgb[:, :, 1] = g * a
    rgb[:, :, 2] = b * a

    rgb = np.swapaxes(rgb, 2, 0)
    rgb = np.swapaxes(rgb, 1, 2)

    return rgb


def xy_outputs(out, scaling, scale=80):
    softmax_out = softmax(out.reshape(out.shape[0], out.shape[1], -1), dim=-1)
    softmax_out = softmax_out.reshape(*out.shape)
    x = torch.arange(scale, device=out.device)
    y = torch.arange(scale, device=out.device)
    grid_x, grid_y = torch.meshgrid(x, y)
    grid_x = grid_x[None, None]
    grid_y = grid_y[None, None]
    prob_x = (grid_x * softmax_out)
    prob_y = (grid_y * softmax_out)

    x_dots = prob_x.sum(-1).sum(-1)
    y_dots = prob_y.sum(-1).sum(-1)

    if (scaling):
        x_dots = (2 * x_dots / scale) - 1
        y_dots = (2 * y_dots / scale) - 1

    return x_dots, y_dots


def get_2d_gaussian(key_x, key_y, std, scale=128, step=1):
    hw = int(scale / 8) * step
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g_yx_set = torch.ones(len(key_x), len(key_x[0]), hw, hw).to(device)

    for i in range(0, len(key_x)):
        key_x_elem = key_x[i]
        key_y_elem = key_y[i]

        x = torch.linspace(-1.0, 1.0, steps=hw).to(device)
        y = torch.linspace(-1.0, 1.0, steps=hw).to(device)
        key_y_elem, key_x_elem = torch.unsqueeze(key_y_elem, -1), torch.unsqueeze(key_x_elem, -1)
        x = torch.reshape(x, [1, 1, hw])
        y = torch.reshape(y, [1, 1, hw])
        g_x = torch.square(x - key_x_elem) / (2 * math.pow(std, 2))
        g_y = torch.square(y - key_y_elem) / (2 * math.pow(std, 2))

        g_x = torch.exp(-g_x)
        g_y = torch.exp(-g_y)

        g_y = torch.unsqueeze(g_y, axis=3)
        g_x = torch.unsqueeze(g_x, axis=2)
        g_yx = torch.matmul(g_y, g_x)
        g_yx_set[i] = g_yx[0]

    return g_yx_set


def double_conv(in_channels, out_channels, first_layer=False, downsampling=True):
    k_size = 3
    s_size = 2
    p_size = 1
    if(first_layer==True):
        k_size = 7
        s_size = 1
        p_size = 3
    if(downsampling==False):
        s_size = 1

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, k_size, stride=s_size, padding=p_size),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(3, 32, first_layer=True)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 256)

        self.conv_last = nn.Conv2d(256, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = conv1

        conv2 = self.dconv_down2(x)
        x = conv2

        conv3 = self.dconv_down3(x)
        x = conv3

        conv4 = self.dconv_down4(x)
        x = conv4

        out = self.conv_last(x)
        out = torch.tanh(out)

        return out


class UNet_Reconstruct(nn.Module):

    def __init__(self, n_class, num_keypoints):
        super().__init__()

        factor = 1
        self.dconv_down1 = double_conv(3, 32//factor, first_layer=True)
        self.dconv_down2 = double_conv(32//factor, 64//factor)
        self.dconv_down3 = double_conv(64//factor, 128//factor)
        self.dconv_down4 = double_conv(128//factor, 256//factor)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = double_conv(256//factor + num_keypoints, 256//factor, downsampling=False)
        self.dconv_up3 = double_conv(256//factor, 128//factor, downsampling=False)
        self.dconv_up2 = double_conv(128//factor, 64//factor, downsampling=False)
        self.dconv_up1 = double_conv(64//factor, 32//factor, downsampling=False)

        self.conv_last = nn.Conv2d(32//factor, 3, 1)
        self.num_keypoints = num_keypoints

    def forward(self, source, result_kps):
        x = source

        conv1 = self.dconv_down1(x)
        x = conv1

        conv2 = self.dconv_down2(x)
        x = conv2

        conv3 = self.dconv_down3(x)
        x = conv3

        x = self.dconv_down4(x)

        #y = torch.ones_like(x[:, :result_kps.shape[1]]) * result_kps[:, :, None, None]
        key_x = result_kps[:,:self.num_keypoints]
        key_y = result_kps[:,self.num_keypoints:(2*self.num_keypoints)]
        y = get_2d_gaussian(key_x, key_y, 0.1)

        x = torch.cat([x, y], dim=1)
        x = self.dconv_up4(x)
        x = self.upsample(x)

        #y = torch.ones_like(x[:, :result_kps.shape[1]]) * result_kps[:, :, None, None]
        #x = torch.cat([x, y], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)

        #y = torch.ones_like(x[:, :result_kps.shape[1]]) * result_kps[:, :, None, None]
        #x = torch.cat([x, y], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)

        #y = torch.ones_like(x[:, :result_kps.shape[1]]) * result_kps[:, :, None, None]
        #x = torch.cat([x, y], dim=1)

        x = self.dconv_up1(x)

        #y = torch.ones_like(x[:, :result_kps.shape[1]]) * result_kps[:, :8, None, None]
        #x = torch.cat([x, y], dim=1)
        out = self.conv_last(x)
        out = torch.tanh(out)

        return out


def train_keypoints(keypoint_model, reconstruct_model, dataloader, lr=1e-3, wd=5 * 1e-4, n_epochs=1000, patience=10,
                    std=0.1):
    best_loss = np.inf
    best_weights = None
    best_weights_r = None
    history = []
    loss_train = []
    loss_val = []
    loss_epoch = []

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss_fn_alex = lpips.LPIPS(net='alex')
    optimizer = optim.Adam(list(keypoint_model.parameters()) + list(reconstruct_model.parameters()), lr=lr,
                           weight_decay=wd)

    if (torch.cuda.is_available()):
        loss_fn_alex.cuda()

    writer = SummaryWriter()

    for epoch in range(n_epochs):
        stats = {'epoch': epoch + 1, 'total': n_epochs}

        for phase in ('train', 'val'):
            training = phase == 'train'
            running_loss = 0.0
            optimizer.zero_grad()

            # compute gradients only during 'train' phase
            with torch.set_grad_enabled(training):
                for batch_index, (source, target) in enumerate(dataloader):
                    target_outputs = keypoint_model(target)
                    batch_size = target_outputs.shape[0]
                    result_x, result_y = xy_outputs(target_outputs, scaling=True)
                    result_x_unscaled, result_y_unscaled = xy_outputs(target_outputs, scaling=False)
                    target_gauss = get_2d_gaussian(result_x, result_y, std=std)
                    source = torch.tensor(source, dtype=torch.float32).to(device)
                    reconstruct = reconstruct_model(source, target_gauss)
                    loss = loss_alex(reconstruct, target, loss_fn_alex)
                    writer.add_scalar('Loss/train', loss, epoch)

                    if (epoch % 10 == 0):
                        img = channels_to_frame(source[0].cpu().detach().numpy())
                        x_cord = result_x_unscaled[0].cpu().detach().numpy()
                        y_cord = result_y_unscaled[0].cpu().detach().numpy()

                        for i in range(len(x_cord)):
                            x = int(x_cord[i])
                            y = int(y_cord[i])
                            img[x - 1:x + 1, y - 1:y + 1] = [255, 0, 0]

                        img = frame_to_channels(img)

                        img_batch = np.zeros((4, 3, scale, scale))
                        img_batch[0] = source[0].cpu().detach().numpy()
                        img_batch[1] = target[0].cpu().detach().numpy()
                        img_batch[2] = reconstruct[0].cpu().detach().numpy()
                        img_batch[3] = img

                        writer.add_images('Image/' + str(epoch), img_batch, epoch)

                # don't update weights and rates when in 'val' phase
                if training:
                    target_outputs.register_hook(lambda x: print(x))
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss
            stats[phase] = epoch_loss

            # early stopping: save weights of the best model so far
            if phase == 'val':
                if epoch_loss < best_loss:
                    print('loss improvement on epoch: %d' % (epoch + 1))
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(keypoint_model.state_dict())
                    best_weights_r = copy.deepcopy(reconstruct_model.state_dict())
                    no_improvements = 0
                else:
                    lr *= 0.1
                    no_improvements += 1

                loss_epoch.append(stats['epoch'])
                loss_train.append(stats['train'])
                loss_val.append(stats['val'])

        history.append(stats)
        print('[{epoch:03d}/{total:03d}] train: {train:.4f} - val: {val:.4f}'.format(**stats))
        if no_improvements >= patience:
            print('early stopping after epoch {epoch:03d}'.format(**stats))
            break

        keypoint_model.load_state_dict(best_weights)
        reconstruct_model.load_state_dict(best_weights_r)