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


def loss_alex(list1, list2, loss_fn_alex, mask):
    loss = 0
    for i in range(0, len(list1)):
        loss += loss_fn_alex(list1[i], list2[i]) * mask[i]
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

def get_gaussian_mean(x, axis, other_axis):
    """
    Args:
        x (float): Input images(BxCxHxW)
        axis (int): The index for weighted mean
        other_axis (int): The other index
    Returns: weighted index for axis, BxC
    """
    u = torch.softmax(torch.sum(x, axis=other_axis), axis=2)
    size = x.shape[axis]
    ind = torch.linspace(-1.0, 1.0, size).to(x.device)
    batch = x.shape[0]
    channel = x.shape[1]
    index = ind.repeat([batch, channel, 1])
    mean_position = torch.sum(index * u, dim=2)
    return mean_position


def xy_outputs(out, scaling, scale=16):
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
        x_dots = (2*x_dots / scale) -1
        y_dots = (2*y_dots / scale) -1

    return x_dots, y_dots


def get_2d_gaussian(key_x, key_y, std, scale=16, step=1):
    hw = scale
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

class UNet_Content(nn.Module):

    def __init__(self):
        super().__init__()

        factor = 1

        self.dconv_down1 = double_conv(3, 32*factor, first_layer=True)
        self.dconv_down2 = double_conv(32*factor, 64*factor)
        self.dconv_down3 = double_conv(64*factor, 128*factor)
        self.dconv_down4 = double_conv(128*factor, 256*factor)

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = conv1

        conv2 = self.dconv_down2(x)
        x = conv2

        conv3 = self.dconv_down3(x)
        x = conv3

        conv4 = self.dconv_down4(x)
        out = conv4

        return out


class UNet_Pose(nn.Module):

    def __init__(self, n_class, std):
        super().__init__()

        factor = 1

        self.dconv_down1 = double_conv(3, 32*factor, first_layer=True)
        self.dconv_down2 = double_conv(32*factor, 64*factor)
        self.dconv_down3 = double_conv(64*factor, 128*factor)
        self.dconv_down4 = double_conv(128*factor, 256*factor)

        self.conv_last = nn.Conv2d(256*factor, n_class, 1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.std = std
        self.heatmap = HeatMap(std, (16,16))

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

        gaussian, result_kps = self.heatmap(out)

        return gaussian, result_kps


class UNet_Reconstruct(nn.Module):

    def __init__(self, num_keypoints):
        super().__init__()

        factor = 1

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = double_conv(256//factor + num_keypoints, 256//factor, downsampling=False)
        self.dconv_up3 = double_conv(256//factor, 128//factor, downsampling=False)
        self.dconv_up2 = double_conv(128//factor, 64//factor, downsampling=False)
        self.dconv_up1 = double_conv(64//factor, 32//factor, downsampling=False)

        self.conv_last = nn.Conv2d(32//factor, 3, 1)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.dconv_up4(x)
        x = self.upsample(x)

        #y = torch.ones_like(x[:, :result_kps.shape[1]]) * result_kps[:, :, None, None]
        #y = get_2d_gaussian(key_x, key_y, self.std, 32)
        #x = torch.cat([x, y], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)

        #y = torch.ones_like(x[:, :result_kps.shape[1]]) * result_kps[:, :, None, None]
        #y = get_2d_gaussian(key_x, key_y, self.std, 64)
        #x = torch.cat([x, y], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)

        #y = torch.ones_like(x[:, :result_kps.shape[1]]) * result_kps[:, :, None, None]
        #y = get_2d_gaussian(key_x, key_y, self.std, 128)
        #x = torch.cat([x, y], dim=1)
        x = self.dconv_up1(x)

        #y = torch.ones_like(x[:, :result_kps.shape[1]]) * result_kps[:, :8, None, None]
        #x = torch.cat([x, y], dim=1)
        out = self.conv_last(x)
        out = torch.tanh(out)

        return out

class UNet_Generator(nn.Module):
    def __init__(self, keypoints=18, std=0.1):
        super(UNet_Generator, self).__init__()
        self.content_encoder = UNet_Content()
        self.pose_encoder = UNet_Pose(keypoints, std)
        self.reconstruct = UNet_Reconstruct(keypoints)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, source, target):
        content = self.content_encoder(source)
        pose, keypoints = self.pose_encoder(target)
        x = torch.cat([content,pose], dim=1)
        recovered = self.reconstruct(x)
        return recovered, keypoints



class IMM(nn.Module):
    def __init__(self, dim=10, heatmap_std=0.1, in_channel=3, h_channel=32):
        """
        It should be noted all params has been fixed to Jakab 2018 paper.
        Goto the original class if params and layers need to be changed.
        Images should be rescaled to 128*128
        """
        super(IMM, self).__init__()
        self.content_encoder = ContentEncoder(in_channel, h_channel)
        self.pose_encoder = PoseEncoder(dim, heatmap_std, in_channel, h_channel)
        self.generator = Generator(channels=8*h_channel+dim, h_channel=h_channel)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        content_x = self.content_encoder(x)
        pose_y, pose_coord, pose_val = self.pose_encoder(y)
        code = torch.cat((content_x, pose_y), dim=1)
        recovered_y = self.generator(code)
        return recovered_y, pose_coord, pose_y, pose_val


class ContentEncoder(nn.Module):
    def __init__(self, in_channel=3, h_channel=64):
        super(ContentEncoder, self).__init__()
        self.conv1_1 = Conv_Block(in_channel, h_channel, (3, 3))
        self.conv1_2 = Conv_Block(h_channel, h_channel, (3, 3))

        self.conv2_1 = Conv_Block(h_channel, 2 * h_channel, (3, 3), downsample=True)
        self.conv2_2 = Conv_Block(2 * h_channel, 2 * h_channel, (3, 3))

        self.conv3_1 = Conv_Block(2 * h_channel, 4 * h_channel, (3, 3), downsample=True)
        self.conv3_2 = Conv_Block(4 * h_channel, 4 * h_channel, (3, 3))

        self.conv4_1 = Conv_Block(4 * h_channel, 8 * h_channel, (3, 3), downsample=True)
        self.conv4_2 = Conv_Block(8 * h_channel, 8 * h_channel, (3, 3))

        # self.out_conv = Conv_Block(8 * h_channel, h_channel, (3, 3))
        self.conv_layers = nn.ModuleList([
            self.conv1_1,
            self.conv1_2,
            self.conv2_1,
            self.conv2_2,
            self.conv3_1,
            self.conv3_2,
            self.conv4_1,
            self.conv4_2
            # self.out_conv
        ])

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


class PoseEncoder(nn.Module):
    def __init__(self, dim=10, heatmap_std=0.1, in_channel=3, h_channel=64, heatmap_size=16):
        """
        Args:
            dim (int): Num of keypoints
        """
        super(PoseEncoder, self).__init__()
        self.conv1_1 = Conv_Block(in_channel, h_channel, (3, 3))
        self.conv1_2 = Conv_Block(h_channel, h_channel, (3, 3))

        self.conv2_1 = Conv_Block(h_channel, 2 * h_channel, (3, 3), downsample=True)
        self.conv2_2 = Conv_Block(2 * h_channel, 2 * h_channel, (3, 3))

        self.conv3_1 = Conv_Block(2 * h_channel, 4 * h_channel, (3, 3), downsample=True)
        self.conv3_2 = Conv_Block(4 * h_channel, 4 * h_channel, (3, 3))

        self.conv4_1 = Conv_Block(4 * h_channel, 8 * h_channel, (3, 3), downsample=True)
        self.conv4_2 = Conv_Block(8 * h_channel, 8 * h_channel, (3, 3))

        self.out_conv = nn.Sequential(nn.Conv2d(8 * h_channel, dim, (1, 1)))
        self.conv_layers = nn.ModuleList([
            self.conv1_1,
            self.conv1_2,
            self.conv2_1,
            self.conv2_2,
            self.conv3_1,
            self.conv3_2,
            self.conv4_1,
            self.conv4_2,
            self.out_conv
        ])
        self.heatmap = HeatMap(heatmap_std, (heatmap_size, heatmap_size))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        heatmap, coord, max_val = self.heatmap(x)
        return heatmap, coord, max_val


class Generator(nn.Module):

    def __init__(self, channels=64 + 10, h_channel=64):
        super(Generator, self).__init__()
        self.conv1_1 = Conv_Block(channels, 8 * h_channel, (3, 3))
        self.conv1_2 = Conv_Block(8 * h_channel, 8 * h_channel, (3, 3), upsample=True)

        self.conv2_1 = Conv_Block(8 * h_channel, 4 * h_channel, (3, 3))
        self.conv2_2 = Conv_Block(4 * h_channel, 4 * h_channel, (3, 3), upsample=True)

        self.conv3_1 = Conv_Block(4 * h_channel, 2 * h_channel, (3, 3))
        self.conv3_2 = Conv_Block(2 * h_channel, 2 * h_channel, (3, 3), upsample=True)

        self.conv4_1 = Conv_Block(2 * h_channel, h_channel, (3, 3))
        self.conv4_2 = Conv_Block(h_channel, h_channel, (3, 3))

        self.final_conv = nn.Conv2d(h_channel, 3, (3, 3), padding=[1, 1])
        self.conv_layers = nn.ModuleList([
            self.conv1_1,
            self.conv1_2,
            self.conv2_1,
            self.conv2_2,
            self.conv3_1,
            self.conv3_2,
            self.conv4_1,
            self.conv4_2,
            self.final_conv
        ])

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        # return x
        #return (nn.functional.tanh(x) + 1) / 2.0
        x = torch.tanh(x)
        return x


class HeatMap(nn.Module):
    """
    Refine the estimated pose map to be gaussian distributed heatmap.
    Calculate the gaussian mean value.
    Params:
    std: standard deviation of gaussian distribution
    output_size: output feature map size
    """

    def __init__(self, std, output_size):
        super(HeatMap, self).__init__()
        self.std = std
        self.out_h, self.out_w = output_size

    def forward(self, x, h_axis=2, w_axis=3):
        """
        x: feature map BxCxHxW
        h_axis: the axis of Height
        w_axis: the axis of width
        """
        # self.in_h, self.in_w = x.shape[h_axis:]
        batch, channel = x.shape[:h_axis]

        # Calculate weighted position of joint(-1~1)
        #x_mean = get_gaussian_mean(x, 2, 3)
        #y_mean = get_gaussian_mean(x, 3, 2)
        x_mean, y_mean = xy_outputs(x, True)

        coord = torch.stack([y_mean, x_mean], dim=2)

        x_mean = x_mean.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.out_h, self.out_w)
        y_mean = y_mean.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.out_h, self.out_w)

        x_ind = torch.tensor(torch.linspace(-1.0, 1.0, self.out_h)).unsqueeze(-1).repeat(batch, channel, 1, self.out_w).to(
            x.device)
        y_ind = torch.tensor(torch.linspace(-1.0, 1.0, self.out_w)).unsqueeze(0).repeat(batch, channel, self.out_h, 1).to(
            x.device)

        dist = (x_ind - x_mean) ** 2 + (y_ind - y_mean) ** 2

        res = torch.exp(-(dist + 1e-6).sqrt_() / (2 * self.std ** 2))
        max_val = torch.amax(res[0], dim=(1,2))
        return res, coord, max_val


class Conv_Block(nn.Module):
    def __init__(self, inc, outc, size, downsample=False, upsample=False):
        super(Conv_Block, self).__init__()
        block = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=size),
            nn.BatchNorm2d(outc),
            nn.LeakyReLU()
        ]
        if downsample:
            block += [nn.MaxPool2d(kernel_size=2, stride=2)]
        if upsample:
            block += [nn.UpsamplingBilinear2d(scale_factor=2)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


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
