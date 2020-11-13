import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import torch
from Models import xy_outputs, get_2d_gaussian, train_keypoints, UNet, UNet_Reconstruct
from TrainDataset import frame_to_channels, channels_to_frame, TrainDataset
import pickle

from torch.utils.data import Dataset, DataLoader

def test_keypoints(keypoint_model, img):

    eval_list = []

    fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))

    im = np.array(img)
    im = resize(im, (80, 80), anti_aliasing=True)
    ax1.imshow(im)

    img_arr = np.swapaxes(im, 2, 0)
    img_arr = np.swapaxes(img_arr, 1, 2)
    eval_list.append(img_arr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    target = torch.tensor(eval_list, dtype=torch.float32).to(device)
    output = keypoint_model(target)

    out_x, out_y = xy_outputs(output, scaling=False)

    out_x = out_x.cpu().detach().numpy()
    out_y = out_y.cpu().detach().numpy()

    ax1.scatter(x=out_x, y=out_y, c='r', s=40)

def test_reconstruct(keypoint_model, reconstruct_model, img_source, img_target, scale=80):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  img_source = resize(img_source, (scale, scale), anti_aliasing=True)
  img_target = resize(img_target, (scale, scale), anti_aliasing=True)
  example_source = frame_to_channels(img_source, scale=scale)
  example_source = torch.tensor([example_source],dtype=torch.float32).to(device)
  example_target = frame_to_channels(img_target, scale=scale)
  example_target = torch.tensor([example_target],dtype=torch.float32).to(device)

  example_output = keypoint_model(example_target)
  example_keypoints_x, example_keypoints_y = xy_outputs(example_output, scaling=True)
  example_gauss = get_2d_gaussian(example_keypoints_x, example_keypoints_y, std=0.1)
  reconstruct_example = reconstruct_model(example_source, example_gauss)
  reconstruct_example = reconstruct_example.cpu().detach().numpy()[0]
  reconstruct_example = channels_to_frame(reconstruct_example)
  fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,15))

  ax1.imshow(resize(img_source, (scale, scale), anti_aliasing=True))
  ax2.imshow(resize(img_target, (scale, scale), anti_aliasing=True))
  ax3.imshow(reconstruct_example)
  ax1.set_title("Source")
  ax2.set_title("Target")
  ax3.set_title("Reconstruct")


def example():
    vid_path = '77.pkl'

    with open(vid_path, 'rb') as file:
        video_list = pickle.load(file)['video']

    dataset = TrainDataset(video_list)
    dataloader = DataLoader(dataset, batch_size=18, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    keypoint_model = UNet(6).to(device)
    reconstruct_model = UNet_Reconstruct(3, 6).to(device)

    train_keypoints(keypoint_model, reconstruct_model, dataloader)

example()