import torch
import numpy as np
from skimage.transform import resize
import random

from torch.utils.data import Dataset, DataLoader

def frame_to_channels(frame, scale):
    # H*W*3 -> 3*H*W
    frame = resize(frame, (scale, scale), anti_aliasing=True)
    frame = np.swapaxes(frame, 2, 0)
    frame = np.swapaxes(frame, 1, 2)

    return frame


def channels_to_frame(channels):
    # 3*H*W -> H*W*3
    channels = np.swapaxes(channels, 1, 2)
    channels = np.swapaxes(channels, 2, 0)

    return channels


class TrainDataset(Dataset):

    def __init__(self, data):
        self.data = data
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.source, self.targets = self.split_source_target()

    def __getitem__(self, index):
        x = self.source[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)

    def split_source_target(self):
        dataset_size = len(self.data)
        source_list = []
        target_list = []

        for i in range(0, dataset_size):
            for j in range(0, 2):
                rand = random.randint(0, dataset_size - 1)
                if (rand == i):
                    rand = (rand + 1) % dataset_size
                source = frame_to_channels(self.data[i], scale=64)
                target = frame_to_channels(self.data[rand], scale=64)
                source_list.append(source)
                target_list.append(target)

        source = torch.tensor(source_list, dtype=torch.float32).to(self.device)
        target = torch.tensor(target_list, dtype=torch.float32).to(self.device)

        return source, target