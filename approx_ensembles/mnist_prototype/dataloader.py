"""!
@brief Noisy MNIST dataset loader

@author Jonah Casebeer {jonahmc2@illinois.edu}
@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import torch
import torchvision
from torch.utils.data import Dataset
import skimage.util
import random
import matplotlib.pyplot as plt
import os
import approx_ensembles.mnist_prototype.cmd_args_parser as cmd_args_parser
from approx_ensembles.__config__ import MNIST_PATH, MNIST_LOG_PATH

class NoisyMNIST(Dataset):
    def __init__(self, noise_types, train):
        self.mnist_dset = torchvision.datasets.MNIST(
            MNIST_PATH, train=train, download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
        self.noise_types = noise_types

    @staticmethod
    def add_normal(img):
        return skimage.util.random_noise(img, mode='gaussian', mean=0,
                                         var=0.05, clip=True)

    @staticmethod
    def add_salt_and_pepper(img):
        return skimage.util.random_noise(img, mode='s&p', salt_vs_pepper=0.5,
                                         clip=True)

    @staticmethod
    def mult_normal(img):
        return skimage.util.random_noise(img, mode='speckle', mean=0,
                                         var=0.05, clip=True)

    def __getitem__(self, idx):
        img_clean, img_label = self.mnist_dset[idx]
        noise_type = random.choice(self.noise_types)
        clean_np = img_clean.numpy()

        if noise_type == 'gaussian':
            img_noisy = self.add_normal(clean_np)
        elif noise_type == 'salt_and_pepper':
            img_noisy = self.add_salt_and_pepper(clean_np)
        elif noise_type == 'speckle':
            img_noisy = self.mult_normal(clean_np)
        elif noise_type == 'none':
            img_noisy = clean_np
        else:
            raise ValueError(
                'Given noise type: {} is invalid'.format(noise_type))
        img_noisy = torch.Tensor(img_noisy)
        return img_clean, img_noisy, img_label, noise_type

    def __len__(self):
        return len(self.mnist_dset)


if __name__ == "__main__":
    args = cmd_args_parser.get_args()

    save_dir = os.path.join(MNIST_LOG_PATH, 'temp')
    os.makedirs(save_dir, exist_ok=True)

    for noise in args.noise_type:
        dset = NoisyMNIST([noise], train=True)
        clean, noisy, label, _ = dset[0]
        
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(clean.numpy()[0])
        ax[1].imshow(noisy.numpy()[0])
        plt.title('Noise {} digit {}'.format(noise, label))
        plt.savefig(save_dir + '/{}.png'.format(noise))
