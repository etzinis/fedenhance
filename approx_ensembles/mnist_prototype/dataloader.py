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
import

class NoisyMNIST(Dataset):
    def __init__(self, params, train):
        self.mnist_dset = torchvision.datasets.MNIST('../data/', train=train, download=True,
                                                     transform=torchvision.transforms.Compose([
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize(
                                                             (0.1307,), (0.3081,))
                                                     ]))
        self.noise_types = params['valid_noises']

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
        else:
            raise ValueError(
                'Given noise type: {} is invalid'.format(noise_type))
        img_noisy = torch.Tensor(img_noisy)
        return img_clean, img_noisy, img_label, noise_type

    def __len__(self):
        return len(self.mnist_dset)


if __name__ == "__main__":
    params = default()

    save_dir = '../temp'
    os.makedirs(save_dir, exist_ok=True)

    for noise in ['gaussian', 'salt_and_pepper', 'speckle']:
        params['valid_noises'] = [noise]

        dset = NoisyMNIST(params, train=True)
        clean, noisy, label, _ = dset[0]
        
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(clean.numpy()[0])
        ax[1].imshow(noisy.numpy()[0])
        plt.title('Noise {} digit {}'.format(noise, label))
        plt.savefig(save_dir + '/{}.png'.format(noise))
