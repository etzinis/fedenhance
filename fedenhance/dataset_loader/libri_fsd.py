"""!
@brief Pytorch dataloader for wham dataset for multiple gender combinations.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import torch
import os
import numpy as np
import pickle
import glob2
import sys

import fedenhance.dataset_loader.abstract_dataset as abstract_dataset
from scipy.io import wavfile
import warnings
from time import time

EPS = 1e-8


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class Dataset(torch.utils.data.Dataset, abstract_dataset.Dataset):
    """ Dataset class for WHAM source separation and speech enhancement tasks.

    Example of kwargs:
        root_dirpath='/mnt/data/wham', task='enh_single',
        split='tr', sample_rate=8000, timelength=4.0,
        normalize_audio=False, n_samples=0, zero_pad=False
    """
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        warnings.filterwarnings("ignore")
        self.kwargs = kwargs

        self.zero_pad = self.get_arg_and_check_validness(
            'zero_pad', known_type=bool)

        self.augment = self.get_arg_and_check_validness(
            'augment', known_type=bool)

        self.split = self.get_arg_and_check_validness(
            'split', known_type=str, choices=['train', 'test', 'val'])

        self.sample_rate = self.get_arg_and_check_validness(
            'sample_rate', known_type=int, choices=[16000])

        self.root_path = self.get_arg_and_check_validness(
            'root_dirpath', known_type=str,
            extra_lambda_checks=[lambda y: os.path.lexists(y)])
        self.dataset_dirpath = self.get_path()

        self.timelength = self.get_arg_and_check_validness(
            'timelength', known_type=float)

        self.time_samples = int(self.sample_rate * self.timelength)

        # If specific Speaker ids are given
        self.speaker_ids = self.get_arg_and_check_validness(
            'speaker_ids', known_type=list)
        # Create the indexing for the dataset.
        available_speaker_ids = os.listdir(self.dataset_dirpath)
        self.all_available_speaker_ids = os.listdir(self.dataset_dirpath)
        sorted_speaker_ids_ints = sorted(map(int, available_speaker_ids))
        available_speaker_ids_ints = sorted_speaker_ids_ints
        if self.speaker_ids:
            available_speaker_ids_ints = [
                sorted_speaker_ids_ints[idx] for idx in self.speaker_ids]

        self.sources_paths = []
        for speaker_id in available_speaker_ids_ints:
            this_dirpath = os.path.join(self.dataset_dirpath, str(speaker_id))
            noise_paths = glob2.glob(this_dirpath+'/noise/*.wav')
            speech_paths = glob2.glob(this_dirpath+'/speech/*.wav')
            this_sources_info = [{
                'noise_path': noise_path,
                'speech_path': speech_path
            } for (noise_path, speech_path) in zip(noise_paths, speech_paths)]
            self.sources_paths += this_sources_info

    def get_path(self):
        path = os.path.join(self.root_path, self.split)
        if os.path.lexists(path):
            return path
        else:
            raise IOError('Dataset path: {} not found!'.format(path))

    def safe_pad(self, tensor_wav):
        if self.zero_pad and tensor_wav.shape[0] < self.time_samples:
            appropriate_shape = tensor_wav.shape
            padded_wav = torch.zeros(
                list(appropriate_shape[:-1]) + [self.time_samples],
                dtype=torch.float32)
            padded_wav[:tensor_wav.shape[0]] = tensor_wav
            return padded_wav[:self.time_samples]
        else:
            return tensor_wav[:self.time_samples]

    def __len__(self):
        return len(self.sources_paths)

    def __getitem__(self, idx):
        if self.augment:
            the_time = int(np.modf(time())[0] * 100000000)
            np.random.seed(the_time)

        example_sources_paths = self.sources_paths[idx]

        _, noise_waveform = wavfile.read(example_sources_paths['noise_path'])
        _, speech_waveform = wavfile.read(example_sources_paths['speech_path'])

        max_len = len(noise_waveform)
        rand_start = 0
        if self.augment and max_len > self.time_samples:
            rand_start = np.random.randint(0, max_len - self.time_samples)
        noise_waveform = noise_waveform[rand_start:rand_start+self.time_samples]
        np_noise_wav = np.array(noise_waveform)
        noise_wav = torch.tensor(np_noise_wav, dtype=torch.float32)
        noise_wav = self.safe_pad(noise_wav)

        max_len = len(speech_waveform)
        rand_start = 0
        if self.augment and max_len > self.time_samples:
            rand_start = np.random.randint(0, max_len - self.time_samples)
        speech_waveform = speech_waveform[
                          rand_start:rand_start + self.time_samples]
        np_speech_wav = np.array(speech_waveform)
        speech_wav = torch.tensor(np_speech_wav, dtype=torch.float32)
        speech_wav = self.safe_pad(speech_wav)

        return speech_wav, noise_wav

    def get_generator(self, batch_size=4, shuffle=True, num_workers=4):
        generator_params = {'batch_size': batch_size,
                            'shuffle': shuffle,
                            'num_workers': num_workers,
                            'drop_last': True}
        return torch.utils.data.DataLoader(self, **generator_params,
                                           pin_memory=True)


def test_generator():
    dataset_root_p = '/mnt/data/FedEnhance/'
    batch_size = 3
    sample_rate = 16000
    timelength = 4.0
    speaker_ids = [x for x in range(100)]
    time_samples = int(sample_rate * timelength)
    max_abs_snr = 5.
    data_loader = Dataset(
        root_dirpath=dataset_root_p,
        speaker_ids=speaker_ids,
        split='train', sample_rate=sample_rate, timelength=timelength,
        zero_pad=True, augment=True)
    generator = data_loader.get_generator(batch_size=batch_size, num_workers=1)

    for speech_wavs, noise_wavs in generator:
        assert speech_wavs.shape == (batch_size, time_samples)
        assert noise_wavs.shape == (batch_size, time_samples)

if __name__ == "__main__":
    test_generator()
