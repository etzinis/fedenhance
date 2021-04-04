"""!
@brief Infer Dataset Specific parameters and return generators

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

from approx_ensembles.fedenhance.__config__ import LIBRI_FS50K
from approx_ensembles.fedenhance.__config__ import CHUNK_LIBRI_FS50K
import approx_ensembles.fedenhance.dataset_loader.libri_fsd as libri_fsd
import approx_ensembles.fedenhance.dataset_loader.chunked_libri_fsd as \
    chunked_libri_fsd


def setup(hparams):
    # Create all generators
    generators = {}
    for data_split in ['train', 'val', 'test']:
        data_loader = libri_fsd.Dataset(
            root_dirpath=LIBRI_FS50K,
            speaker_ids=[],
            split=data_split, sample_rate=hparams['fs'],
            timelength=hparams['audio_timelength'],
            zero_pad=True, augment=data_split=='train')
        generators[data_split] = data_loader.get_generator(
            batch_size=hparams['batch_size'], num_workers=hparams['n_jobs'])
    return generators

def mixit_setup(hparams):
    # Create all generators
    generators = {}
    for data_split in ['train', 'val', 'test']:
        data_loader = libri_fsd.Dataset(
            root_dirpath=LIBRI_FS50K,
            speaker_ids=[],
            split=data_split, sample_rate=hparams['fs'],
            timelength=hparams['audio_timelength'],
            zero_pad=True, augment=data_split=='train')
        generators[data_split] = data_loader.get_generator(
            batch_size=2*hparams['batch_size'],
            num_workers=hparams['n_jobs'])
    return generators

def enhancement_single_node_setup(hparams):
    # Create all generators
    generators = {}
    for data_split in ['train', 'val', 'test']:
        if data_split == 'train':
            available_speech_percentage = hparams['available_speech_percentage']
        else:
            available_speech_percentage = 0.5
        data_loader = chunked_libri_fsd.Dataset(
            root_dirpath=CHUNK_LIBRI_FS50K,
            speaker_ids=[],
            available_speech_percentage=available_speech_percentage,
            split=data_split, sample_rate=hparams['fs'],
            timelength=hparams['audio_timelength'],
            zero_pad=True, augment=data_split=='train')
        generators[data_split] = data_loader.get_generator(
            batch_size=hparams['batch_size'],
            num_workers=hparams['n_jobs'])
    return generators
