"""!
@brief Infer Dataset Specific parameters and return generators

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""
import os
from approx_ensembles.fedenhance.__config__ import LIBRI_FS50K
from approx_ensembles.fedenhance.__config__ import CHUNK_LIBRI_FS50K
from approx_ensembles.fedenhance.__config__ import CHUNK_WHAM_ENH
import approx_ensembles.fedenhance.dataset_loader.libri_fsd as libri_fsd
import approx_ensembles.fedenhance.dataset_loader.wham_chunk as \
    wham_chunk
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

def pretrain_enhancement_single_node_setup(hparams):
    # Create all generators
    generators = {}
    for data_split in ['train', 'val', 'test']:
        if data_split == 'train':
            available_speech_percentage = hparams['available_speech_percentage']
        else:
            available_speech_percentage = 0.5
        data_loader = wham_chunk.Dataset(
            root_dirpath=CHUNK_WHAM_ENH,
            speaker_ids=[],
            available_speech_percentage=available_speech_percentage,
            split=data_split, sample_rate=hparams['fs'],
            timelength=hparams['audio_timelength'],
            zero_pad=True, augment=data_split=='train')
        generators[data_split] = data_loader.get_generator(
            batch_size=hparams['batch_size'],
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

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def enhancement_federated_setup(hparams):
    # Create all generators
    available_speaker_ids = os.listdir(os.path.join(CHUNK_LIBRI_FS50K, 'train'))
    # Split the available speaker ids to the individual nodes.
    splitted_ids = list(split(list(range(len(available_speaker_ids))),
                              hparams['n_fed_nodes']))

    # The first X nodes are always supervised, declared by the parameter.
    federated_generators_list = []
    num_supervised_nodes = int(
        hparams['p_supervised'] * len(splitted_ids))
    for j, ids in enumerate(splitted_ids):
        is_supervised = j < num_supervised_nodes

        data_loader = chunked_libri_fsd.Dataset(
            root_dirpath=CHUNK_LIBRI_FS50K,
            speaker_ids=ids,
            available_speech_percentage=hparams['available_speech_percentage'],
            split='train', sample_rate=hparams['fs'],
            timelength=hparams['audio_timelength'],
            zero_pad=True, augment=True)
        train_generator = data_loader.get_generator(
            batch_size=hparams['batch_size'],
            num_workers=hparams['n_jobs'])

        federated_generators_list.append(
            {'is_supervised': is_supervised,
             'node_id': j,
             'speaker_ids': ids,
             'train_generator': train_generator})

    val_generators = {}
    for data_split in ['val', 'test']:
        available_speech_percentage = 0.5
        data_loader = chunked_libri_fsd.Dataset(
            root_dirpath=CHUNK_LIBRI_FS50K,
            speaker_ids=[],
            available_speech_percentage=available_speech_percentage,
            split=data_split, sample_rate=hparams['fs'],
            timelength=hparams['audio_timelength'],
            zero_pad=True, augment=data_split=='train')
        val_generators[data_split] = data_loader.get_generator(
            batch_size=hparams['batch_size'],
            num_workers=hparams['n_jobs'])
    return federated_generators_list, val_generators


def enhancement_federated_individual_setup(hparams):
    # Create all generators
    available_speaker_ids = os.listdir(os.path.join(CHUNK_LIBRI_FS50K, 'train'))
    print(f'All available speakers: {len(available_speaker_ids)}')
    # Split the available speaker ids to the individual nodes.
    splitted_ids = list(split(list(range(len(available_speaker_ids))),
                              hparams['n_fed_nodes']))

    # The first X nodes are always supervised, declared by the parameter.
    federated_generators_list = []
    num_supervised_nodes = int(
        hparams['p_supervised'] * len(splitted_ids))
    for j, ids in enumerate(splitted_ids):
        is_supervised = j < num_supervised_nodes

        data_loader = chunked_libri_fsd.Dataset(
            root_dirpath=CHUNK_LIBRI_FS50K,
            speaker_ids=ids,
            available_speech_percentage=hparams['available_speech_percentage'],
            split='train', sample_rate=hparams['fs'],
            timelength=hparams['audio_timelength'],
            zero_pad=True, augment=True)
        train_generator = data_loader.get_generator(
            batch_size=hparams['batch_size'],
            num_workers=hparams['n_jobs'])

        federated_generators_list.append(
            {'is_supervised': is_supervised,
             'node_id': j,
             'speaker_ids': ids,
             'train_generator': train_generator})

    val_generators = {}
    for data_split in ['val', 'test']:
        available_speech_percentage = 0.5
        data_loader = chunked_libri_fsd.Dataset(
            root_dirpath=CHUNK_LIBRI_FS50K,
            speaker_ids=[],
            available_speech_percentage=available_speech_percentage,
            split=data_split, sample_rate=hparams['fs'],
            timelength=hparams['audio_timelength'],
            zero_pad=True, augment=data_split=='train')
        val_generators[data_split] = data_loader.get_generator(
            batch_size=hparams['batch_size'],
            num_workers=hparams['n_jobs'])
    return federated_generators_list, val_generators
