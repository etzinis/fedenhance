"""!
@brief Evaluate checkpoints of local optima for different interpolation values.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""
from approx_ensembles.mnist_prototype.net import MNISTnet
from approx_ensembles.__config__ import MNIST_LOG_PATH
import approx_ensembles.mnist_prototype.cmd_args_parser as parser
from torch.utils.data import DataLoader
from approx_ensembles.mnist_prototype.dataloader import NoisyMNIST

import torch
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import joblib

def _init_fn(worker_id):
    np.random.seed(13)


def infer_checkpoint_path(model_name, epoch_count):
    model_dirpath = os.path.join(MNIST_LOG_PATH, model_name)
    checkpoint_path = os.path.join(model_dirpath, 'chckpt_ep_{}.pt'.format(
        str(epoch_count)))
    return checkpoint_path


def get_model(model_name, epoch_cnt):
    checkpoint_path = infer_checkpoint_path(model_name, epoch_cnt)
    if 'small' in model_name:
        model = MNISTnet(n_intermediate_layers=0)
    elif 'big' in model_name:
        model = MNISTnet(n_intermediate_layers=4)
    else:
        raise ValueError('Cannot understand the number of layers')
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(checkpoint_path))
    return model


def run_model_on_data(model, data_gen):
    model.cuda()
    model.eval()
    eval_dfs = []
    with torch.no_grad():
        for (clean, noisy, labels, this_noise_type) in data_gen:
            clean, noisy = clean.cuda(), noisy.cuda()
            pred = model(noisy)
            mse = torch.abs(clean - pred) ** 2
            l1 = torch.abs(clean - pred)
            psnr = 10 * torch.log10(1. / (mse + 1e-16))

            psnr_df = pd.DataFrame(psnr.mean([-1, -2, -3]).detach().cpu(),
                                   columns=['PSNR'])
            l1_df = pd.DataFrame(l1.mean([-1, -2, -3]).detach().cpu(),
                                 columns=['L1'])
            labels_df = pd.DataFrame(labels, columns=['labels'])
            concatenated_df = pd.concat([psnr_df, labels_df, l1_df], axis=1,
                                        join='outer')
            eval_dfs.append(concatenated_df)

    return pd.concat(eval_dfs)


def create_new_interpolated_model(new_model, model1, model2, alpha):
    for i, layer in enumerate(model1.module.convs):
        if hasattr(layer, 'weight'):
            new_model.module.convs[i].weight = torch.nn.parameter.Parameter(
                alpha * model2.module.convs[i].weight +
                (1. - alpha) * model1.module.convs[i].weight)
        if hasattr(layer, 'bias'):
            new_model.module.convs[i].bias = torch.nn.parameter.Parameter(
                alpha * model2.module.convs[i].bias +
                (1. - alpha) * model1.module.convs[i].bias)
    return new_model

def fetch_and_eval_interpolated_checkpoints_on_data(
        model_name_1='small_snp_seed2568_seed_2568',
        model_name_2='small_gau_seed2568_seed_2568',
        epoch_cnt_1=150, epoch_cnt_2=0, data_gen=None,
        interpolation_points=[]):
    model1 = get_model(model_name_1, epoch_cnt_1)
    model2 = get_model(model_name_2, epoch_cnt_2)
    new_model = get_model(model_name_1, epoch_cnt_1)

    result_dic = {}
    for alpha in interpolation_points:
        new_model = create_new_interpolated_model(new_model, model1, model2,
                                                  alpha)
        eval_df = run_model_on_data(new_model, data_gen)
        result_dic[alpha] = eval_df
    return result_dic


def get_relevant_model_names(model_name):
    models_dirpath = MNIST_LOG_PATH
    stem = model_name.split('_')[0]
    relevant_model_names = []
    for another_model_name in os.listdir(models_dirpath):
        if another_model_name.startswith(stem) and another_model_name != \
                model_name:
            relevant_model_names.append(another_model_name)
    return relevant_model_names


if __name__ == "__main__":
    args = parser.get_args()
    hparams = vars(args)
    epochs_counts = [0, 1, 2, 5, 10, 15, 25, 35, 50, 75, 100,
                     150, 200, 250, 300, 400, 500, 599]
    interpolation_points = np.arange(0., 1.01, 0.1).tolist()
    bs = hparams['batch_size']
    num_workers = hparams['n_jobs']

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [cad for cad in hparams['cuda_available_devices']])

    # model_name = 'small_snp_seed2568_seed_2568'
    # model_name = 'small_gau_seed2568_seed_2568'
    # model_name = 'big_snp_seed_2568'
    # model_name = 'big_gau_seed_2568'
    model_name = hparams['eval_model_name']
    relevant_model_names = get_relevant_model_names(model_name)

    task = 'gaussian'

    model_pairs = [(model_name, m) for m in relevant_model_names]
    val_noise_types = hparams['noise_type']
    val_generators = dict(
        [(n, DataLoader(NoisyMNIST([n], train=False),
                        batch_size=bs, shuffle=False,
                        num_workers=num_workers,   pin_memory=True, worker_init_fn=_init_fn
                        ))
         for n in val_noise_types])

    for models in model_pairs:
        print('Evaluating pair: {}'.format(models))
        for n_type in tqdm(hparams['noise_type']):
            results_dirpath = os.path.join(MNIST_LOG_PATH, '../',
                                           'interpolation_results',
                                           models[0], models[1], n_type)
            os.makedirs(results_dirpath, exist_ok=True)

            for epoch_cnt in epochs_counts:
                try:
                    result_dic = fetch_and_eval_interpolated_checkpoints_on_data(
                        model_name_1 = models[0],
                        model_name_2 = models[1],
                        epoch_cnt_1 = epoch_cnt, epoch_cnt_2 = epoch_cnt,
                        data_gen=val_generators[n_type],
                        interpolation_points=interpolation_points)

                    savepath = os.path.join(results_dirpath,
                                            'epoch_{}.pkl'.format(epoch_cnt))
                    joblib.dump(result_dic, savepath)
                except Exception as e:
                    print('Failed at: {}'.format(epoch_cnt))
                    print(e)