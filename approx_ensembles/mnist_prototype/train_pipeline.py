"""!
@brief Running training for noisy mnist

@author Jonah Casebeer {jonahmc2@illinois.edu}
@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import sys

from approx_ensembles.__config__ import API_KEY
from comet_ml import Experiment
from pprint import pprint

import tqdm
import os

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import approx_ensembles.mnist_prototype.cmd_args_parser as parser
import approx_ensembles.mnist_prototype.cometml_report as cometml_report

from approx_ensembles.mnist_prototype.net import MNISTnet
from approx_ensembles.mnist_prototype.dataloader import NoisyMNIST
from approx_ensembles.__config__ import MNIST_LOG_PATH


args = parser.get_args()
hparams = vars(args)

# Setup datasets, evaluate on all types of noises
train_generator = DataLoader(NoisyMNIST(hparams['noise_type'], train=True),
                             batch_size=hparams['batch_size'], shuffle=True)
val_noise_types = ['none', 'salt_and_pepper', 'gaussian', 'speckle']
val_generators = dict(
    [(n, DataLoader(NoisyMNIST([n], train=False),
                    batch_size=hparams['batch_size'], shuffle=True))
     for n in val_noise_types])

# Setup experiment in cometml configurations
experiment = Experiment(API_KEY, project_name=hparams["project_name"])
experiment.log_parameters(hparams)
experiment_name = '_'.join(
    hparams['cometml_tags'] + ['seed'] + [str(hparams['seed'])])

# Setup paths
model_log_dir = os.path.join(MNIST_LOG_PATH, experiment_name)

for tag in hparams['cometml_tags']:
    experiment.add_tag(tag)
if hparams['experiment_name'] is not None:
    experiment.set_name(hparams['experiment_name'])
else:
    experiment.set_name(experiment_name)

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
    [cad for cad in hparams['cuda_available_devices']])

val_losses = {}
train_loss = F.l1_loss
def batch_l1_metric(pred, clean):
    return torch.reshape(torch.abs(pred - clean), [pred.shape[0], -1]).mean(-1)

for noise_type in val_noise_types:
    val_losses['val_' + noise_type] = batch_l1_metric

torch.manual_seed(hparams['seed'])
model = MNISTnet()
model = torch.nn.DataParallel(model).cuda()
opt = torch.optim.SGD(model.parameters(), lr=hparams['learning_rate'])

for i in range(hparams['n_epochs']):
    res_dic = {}
    for loss_name in val_losses:
        res_dic[loss_name] = {'mean': 0., 'std': 0., 'acc': []}
    print("Noisy MNIST: {} - {} || Epoch: {}/{}".format(
        experiment.get_key(), experiment.get_tags(), i+1, hparams['n_epochs']))
    model.train()

    for (clean, noisy, _, _) in tqdm.tqdm(train_generator, desc='Training'):
        opt.zero_grad()
        clean, noisy = clean.cuda(), noisy.cuda()
        pred = model(noisy)
        loss = train_loss(pred, clean)
        loss.backward()
        opt.step()

        if i % hparams['ckpt_period'] == 0:
            ckpt_path = os.path.join(model_log_dir, 'chckpt')
            os.makedirs(ckpt_path, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path + '_ep_{}.pt'.format(i))

    for n_type in val_noise_types:
        model.eval()
        with torch.no_grad():
            for (clean, noisy, labels, this_noise_type) in tqdm.tqdm(
                    val_generators[n_type], desc='Validation {}'.format(n_type)):
                clean, noisy = clean.cuda(), noisy.cuda()
                pred = model(noisy)
                l = val_losses['val_' + n_type](pred, clean)
                res_dic['val_' + n_type]['acc'] += l.tolist()

            cometml_report.plot_denoised_images(
                clean=clean.detach().cpu().numpy(),
                noisy=noisy.detach().cpu().numpy(),
                pred=pred.detach().cpu().numpy(),
                experiment=experiment, step=i, losses=l.tolist(),
                labels=labels.tolist(), noise_type=n_type,
                num_images=hparams['num_saved_images'])

    res_dic = cometml_report.report_losses_mean_and_std(
        res_dic, experiment, i, i)

    for loss_name in res_dic:
        res_dic[loss_name]['acc'] = []
    pprint(res_dic)