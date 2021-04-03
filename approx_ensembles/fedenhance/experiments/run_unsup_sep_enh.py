"""!
@brief Running an experiment with the improved version of SuDoRmRf on
universal source separation with multiple sources.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import sys

from approx_ensembles.fedenhance.__config__ import API_KEY
from comet_ml import Experiment, OfflineExperiment

import torch
from torch.nn import functional as F
from tqdm import tqdm
from pprint import pprint
import approx_ensembles.fedenhance.experiments.utils.improved_cmd_args_parser\
    as parser
import approx_ensembles.fedenhance.experiments.utils.mixture_consistency as mixture_consistency
import approx_ensembles.fedenhance.experiments.utils.dataset_setup as dataset_setup
import approx_ensembles.fedenhance.losses.sisdr as sisdr_lib
import approx_ensembles.fedenhance.losses.snr as snr_lib
import approx_ensembles.fedenhance.losses.sdr as sdr_lib
import approx_ensembles.fedenhance.models.groupcomm_sudormrf_v2 as \
    groupcomm_sudormrf_v2
import approx_ensembles.fedenhance.experiments.utils.cometml_loss_report as cometml_report
import approx_ensembles.fedenhance.experiments.utils.cometml_log_audio as cometml_audio_logger
import approx_ensembles.fedenhance.experiments.utils.log_audio as offline_audio_logger

# torch.backends.cudnn.enabled = False
args = parser.get_args()
hparams = vars(args)
generators = dataset_setup.mixit_setup(hparams)

if hparams['cometml_log_audio']:
    audio_loggers = {
        'separation': cometml_audio_logger.AudioLogger(
            fs=hparams["fs"], bs=1, n_sources=2),
        'enhancement': cometml_audio_logger.AudioLogger(
            fs=hparams["fs"], bs=1, n_sources=2)
    }

# offline_savedir = os.path.join('/home/thymios/offline_exps',
#                                hparams["project_name"],
#                                '_'.join(hparams['cometml_tags']))
# if not os.path.exists(offline_savedir):
#     os.makedirs(offline_savedir)
# audio_logger = offline_audio_logger.AudioLogger(dirpath=offline_savedir,
#     fs=hparams["fs"], bs=hparams["batch_size"], n_sources=4)

experiment = Experiment(API_KEY, project_name=hparams['project_name'])
experiment.log_parameters(hparams)
experiment_name = '_'.join(hparams['cometml_tags'])
for tag in hparams['cometml_tags']:
    experiment.add_tag(tag)
if hparams['experiment_name'] is not None:
    experiment.set_name(hparams['experiment_name'])
else:
    experiment.set_name(experiment_name)

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
    [cad for cad in hparams['cuda_available_devices']])

sup_back_loss_tr_loss_name, sup_back_loss_tr_loss = (
    'tr_back_loss_SNR',
    snr_lib.FixedMixIT2Sources2NoisesSNRwithZeroRefs(
        supervised=True,
        zero_mean=False,
        backward_loss=True,
        sources_weight=0.8,
        inactivity_threshold=-40.)
)

unsup_back_loss_tr_loss_name, unsup_back_loss_tr_loss = (
    'tr_back_loss_SNR',
    snr_lib.FixedMixIT2Sources2NoisesSNRwithZeroRefs(
        supervised=False,
        zero_mean=False,
        backward_loss=True,
        sources_weight=0.8,
        inactivity_threshold=-40.)
)

val_losses = {}
all_losses = []
for val_set in [x for x in generators if not x == 'train']:
    val_losses[val_set] = {}
    for task in ['separation', 'enhancement']:
        metric_name = 'SISDRi_{}'.format(task)
        all_losses.append(val_set + '_{}'.format(metric_name))
        if task == 'separation':
            n_estimated_sources = 2
            n_actual_sources = 2
        elif task == 'enhancement':
            n_estimated_sources = 2
            n_actual_sources = 1
        else:
            raise ValueError(f'Task {task} is invalid!')
        val_losses[val_set][val_set + '_{}'.format(metric_name)] = \
            sisdr_lib.StabilizedPermInvSISDRMetric(
                zero_mean=True,
                single_source=False,
                n_estimated_sources=n_estimated_sources,
                n_actual_sources=n_actual_sources,
                backward_loss=False,
                improvement=True,
                return_individual_results=True)
all_losses.append(unsup_back_loss_tr_loss_name)
all_losses.append(sup_back_loss_tr_loss_name)

if hparams['model_type'] == 'sudo_groupcomm_v2':
    model = groupcomm_sudormrf_v2.GroupCommSudoRmRf(
        in_audio_channels=1,
        out_channels=hparams['out_channels'],
        in_channels=hparams['in_channels'],
        num_blocks=hparams['num_blocks'],
        upsampling_depth=hparams['upsampling_depth'],
        enc_kernel_size=hparams['enc_kernel_size'],
        enc_num_basis=hparams['enc_num_basis'],
        num_sources=4,
        group_size=hparams['group_size'])
else:
    raise ValueError('Invalid model: {}.'.format(hparams['model_type']))

numparams = 0
for f in model.parameters():
    if f.requires_grad:
        numparams += f.numel()
experiment.log_parameter('Parameters', numparams)
print('Trainable Parameters: {}'.format(numparams))

model = torch.nn.DataParallel(model).cuda()
opt = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
# opt = torch.optim.SGD(model.parameters(), lr=hparams['learning_rate'])
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer=opt, mode='max', factor=1. / hparams['divide_lr_by'],
#     patience=hparams['patience'], verbose=True)


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


tr_step = 0
val_step = 0
prev_epoch_val_loss = 0.
for i in range(hparams['n_global_epochs']):
    res_dic = {}
    for loss_name in all_losses:
        res_dic[loss_name] = {'mean': 0., 'std': 0., 'median': 0., 'acc': []}
    print("Single node FedEnhance Sudo-RM-RF: {} - {} || Epoch: {}/{}".format(
        experiment.get_key(), experiment.get_tags(), i+1,
        hparams['n_global_epochs']))
    model.train()

    sum_loss = 0.
    train_tqdm_gen = tqdm(generators['train'], desc='Training')
    for cnt, data in enumerate(train_tqdm_gen):
        opt.zero_grad()
        # data: <speech> (batch, time_samples), <noise> (batch, time_samples)
        speech_wavs, noise_wavs = data
        input_active_speakers = torch.reshape(speech_wavs,
                                              [hparams['batch_size'], 2, -1])
        input_noises = torch.reshape(noise_wavs,
                                     [hparams['batch_size'], 2, -1])
        # Create a mask for zeroing out the second mixture.
        p_single_mix = 0.5
        zero_out_mask = (
                torch.rand([hparams['batch_size'], 1]) > p_single_mix).to(
            torch.float32)
        input_active_speakers[:, 1] = input_active_speakers[:, 1] * zero_out_mask
        input_noises[:, 1] = input_noises[:, 1] * zero_out_mask

        input_active_speakers = input_active_speakers.cuda()
        input_noises = input_noises.cuda()

        input_mom = input_active_speakers.sum(1, keepdim=True) + input_noises.sum(1, keepdim=True)
        input_mom = input_mom.cuda()

        input_mix_std = input_mom.std(-1, keepdim=True)
        input_mix_mean = input_mom.mean(-1, keepdim=True)
        input_mom = (input_mom - input_mix_mean) / (input_mix_std + 1e-9)

        rec_sources_wavs = model(input_mom)
        rec_sources_wavs = (rec_sources_wavs * input_mix_std) + input_mix_mean
        rec_sources_wavs = mixture_consistency.apply(rec_sources_wavs,
                                                     input_mom)

        # Percentage of supervised data.
        if i > 10:
            p_supervised = 0.
        else:
            p_supervised = 1.
        if (torch.rand([1]) < p_supervised).item():
            l = sup_back_loss_tr_loss(rec_sources_wavs, input_active_speakers,
                                      input_noises, input_mom)
        else:
            l = unsup_back_loss_tr_loss(
                rec_sources_wavs, input_active_speakers,
                input_noises, input_mom)
        l.backward()

        if hparams['clip_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           hparams['clip_grad_norm'])

        opt.step()
        sum_loss += l.detach().item()
        train_tqdm_gen.set_description(
            "Training, Running Avg Loss: {}".format(sum_loss / (cnt + 1)))

    if hparams['patience'] > 0:
        if tr_step % hparams['patience'] == 0:
            new_lr = (hparams['learning_rate']
                      / (hparams['divide_lr_by'] ** (tr_step // hparams['patience'])))
            print('Reducing Learning rate to: {}'.format(new_lr))
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr
    tr_step += 1

    for val_set in [x for x in generators if not x == 'train']:
        if generators[val_set] is None or val_set == 'test':
            continue
        model.eval()
        with torch.no_grad():
            for data in tqdm(generators[val_set],
                             desc='Validation on {}'.format(val_set)):
                speech_wavs, noise_wavs = data
                input_active_speakers = torch.reshape(
                    speech_wavs, [hparams['batch_size'], 2, -1])
                input_noises = torch.reshape(
                    noise_wavs, [hparams['batch_size'], 2, -1])
                # Create a mask for zeroing out the second mixture for
                # enhancement (half the examples).
                zero_out_mask = torch.ones([hparams['batch_size'], 1],
                                           dtype=torch.float32)
                zero_out_mask[hparams['batch_size']//2:] = 0.
                input_active_speakers[:, 1] *= zero_out_mask
                input_noises[:, 1] *= zero_out_mask

                input_active_speakers = input_active_speakers.cuda()
                input_noises = input_noises.cuda()

                input_mom = input_active_speakers.sum(1, keepdim=True) + input_noises.sum(1, keepdim=True)
                input_mom = input_mom.cuda()

                input_mix_std = input_mom.std(-1, keepdim=True)
                input_mix_mean = input_mom.mean(-1, keepdim=True)
                input_mom = (input_mom - input_mix_mean) / (input_mix_std + 1e-9)

                rec_sources_wavs = model(input_mom)
                rec_sources_wavs = (rec_sources_wavs * input_mix_std) + input_mix_mean
                rec_sources_wavs = mixture_consistency.apply(rec_sources_wavs,
                                                             input_mom)

                for loss_name, loss_func in val_losses[val_set].items():
                    if 'separation' in loss_name:
                        l, best_perm = loss_func(
                            rec_sources_wavs[:hparams['batch_size']//2, :2],
                            input_active_speakers[:hparams['batch_size']//2],
                            return_best_permutation=True,
                            initial_mixtures=input_mom[:hparams['batch_size']//2])
                        res_dic[loss_name]['acc'] += filter(lambda v: v == v,
                                                            l.tolist())
                    else:
                        l, best_perm = loss_func(
                            rec_sources_wavs[hparams['batch_size'] // 2:, :2],
                            input_active_speakers[
                            hparams['batch_size'] // 2:, 0:1],
                            return_best_permutation=True,
                            initial_mixtures=input_mom[hparams['batch_size']//2:])
                        res_dic[loss_name]['acc'] += filter(lambda v: v == v,
                                                            l.tolist())

            if hparams['cometml_log_audio']:
                audio_loggers['separation'].log_batch(
                    rec_sources_wavs[0:1, :2],
                    input_active_speakers[0:1],
                    input_mom[0:1],
                    experiment, step=val_step, tag=val_set+'_sep_speakers',
                    overwrite=True)
                audio_loggers['separation'].log_batch(
                    rec_sources_wavs[0:1, :2],
                    input_noises[0:1],
                    input_mom[0:1],
                    experiment, step=val_step, tag=val_set + '_sep_noises',
                    overwrite=True)
                audio_loggers['enhancement'].log_batch(
                    rec_sources_wavs[-1:, :2],
                    input_active_speakers[-1:],
                    input_mom[-1:],
                    experiment, step=val_step, tag=val_set+'_enh_speakers',
                    overwrite=True)
                audio_loggers['enhancement'].log_batch(
                    rec_sources_wavs[-1:, :2],
                    input_noises[-1:],
                    input_mom[-1:],
                    experiment, step=val_step, tag=val_set + '_enh_noises',
                    overwrite=True)


    val_step += 1

    res_dic = cometml_report.report_losses_mean_and_std(res_dic,
                                                        experiment,
                                                        tr_step,
                                                        val_step)

    for loss_name in res_dic:
        res_dic[loss_name]['acc'] = []
    pprint(res_dic)