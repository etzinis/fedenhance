"""!
@brief Runnign pretrainign on WHAM dataset for global model fetching.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import sys
import copy

from approx_ensembles.fedenhance.__config__ import API_KEY
from comet_ml import Experiment, OfflineExperiment

import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
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
import approx_ensembles.fedenhance.losses.mixit as mixit_lib
import approx_ensembles.fedenhance.losses.asteroid_sdr as asteroid_sdr_lib
import approx_ensembles.fedenhance.experiments.utils.cometml_loss_report as cometml_report
import approx_ensembles.fedenhance.experiments.utils.cometml_log_audio as cometml_audio_logger
import approx_ensembles.fedenhance.experiments.utils.log_audio as offline_audio_logger
from approx_ensembles.fedenhance.__config__ import FED_LOG_DIR

# torch.backends.cudnn.enabled = False
args = parser.get_args()
hparams = vars(args)
_, libri_fsd_val_generators = dataset_setup.enhancement_federated_setup(
    hparams)
pretrain_generators = \
    dataset_setup.pretrain_enhancement_single_node_setup(hparams)

if hparams['cometml_log_audio']:
    audio_loggers = {
        'enhancement': cometml_audio_logger.AudioLogger(
            fs=hparams["fs"], bs=1, n_sources=hparams['max_num_sources'])
    }

experiment = Experiment(API_KEY, project_name=hparams['project_name'])
experiment.log_parameters(hparams)
experiment_name = '_'.join(hparams['cometml_tags'])
log_dir = os.path.join(FED_LOG_DIR, experiment_name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
for tag in hparams['cometml_tags']:
    experiment.add_tag(tag)
if hparams['experiment_name'] is not None:
    experiment.set_name(hparams['experiment_name'])
else:
    experiment.set_name(experiment_name)

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
    [cad for cad in hparams['cuda_available_devices']])

sup_back_loss_tr_loss_name, sup_back_loss_tr_loss = (
    'tr_back_loss_sup_SNR',
    snr_lib.FixedMixIT1Source2NoisesSNRwithZeroRefs(
        supervised=True,
        zero_mean=False,
        backward_loss=True,
        inactivity_threshold=-40.)
)

unsup_back_loss_tr_loss_name, unsup_back_loss_tr_loss = (
    'tr_back_loss_asteroid_mixit_SISNR',
    mixit_lib.MixITLossWrapper(asteroid_sdr_lib.pairwise_neg_sisdr,
                               generalized=False)
)

supervised_ast_loss = asteroid_sdr_lib.PITLossWrapper(
    asteroid_sdr_lib.pairwise_neg_sisdr, pit_from='pw_mtx')


val_losses = {'test': {}, 'val': {}}
all_losses = []
val_generators_l = [('wham_pre', pretrain_generators),
                    ('', libri_fsd_val_generators)]
for gen_prefix, val_generators in val_generators_l:
    for val_set in [x for x in val_generators if not x == 'train']:
        for num_noises in [1, 2]:
            if len(gen_prefix) > 1:
                metric_name = '{}_SISDRi_enhancement_{}_noises'.format(
                    gen_prefix, num_noises)
            else:
                metric_name = 'SISDRi_enhancement_{}_noises'.format(
                    num_noises)
            all_losses.append(val_set + '_{}'.format(metric_name))
            n_estimated_sources = 1
            n_actual_sources = 1
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
        num_sources=hparams['max_num_sources'],
        group_size=hparams['group_size'])
else:
    raise ValueError('Invalid model: {}.'.format(hparams['model_type']))

numparams = 0
for f in model.parameters():
    if f.requires_grad:
        numparams += f.numel()
experiment.log_parameter('Parameters', numparams)
print('Trainable Parameters: {}'.format(numparams))

# global_model = torch.nn.DataParallel(global_model).cuda()
model = model.cuda()
opt = torch.optim.Adam(model.parameters(),
                       lr=hparams['learning_rate'])


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


def snr_loss(est, ref, eps=1e-9):
    error_element = est - ref
    error = torch.sum(error_element**2, dim=-1)
    power_ref = torch.sum(ref**2, dim=-1)
    return 10. * torch.log10(error + eps + 0.001 * power_ref)


def my_mixit(rec_sources_wavs, input_active_speakers, input_noises,
             input_mom, loss_func=snr_loss):
    ref_mix_1 = normalize_tensor_wav(
        input_active_speakers + input_noises[:, 0:1])
    ref_mix_2 = normalize_tensor_wav(input_noises[:, 1:2])

    er_00 = loss_func(rec_sources_wavs[:, 0:1] + rec_sources_wavs[:, 1:2], ref_mix_1)
    er_01 = loss_func(rec_sources_wavs[:, 2:3], ref_mix_2)

    er_10 = loss_func(rec_sources_wavs[:, 0:1] + rec_sources_wavs[:, 2:3], ref_mix_1)
    er_11 = loss_func(rec_sources_wavs[:, 1:2], ref_mix_2)

    errors = torch.cat([er_00 + er_01,
                        er_10 + er_11], 1)
    return torch.mean(torch.min(errors, 1)[0])


def sup_sisdr(rec_sources_wavs, input_active_speakers, input_noises,
              input_mom, use_activity_masks):
    ref_speech = normalize_tensor_wav(input_active_speakers)
    ref_noises = normalize_tensor_wav(input_noises)

    if use_activity_masks:
        ref_speech_powers = torch.sum(input_active_speakers ** 2, dim=-1,
                                      keepdim=True)
        input_mom_powers = torch.sum(input_mom ** 2, dim=-1, keepdim=True)
        mixtures_input_snr = 10. * torch.log10(
            ref_speech_powers / (input_mom_powers + 1e-9))
        ref_speech_activity_mask = mixtures_input_snr.ge(0.001)

        ref_noise_powers = torch.sum(input_noises ** 2, dim=-1,
                                     keepdim=True)
        mixtures_input_snr = 10. * torch.log10(
            ref_noise_powers / (input_mom_powers + 1e-9))
        ref_noise_activity_mask = mixtures_input_snr.ge(0.001)

        speech_error = ref_speech_activity_mask * torch.clamp(
            asteroid_sdr_lib.pairwise_neg_sisdr(
                rec_sources_wavs[:, 0:1], ref_speech), min=-50., max=50.)

        noise_error = ref_noise_activity_mask * torch.clamp(
            supervised_ast_loss(
                rec_sources_wavs[:, 1:], ref_noises), min=-50., max=50.)
    else:
        speech_error = torch.clamp(
            asteroid_sdr_lib.pairwise_neg_sisdr(
                rec_sources_wavs[:, 0:1], ref_speech), min=-50.,
            max=50.)

        noise_error = torch.clamp(
            supervised_ast_loss(
                rec_sources_wavs[:, 1:], ref_noises), min=-50., max=50.)

    return speech_error.mean() + noise_error


def ast_mixit(rec_sources_wavs, input_active_speakers, input_noises, input_mom):
    ref_mix_1 = normalize_tensor_wav(
        input_active_speakers + input_noises[:, 0:1])
    ref_mix_2 = normalize_tensor_wav(input_noises[:, 1:2])

    ref_mix1_powers = torch.sum(
        (input_active_speakers + input_noises[:, 0:1]) ** 2,
        dim=-1, keepdim=True)
    input_mom_powers = torch.sum(input_mom ** 2, dim=-1, keepdim=True)
    mixtures_input_snr = 10. * torch.log10(
        ref_mix1_powers / (input_mom_powers + 1e-9))
    ref_mix1_activity_mask = mixtures_input_snr.ge(0.001)

    ref_mix2_powers = torch.sum(input_noises[:, 1:2]**2, dim=-1, keepdim=True)
    mixtures_input_snr = 10. * torch.log10(
        ref_mix2_powers / (input_mom_powers + 1e-9))
    ref_mix2_activity_mask = mixtures_input_snr.ge(0.001)

    er_00 = ref_mix1_activity_mask * torch.clamp(
        asteroid_sdr_lib.pairwise_neg_sisdr(
            rec_sources_wavs[:, 0:1] + rec_sources_wavs[:, 1:2], ref_mix_1),
        min=-50., max=50.)
    er_01 = ref_mix2_activity_mask * torch.clamp(
        asteroid_sdr_lib.pairwise_neg_sisdr(
            rec_sources_wavs[:, 2:3], ref_mix_2), min=-50., max=50.)

    er_10 = ref_mix1_activity_mask * torch.clamp(
        asteroid_sdr_lib.pairwise_neg_sisdr(
            rec_sources_wavs[:, 0:1] + rec_sources_wavs[:, 2:3], ref_mix_1),
        min=-50., max=50.)
    er_11 = ref_mix2_activity_mask * torch.clamp(
        asteroid_sdr_lib.pairwise_neg_sisdr(
            rec_sources_wavs[:, 1:2], ref_mix_2), min=-50., max=50.)

    errors = torch.cat([er_00 + er_01,
                        er_10 + er_11], 1)
    return torch.mean(torch.min(errors, 1)[0])

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

tr_step = 0
val_step = 0
prev_epoch_val_loss = 0.
for i in range(hparams['n_global_epochs']):
    res_dic = {}
    for loss_name in all_losses:
        res_dic[loss_name] = {'mean': 0., 'std': 0., 'median': 0., 'acc': []}
    print("Pre-train Sudo-RM-RF: {} - {} || Epoch: {}/{}".format(
        experiment.get_key(), experiment.get_tags(), i+1,
        hparams['n_global_epochs']))
    model.train()
    sum_loss = 0.

    train_tqdm_gen = tqdm(pretrain_generators['train'], desc='Training')
    for cnt, data in enumerate(train_tqdm_gen):
        opt.zero_grad()
        # data: <speech> (batch, time_samples), <noise> (batch, time_samples)
        input_active_speakers, noise_wavs, extra_noise_wavs = data
        input_active_speakers = input_active_speakers.unsqueeze(1)
        input_noises = torch.stack([noise_wavs, extra_noise_wavs], 1)

        # Create a mask for zeroing out the second noise.
        zero_out_mask = (torch.rand([hparams['batch_size'], 1]) >
                         hparams['p_single_mix']).to(torch.float32)

        # Zero out mixture equal probability to zero out a noise mixture or
        # the mixture also containing the speaker.
        if (torch.rand([1]) < 0.5).item():
            input_noises[:, 1] = input_noises[:, 1] * zero_out_mask
        else:
            input_noises[:, 0] = input_noises[:, 0] * zero_out_mask
            input_active_speakers[:, 0] = input_active_speakers[:,
                                          0] * zero_out_mask

        input_active_speakers = input_active_speakers.cuda()
        input_noises = input_noises.cuda()

        input_mom = input_active_speakers.sum(1, keepdim=True) + \
                    input_noises.sum(1, keepdim=True)
        input_mom = input_mom.cuda()

        input_mix_std = input_mom.std(-1, keepdim=True)
        input_mix_mean = input_mom.mean(-1, keepdim=True)
        input_mom = (input_mom - input_mix_mean) / (
                    input_mix_std + 1e-9)

        rec_sources_wavs = model(input_mom)
        # rec_sources_wavs = (rec_sources_wavs * input_mix_std) + input_mix_mean
        rec_sources_wavs = mixture_consistency.apply(rec_sources_wavs,
                                                     input_mom)

        # Percentage of supervised data.
        p_supervised = hparams['p_supervised']
        if (torch.rand([1]) < p_supervised).item():
            l = sup_sisdr(rec_sources_wavs, input_active_speakers,
                          input_noises, input_mom,
                          use_activity_masks=False)
        else:
            l = ast_mixit(rec_sources_wavs, input_active_speakers,
                          input_noises, input_mom)
        l.backward()

        if hparams['clip_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           hparams['clip_grad_norm'])

        opt.step()
        sum_loss += l.detach().item()
        train_tqdm_gen.set_description(
            "Training, Running Avg Loss: {}".format(
                sum_loss / (cnt + 1)))

    if hparams['patience'] > 0:
        if tr_step % hparams['patience'] == 0:
            new_lr = (hparams['learning_rate']
                      / (hparams['divide_lr_by'] ** (
                                tr_step // hparams['patience'])))
            print('Reducing Learning rate to: {}'.format(new_lr))
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr
    tr_step += 1

    for gen_prefix, val_generators in val_generators_l:
        for val_set in [x for x in val_generators if not x == 'train']:
            if val_generators[val_set] is None:
                continue
            model.eval()
            with torch.no_grad():
                for data in tqdm(val_generators[val_set],
                                 desc=f'Validation on {gen_prefix}, '
                                      f'{val_set}'):
                    input_active_speakers, noise_wavs, extra_noise_wavs = data
                    input_active_speakers = input_active_speakers.unsqueeze(1)
                    input_noises = torch.stack([noise_wavs, extra_noise_wavs], 1)
                    # Create a mask for zeroing out the second mixture for
                    # enhancement with 1 source (half the examples).
                    zero_out_mask = torch.ones([hparams['batch_size'], 1],
                                               dtype=torch.float32)
                    zero_out_mask[hparams['batch_size']//2:] = 0.
                    input_noises[:, 1] *= zero_out_mask

                    input_active_speakers = input_active_speakers.cuda()
                    input_noises = input_noises.cuda()

                    input_mom = input_active_speakers.sum(1, keepdim=True) + input_noises.sum(1, keepdim=True)
                    input_mom = input_mom.cuda()

                    input_mix_std = input_mom.std(-1, keepdim=True)
                    input_mix_mean = input_mom.mean(-1, keepdim=True)
                    input_mom = (input_mom - input_mix_mean) / (input_mix_std + 1e-9)

                    rec_sources_wavs = model(input_mom)
                    # rec_sources_wavs = (rec_sources_wavs * input_mix_std) + input_mix_mean
                    rec_sources_wavs = mixture_consistency.apply(rec_sources_wavs,
                                                                 input_mom)
                    for loss_name, loss_func in val_losses[val_set].items():
                        if '2_noises' in loss_name:
                            l, best_perm = loss_func(
                                rec_sources_wavs[:hparams['batch_size']//2, :1],
                                input_active_speakers[:hparams['batch_size']//2],
                                return_best_permutation=True,
                                initial_mixtures=input_mom[:hparams['batch_size']//2])
                            res_dic[loss_name]['acc'] += filter(lambda v: v == v,
                                                                l.tolist())
                        else:
                            l, best_perm = loss_func(
                                rec_sources_wavs[hparams['batch_size'] // 2:, :1],
                                input_active_speakers[hparams['batch_size'] // 2:],
                                return_best_permutation=True,
                                initial_mixtures=input_mom[hparams['batch_size']//2:])
                            res_dic[loss_name]['acc'] += filter(lambda v: v == v,
                                                                l.tolist())

                if hparams['cometml_log_audio']:
                    ref_sources = torch.cat([input_active_speakers,
                                             input_noises], 1)
                    audio_loggers['enhancement'].log_batch(
                        rec_sources_wavs[0:1],
                        ref_sources[0:1, :hparams['max_num_sources']],
                        input_mom[0:1],
                        experiment, step=val_step,
                        tag=val_set+f'_{gen_prefix}_enh_2_noises',
                        overwrite=True)
                    audio_loggers['enhancement'].log_batch(
                        rec_sources_wavs[-1:],
                        ref_sources[-1:, :hparams['max_num_sources']],
                        input_mom[-1:],
                        experiment, step=val_step,
                        tag=val_set + f'_{gen_prefix}_enh_1_noise',
                        overwrite=True)

    val_step += 1

    # Save all models and decide later.
    torch.save(model.state_dict(),
               os.path.join(log_dir, f'model_epoch_{tr_step}'))

    res_dic = cometml_report.report_losses_mean_and_std(res_dic,
                                                        experiment,
                                                        tr_step,
                                                        val_step)

    for loss_name in res_dic:
        res_dic[loss_name]['acc'] = []
    pprint(res_dic)