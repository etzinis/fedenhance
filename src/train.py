import numpy as np
import tqdm
import os
import pickle as pkl

import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F

from net import MNISTnet
from dataloader import NoisyMNIST

def run_validation(net, params, n_batches_seen):
    test_dataset = NoisyMNIST(params, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    net.eval()

    labels_list = []
    loss_list = []
    noise_list = []

    noisy_list = []
    denoised_list = []
    clean_list = []
    
    with torch.no_grad():
        for i, (clean, noisy, label, noise_type) in enumerate(tqdm.tqdm(test_loader)):
            clean, noisy = clean.cuda(), noisy.cuda()

            pred = net(noisy)
            loss = F.l1_loss(pred, clean)

            labels_list.append(label.item())
            loss_list.append(loss.item())
            noise_list.append(noise_type)

            if i < params['n_img_save']: 
                noisy_list.append(noisy.cpu().numpy()[0])
                denoised_list.append(pred.cpu().numpy()[0])
                clean_list.append(clean.cpu().numpy()[0])
    
    data_dict = {
        'labels': labels_list,
        'noise_type' : noise_list,
        'losses': loss_list,
        
        'noisy' : noisy_list,
        'clean' : clean_list,
        'pred' : denoised_list,
    }

    save_val_data(data_dict, params, n_batches_seen)
    net.train()

    return np.array(loss_list).mean()


def save_val_data(data_dict, params, n_batches_seen):
    run_dir = os.path.join(params['logs_dir'], params['run_name'])
    os.makedirs(run_dir, exist_ok=True)

    preds_name = os.path.join(run_dir, '{}.pkl'.format(n_batches_seen))

    with open(preds_name, 'wb') as f:
        pkl.dump(data_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

def save_model(net, params, n_batches_seen):
    run_dir = os.path.join(params['ckpt_dir'], params['run_name'])
    os.makedirs(run_dir, exist_ok=True)

    ckpt_name = os.path.join(run_dir, '{}.pt'.format(n_batches_seen))
    torch.save(net.state_dict(), ckpt_name)

def train(params, seed):
    # make the training deterministic with the seed
    torch.backends.cudnn.enabled = False
    torch.manual_seed(seed)

    # init
    net = MNISTnet(params)
    opt = torch.optim.SGD(net.parameters(), lr=params['lr'])
    net.cuda()

    n_batches_seen = 0
    val_loss = -1

    for _ in range(params['n_epochs']):
        cur_dataset = NoisyMNIST(params, train=True)
        cur_loader = DataLoader(cur_dataset, batch_size=params['batch_size'], shuffle=True)
        pbar_cur_loader = tqdm.tqdm(cur_loader)

        for (clean, noisy, _, _) in pbar_cur_loader:
            net.train()

            clean, noisy = clean.cuda(), noisy.cuda()

            pred = net(noisy)
            loss = F.l1_loss(pred, clean)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar_cur_loader.set_description("cur_L: {:10.2f}, val_L: {:10.2f}".format(loss.item(), val_loss))

            if n_batches_seen % params['val_period'] == 0:
                val_loss = run_validation(net, params, n_batches_seen)

            if n_batches_seen % params['ckpt_period'] == 0:
                save_model(net, params, n_batches_seen)

            n_batches_seen += 1

if __name__ == "__main__":
    import argparse
    import datetime
    from train_config import *

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="")
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--seed', type=int)

    args = vars(parser.parse_args())
    params = globals()[args['cfg']]()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu'])
    run_name = '{}_{}_{}'.format(args['cfg'], 
                                    datetime.date.today(), 
                                    datetime.datetime.now().strftime('%H%M%S'))
    params['run_name'] = run_name

    train(params, args['seed'])
