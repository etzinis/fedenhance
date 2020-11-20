import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os

def load_log(params, batch):

    # get the saved data
    run_dir = os.path.join(params['logs_dir'], params['run_name'])
    preds_name = os.path.join(run_dir, '{}.pkl'.format(batch))
    with open(preds_name, 'rb') as f:
        data_dict = pkl.load(f)

    # make a dummy dir to look at things
    save_dir = '../temp'
    os.makedirs(save_dir, exist_ok=True)


    # look at it
    print('Mean Error : {}'.format(np.array(data_dict['losses']).mean()))

    clean = data_dict['clean']
    noisy = data_dict['noisy']
    pred = data_dict['pred']
    for i in range(3):
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(clean[i][0])
        ax[0].set_title('Clean')

        ax[1].imshow(noisy[i][0])
        ax[1].set_title('Noisy')

        ax[2].imshow(pred[i][0])
        ax[2].set_title('Denoised')


        plt.suptitle('Loss {} Noise {} Digit {}'.format(data_dict['losses'][i], 
                                                        data_dict['noise_type'][i], 
                                                        data_dict['labels'][i]))

        plt.savefig(save_dir + '/result_{}.png'.format(i))

if __name__ == "__main__":

    import argparse
    import datetime
    from train_config import *

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="")
    parser.add_argument('--date', default="")
    parser.add_argument('--num', default="")

    args = vars(parser.parse_args())
    params = globals()[args['cfg']]()

    run_name = '{}_{}'.format(args['cfg'], args['date'])
    params['run_name'] = run_name

    load_log(params, args['num'])
