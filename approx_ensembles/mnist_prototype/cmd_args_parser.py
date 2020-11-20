"""!
@brief Experiment Argument Parser

@author Jonah Casebeer {jonahmc2@illinois.edu}
@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import argparse


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(
        description='CometML Experiment Argument Parser')
    # types of noises added to images
    parser.add_argument("-n", "--train_noise", type=str, nargs='+',
                        help="Training noise",
                        default=None,
                        choices=['none', 'salt_n_pepper',
                                 'gaussian', 'speckle'])
    # device params
    parser.add_argument("-cad", "--cuda_available_devices", type=str,
                        nargs="+",
                        help="""A list of Cuda IDs that would be 
                            available for running this experiment""",
                        default=['0'],
                        choices=['0', '1', '2', '3'])

    parser.add_argument("--val", type=str, nargs='+',
                        help="Validation dataset",
                        default=None,
                        choices=['WSJ2MIX8K', 'WSJ2MIX8KPAD',
                                 'TIMITMF8K', 'WSJ2MIX8KNORMPAD',
                                 'AUGMENTED_WSJMIX', 'AUGMENTED_ESC50'])
    parser.add_argument("--train_val", type=str, nargs='+',
                        help="Validating on the training dataset",
                        default=None,
                        choices=['WSJ2MIX8K', 'WSJ2MIX8KPAD',
                                 'TIMITMF8K', 'WSJ2MIX8KNORMPAD',
                                 'AUGMENTED_WSJMIX', 'AUGMENTED_ESC50'])

    # Logging experiments
    parser.add_argument("-elp", "--experiment_logs_path", type=str,
                        help="""Path for experiment's checkpoints.""",
                        default=None)
    parser.add_argument("-mlp", "--metrics_logs_path", type=str,
                        help="""Path for logging metrics.""",
                        default=None)
    parser.add_argument('--log_audio', action='store_true', default=False)

    parser.add_argument("--n_train", type=int,
                        help="""Reduce the number of training 
                            samples to this number.""", default=None)
    parser.add_argument("--n_val", type=int,
                        help="""Reduce the number of evaluation 
                            samples to this number.""", default=None)
    parser.add_argument("-ri", "--return_items", type=str, nargs='+',
                        help="""A list of elements that this 
                        dataloader should return. See available 
                        choices which are based on the saved data 
                        names which are available. There is no type 
                        checking in this return argument.""",
                        default=['mixture_wav', 'clean_sources_wavs'],
                        choices=['mixture_wav',
                                 'clean_sources_wavs',
                                 'mixture_wav_norm', 'wav', 'class_id',
                                 'clean_sources_wavs_norm'])
    parser.add_argument("-tags", "--cometml_tags", type=str,
                        nargs="+", help="""A list of tags for the cometml 
                        experiment.""",
                        default=[])
    parser.add_argument("--experiment_name", type=str,
                        help="""Name of current experiment""",
                        default=None)
    parser.add_argument("--project_name", type=str,
                        help="""Name of current experiment""",
                        default="first_wsj02mix")

    # Augmented Dataset parameters
    parser.add_argument("-priors", "--datasets_priors", type=float, nargs='+',
                        help="The prior probability of finding a sample from "
                             "each given dataset. The length of this list "
                             "must be equal to the number of dataset paths "
                             "given above. The sum of this list must add up "
                             "to 1.",
                        default=[1])
    parser.add_argument("-fs", type=float,
                        help="""Sampling rate of the audio.""", default=8000.)
    parser.add_argument("--selected_timelength", type=float,
                        help="""The timelength of the sources that you want 
                                to load in seconds.""",
                        default=4.)
    parser.add_argument("--max_abs_snr", type=float,
                        help="""The maximum absolute value of the SNR of 
                                the mixtures.""", default=2.5)

    # training params
    parser.add_argument("-bs", "--batch_size", type=int,
                        help="""The number of samples in each batch. 
                            Warning: Cannot be less than the number of 
                            the validation samples""", default=4)
    parser.add_argument("--n_jobs", type=int,
                        help="""The number of cpu workers for 
                            loading the data, etc.""", default=4)
    parser.add_argument("--n_epochs", type=int,
                        help="""The number of epochs that the 
                        experiment should run""", default=50)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="""Initial Learning rate""", default=1e-2)

    return parser.parse_args()
