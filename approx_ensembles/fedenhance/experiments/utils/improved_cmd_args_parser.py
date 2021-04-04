"""!
@brief FedEnhance experiment argument parser

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import argparse


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(
        description='Experiment Argument Parser')
    # ===============================================
    # Datasets arguments
    parser.add_argument("--audio_timelength", type=float,
                        help="""The timelength of the audio that you want 
                                to load in seconds.""",
                        default=4.)
    parser.add_argument("--normalize_audio", action='store_true',
                        help="""Normalize using mean and standard deviation 
                        before processing each audio file.""",
                        default=False)
    # ===============================================
    # Separation task arguments
    parser.add_argument("--available_speech_percentage", type=float,
                        help="""The percentage of available speech 
                        utterances assuming a 1:1 mapping with noises. The 
                        rest of the available speech files are going to be 
                        randomly returned as extra noise.""",
                        default=0.5)
    parser.add_argument("--min_num_sources", type=int,
                        help="""The minimum number of sources in a mixture.""",
                        default=1)
    parser.add_argument("--max_num_sources", type=int,
                        help="""The maximum number of sources in a mixture.""",
                        default=4)
    # ===============================================
    # Training params
    parser.add_argument("-bs", "--batch_size", type=int,
                        help="""The number of samples in each batch. 
                                Warning: Cannot be less than the number of 
                                the validation samples""", default=4)
    parser.add_argument("--n_global_epochs", type=int,
                        help="""The number of epochs that all the nodes will
                        be trained on.""", default=500)
    parser.add_argument("--n_local_epochs", type=int,
                        help="""The number of epochs that each node runs 
                        before communicating the update.""", default=1)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="""Initial Learning rate""", default=1e-3)
    parser.add_argument("--divide_lr_by", type=float,
                        help="""The factor that the learning rate 
                            would be divided by""", default=3.)
    parser.add_argument("--patience", type=int,
                        help="""Patience until reducing the learning rate .""",
                        default=5)
    parser.add_argument("--optimizer", type=str,
                        help="""The optimizer that you want to use""",
                        default="adam",
                        choices=['adam', 'radam'])
    parser.add_argument("--clip_grad_norm", type=float,
                        help="""The norm value which all gradients 
                            are going to be clipped, 0 means that no 
                            grads are going to be clipped""",
                        default=5.)
    parser.add_argument("-fs", type=int,
                        help="""Sampling rate of the audio.""", default=16000)
    parser.add_argument("--max_abs_snr", type=float,
                        help="""The maximum absolute value of the SNR of 
                                the mixtures.""", default=5.)
    # ===============================================
    # CometML experiment configuration arguments
    parser.add_argument("-tags", "--cometml_tags", type=str,
                        nargs="+", help="""A list of tags for the cometml 
                            experiment.""",
                        default=[])
    parser.add_argument("--experiment_name", type=str,
                        help="""Name of current experiment""",
                        default=None)
    parser.add_argument("--project_name", type=str,
                        help="""Name of current experiment""",
                        default="yolo_experiment")
    # ===============================================
    # Device params
    parser.add_argument("-cad", "--cuda_available_devices", type=str,
                        nargs="+",
                        help="""A list of Cuda IDs that would be 
                            available for running this experiment""",
                        default=['0'],
                        choices=['0', '1', '2', '3'])
    parser.add_argument("--n_jobs", type=int,
                        help="""The number of cpu workers for 
                                        loading the data, etc.""", default=4)
    # ===============================================
    # Local experiment logging
    parser.add_argument("-elp", "--experiment_logs_path", type=str,
                        help="""Path for logging experiment's audio.""",
                        default=None)
    parser.add_argument("-mlp", "--metrics_logs_path", type=str,
                        help="""Path for logging metrics.""",
                        default=None)
    # Cometml experiment logging
    parser.add_argument("--cometml_log_audio", action='store_true',
                        help="""Log audio online.""",
                        default=False)
    # ===============================================
    # Separation model (SuDO-RM-RF) params
    parser.add_argument("--out_channels", type=int,
                        help="The number of channels of the internal "
                             "representation outside the U-Blocks.",
                        default=128)
    parser.add_argument("--in_channels", type=int,
                        help="The number of channels of the internal "
                             "representation inside the U-Blocks.",
                        default=512)
    parser.add_argument("--num_blocks", type=int,
                        help="Number of the successive U-Blocks.",
                        default=8)
    parser.add_argument("--upsampling_depth", type=int,
                        help="Number of successive upsamplings and "
                             "effectively downsampling inside each U-Block. "
                             "The aggregation of all scales is performed by "
                             "addition.",
                        default=5)
    parser.add_argument("--group_size", type=int,
                        help="The number of individual computation groups "
                             "applied if group communication module is used.",
                        default=16)
    parser.add_argument("--enc_kernel_size", type=int,
                        help="The width of the encoder and decoder kernels.",
                        default=21)
    parser.add_argument("--enc_num_basis", type=int,
                        help="Number of the encoded basis representations.",
                        default=512)

    parser.add_argument("--model_type", type=str,
                        help="The type of model you would like to use.",
                        default='relu',
                        choices=['relu', 'softmax', 'groupcomm',
                                 'sudo_groupcomm_v2', 'causal'])
    return parser.parse_args()
