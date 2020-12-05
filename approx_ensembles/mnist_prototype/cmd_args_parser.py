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
    parser.add_argument("-n", "--noise_type", type=str, nargs='+',
                        help="Noise type added to the image",
                        default=['salt_and_pepper', 'gaussian', 'speckle'],
                        choices=['none', 'salt_and_pepper',
                                 'gaussian', 'speckle'])
    # Seed for controlling the initial network configuration
    parser.add_argument("--seed", type=int,
                        help="""The seed for initialization""", default=7)

    # device params
    parser.add_argument("-cad", "--cuda_available_devices", type=str,
                        nargs="+",
                        help="""A list of Cuda IDs that would be 
                            available for running this experiment""",
                        default=['0'],
                        choices=['0', '1', '2', '3'])

    # CometML experiment parameters
    parser.add_argument("-tags", "--cometml_tags", type=str,
                        nargs="+", help="""A list of tags for the cometml 
                        experiment.""",
                        default=[])
    parser.add_argument("--experiment_name", type=str,
                        help="""Name of current experiment""",
                        default=None)
    parser.add_argument("--project_name", type=str,
                        help="""Name of current experiment""",
                        default="noisy_mnist_approx_ensembles")
    parser.add_argument("--ckpt_period", type=int,
                        help="""The number of epochs needed for saving a 
                        checkpoint""",
                        default=2)
    parser.add_argument("--num_saved_images", type=int,
                        help="""The number of saved image results for each 
                        noise type""",
                        default=5)

    # Evaluation only parameter
    parser.add_argument("--eval_model_name", type=str,
                        help="""Name of model or set of checkpoints saved""",
                        default='')

    # training params
    parser.add_argument("-bs", "--batch_size", type=int,
                        help="""The number of samples in each batch. 
                            Warning: Cannot be less than the number of 
                            the validation samples""", default=32)
    parser.add_argument("--n_jobs", type=int,
                        help="""The number of cpu workers for 
                            loading the data, etc.""", default=4)
    parser.add_argument("-nl", "--n_intermediate_layers", type=int,
                        help="""The number of intermediate layers in the 
                        model. If 0 then only 2 layers in total are used.""",
                        default=4)
    parser.add_argument("--n_epochs", type=int,
                        help="""The number of epochs that the 
                        experiment should run""", default=5000)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="""Initial Learning rate""", default=2e-3)

    return parser.parse_args()
