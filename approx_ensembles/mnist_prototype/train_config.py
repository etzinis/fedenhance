def default():
    params = {
        # train params
        'valid_noises': ['gaussian', 'salt_and_pepper', 'speckle'],

        # dir params
        'ckpt_dir': '../ckpts',
        'logs_dir': '../logs',

        # opt params
        'lr': 2e-3,
        'n_epochs': 20,
        'batch_size': 32,

        # validation params
        'val_period': 2000,
        'ckpt_period': 2000,
        'n_img_save' : 10,
    }

    return params
