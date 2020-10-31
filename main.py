import os
from utils.utils import write_print, mkdir
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from datetime import datetime
import zipfile
import torch
import numpy as np


SAVE_NAME_FORMAT = 'files_{}.{}'


def zip_directory(path, zip_file):
    """Stores all py and cfg project files inside a zip file

    [description]

    Arguments:
        path {string} -- current path
        zip_file {zipfile.ZipFile} -- zip file to contain the project files
    """
    files = os.listdir(path)
    for file in files:
        if file.endswith('.py') or file.endswith('cfg'):
            zip_file.write(os.path.join(path, file))
            if file.endswith('cfg'):
                os.remove(file)


def save_config(config):
    """saves the configuration of the experiment

    [description]

    Arguments:
        config {dict} -- contains argument and its value

    Returns:
        string -- version based on the current time
    """
    version = str(datetime.now()).replace(':', '_')
    cfg_name = SAVE_NAME_FORMAT.format(version, 'cfg')
    with open(cfg_name, 'w') as f:
        for k, v in config.items():
            f.write('{}: {}\n'.format(str(k), str(v)))

    zip_name = SAVE_NAME_FORMAT.format(version, 'zip')
    zip_file = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    zip_directory('.', zip_file)
    zip_file.close()

    return version


def string_to_boolean(v):
    """Converts string to boolean

    [description]

    Arguments:
        v {string} -- string representation of a boolean values;
        must be true or false

    Returns:
        boolean -- boolean true or false
    """
    return v.lower() in ('true')


def main(version, config):
    # for fast training
    cudnn.benchmark = True

    # create directories if not exist
    mkdir(config.log_path)

    if config.mode == 'train':
        temp_save_path = os.path.join(config.model_save_path, version)
        mkdir(temp_save_path)

        data_loader = get_loader(config.data_path + config.train_data_path,
                                 config.train_x_key, config.train_y_key,
                                 config.batch_size, config.mode)
        solver = Solver(version, data_loader, vars(config))
        solver.train()
    elif config.mode == 'test':
        data_loader = get_loader(config.data_path + config.test_data_path,
                                 config.test_x_key, config.test_y_key,
                                 config.batch_size, config.mode)
        solver = Solver(version, data_loader, vars(config))
        solver.test()


if __name__ == '__main__':
    torch.set_printoptions(threshold=np.nan)
    parser = argparse.ArgumentParser()

    # dataset info
    parser.add_argument('--input_channels', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('--class_count', type=int, default=256,
                        help='Number of classes in dataset')

    # training settings
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=74,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--pretrained_model', type=str,
                        default=None,
                        help='Pre-trained model')
    parser.add_argument('--config', type=str, default='D',
                        choices=['A', 'B', 'C', 'D', 'E'],
                        help='Model configuration')
    parser.add_argument('--use_batch_norm', type=string_to_boolean,
                        default=False,
                        help='Toggles batch normalization in layers')
    parser.add_argument('--init_weights', type=string_to_boolean, default=True,
                        help='Toggles weight initialization')

    # misc
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='Mode of execution')
    parser.add_argument('--use_gpu', type=string_to_boolean, default=True,
                        help='Toggles the use of GPU')
    parser.add_argument('--use_tensorboard', type=string_to_boolean,
                        default=True,
                        help='Toggles the use of Tensorboard')

    # dataset
    # parser.add_argument('--data_path', type=str, default='../data/c256/')
    # parser.add_argument('--train_data_path', type=str,
    #                     default='caltech_256_60_train_nobg_norm.hdf5')
    # parser.add_argument('--train_x_key', type=str, default='train_x')
    # parser.add_argument('--train_y_key', type=str, default='train_y')
    # parser.add_argument('--test_data_path', type=str,
    #                     default='caltech_256_60_test_nobg_norm.hdf5')
    # parser.add_argument('--test_x_key', type=str, default='test_x')
    # parser.add_argument('--test_y_key', type=str, default='test_y')

    # path
    parser.add_argument('--model_save_path', type=str, default='./weights',
                        help='Path for saving weights')

    # epoch step size
    parser.add_argument('--loss_log_step', type=int, default=1)
    parser.add_argument('--model_save_step', type=int, default=1)
    parser.add_argument('--train_eval_step', type=int, default=1)

    config = parser.parse_args()

    args = vars(config)
    print(args)
    write_print('hello.txt', '------------ Options -------------')
    for k, v in args.items():
        write_print('hello.txt', '{}: {}'.format(str(k), str(v)))
    write_print('hello.txt', '-------------- End ----------------')

    # main(version, config)
