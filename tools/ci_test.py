import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
sys.path.append('./')
sys.path.append('../')
import skimage.io as skio
import skimage.transform as skt
import numpy as np
from data import CustomDataLoader
from data.super_dataset import SuperDataset
from models import create_model
from configs import parse_config
from utils.util import check_path
import random
import argparse

def make_toy_dataset():
    check_path('./toy_dataset')

    # paired
    check_path('./toy_dataset/trainpairedA')
    check_path('./toy_dataset/trainpairedB')

    # paired numpy
    check_path('./toy_dataset/trainnumpypairedA')
    check_path('./toy_dataset/trainnumpypairedB')

    # unpaired
    check_path('./toy_dataset/trainunpairedA')
    check_path('./toy_dataset/trainunpairedB')

    # unpaired numpy
    check_path('./toy_dataset/trainnumpyunpairedA')
    check_path('./toy_dataset/trainnumpyunpairedB')


    # landmark
    check_path('./toy_dataset/trainlmkA')
    check_path('./toy_dataset/trainlmkB')

    for i in range(6):
        A0 = np.random.randn(8, 8, 3) * 0.5 + 0.5
        A0[:,:,0] = 0
        A0 = np.clip(A0, 0, 1)

        A1 = np.random.randn(8, 8, 3) * 0.5 + 0.5
        A1[:,:,1] = 0
        A1 = np.clip(A1, 0, 1)

        A2 = np.random.randn(8, 8, 3) * 0.5 + 0.5
        A2[:,:,2] = 0
        A2 = np.clip(A2, 0, 1)

        B = np.random.randn(8, 8, 3) * 0.5 + 0.5
        B = np.clip(B, 0, 1)

        A0 = skt.resize(A0, (128, 128))
        A1 = skt.resize(A1, (128, 128))
        A2 = skt.resize(A2, (128, 128))
        B = skt.resize(B, (128, 128))

        # paired numpy
        np.save('./toy_dataset/trainnumpypairedA/%d.npy' % i, A0.astype(np.float32))
        np.save('./toy_dataset/trainnumpypairedB/%d.npy' % i, B.astype(np.float32))

        # unpaired numpy
        np.save('./toy_dataset/trainnumpyunpairedA/%d.npy' % i, A0.astype(np.float32))
        np.save('./toy_dataset/trainnumpyunpairedB/%d.npy' % i, B.astype(np.float32))

        A0 = A0 * 255.0
        A1 = A1 * 255.0
        A2 = A2 * 255.0
        B = B * 255.0

        # paired
        skio.imsave('./toy_dataset/trainpairedA/%d.png' % i, A0.astype(np.uint8))
        skio.imsave('./toy_dataset/trainpairedB/%d.png' % i, B.astype(np.uint8))

        # unpaired
        skio.imsave('./toy_dataset/trainunpairedA/%d.png' % i, A0.astype(np.uint8))
        skio.imsave('./toy_dataset/trainunpairedB/%d.png' % i, B.astype(np.uint8))

        landmark = np.random.rand(101, 2) * 0.5 + 0.5
        landmark = np.clip(landmark, 0, 1)

        # landmark
        np.save('./toy_dataset/trainlmkA/%d.npy' % i, landmark.astype(np.float32))
        np.save('./toy_dataset/trainlmkB/%d.npy' % i, landmark.astype(np.float32))

def main(args):
    make_toy_dataset()
    config_dir = './exp'
    if not os.path.exists(config_dir):
        config_dir = './../exp'

    config_files = os.listdir(config_dir)
    if not args.all_tests:
        random.shuffle(config_files)
        config_files = config_files[:2]

    for cfg in config_files:
        if (not cfg.endswith('.yaml')) or "example" in cfg:
            continue
        print('Current:', cfg)

        try:
            # parse config
            config = parse_config(os.path.join(config_dir, cfg))

            config['common']['gpu_ids'] = None
            config['training']['continue_train'] = False
            config['dataset']['n_threads'] = 0
            config['dataset']['batch_size'] = 2

            if 'patch_size' in config['dataset']:
                config['dataset']['patch_size'] = 64
            if 'patch_batch_size' in config['dataset']:
                config['dataset']['patch_batch_size'] = 2

            config['dataset']['preprocess'] = ['scale_width']

            config['dataset']['paired_trainA_folder'] = ''
            config['dataset']['paired_trainB_folder'] = ''
            config['dataset']['paired_train_filelist'] = ''
            config['dataset']['paired_valA_folder'] = ''
            config['dataset']['paired_valB_folder'] = ''
            config['dataset']['paired_val_filelist'] = ''

            config['dataset']['unpaired_trainA_folder'] = ''
            config['dataset']['unpaired_trainB_folder'] = ''
            config['dataset']['unpaired_trainA_filelist'] = ''
            config['dataset']['unpaired_trainB_filelist'] = ''
            config['dataset']['unpaired_valA_folder'] = ''
            config['dataset']['unpaired_valB_folder'] = ''
            config['dataset']['unpaired_valA_filelist'] = ''
            config['dataset']['unpaired_valB_filelist'] = ''

            config['dataset']['dataroot'] = "./toy_dataset"

            # create dataset
            dataset = SuperDataset(config)
            dataset.config = dataset.convert_old_config_to_new()
            dataset.static_data.load_static_data()
            dataset.static_data.create_transforms()

            print('The number of training images = %d' % len(dataset))
            dataloader = CustomDataLoader(config, dataset)

            # create model
            model = create_model(config)
            model.setup(config)

            # train
            for data in dataloader:
                model.set_input(data)
                model.optimize_parameters()
                losses = model.get_current_losses()
                print(losses)

        except ImportError as error:
            print(error)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ci_test')
    parser.add_argument('--all_tests', action='store_true')
    args = parser.parse_args()
    main(args)
