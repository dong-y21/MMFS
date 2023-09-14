import os
from utils.util import check_path_is_img
from utils.data_utils import Transforms
from utils.augmentation import ImagePathToImage
import random

def add_unpaired_data(data, transforms, config, shuffle=False):
    A_paths = []
    B_paths = []
    if config['dataset']['unpaired_' + config['common']['phase'] + 'A_filelist'] != '':
        unpaired_data_file1 = open(config['dataset']['unpaired_' + config['common']['phase'] + 'A_filelist'], 'r')
        Lines = unpaired_data_file1.readlines()
        if shuffle:
            random.shuffle(Lines)
        for line in Lines:
            if not config['dataset']['use_absolute_datafile']:
                file = os.path.join(config['dataset']['dataroot'], line.strip())
            else:
                file = line.strip()
            if os.path.exists(file):
                A_paths.append(file)
        unpaired_data_file1.close()

        unpaired_data_file2 = open(config['dataset']['unpaired_' + config['common']['phase'] + 'B_filelist'], 'r')
        Lines = unpaired_data_file2.readlines()
        if shuffle:
            random.shuffle(Lines)
        for line in Lines:
            if not config['dataset']['use_absolute_datafile']:
                file = os.path.join(config['dataset']['dataroot'], line.strip())
            else:
                file = line.strip()
            if os.path.exists(file):
                B_paths.append(file)
        unpaired_data_file2.close()
    elif config['dataset']['unpaired_' + config['common']['phase'] + 'A_folder'] != '' and \
         config['dataset']['unpaired_' + config['common']['phase'] + 'B_folder'] != '':
        dir_A = config['dataset']['unpaired_' + config['common']['phase'] + 'A_folder']
        filenames = os.listdir(dir_A)
        if shuffle:
            random.shuffle(filenames)
        for filename in filenames:
            if not check_path_is_img(filename):
                continue
            A_path = os.path.join(dir_A, filename)
            if os.path.exists(A_path):
                A_paths.append(A_path)

        dir_B = config['dataset']['unpaired_' + config['common']['phase'] + 'B_folder']
        filenames = os.listdir(dir_B)
        if shuffle:
            random.shuffle(filenames)
        for filename in filenames:
            if not check_path_is_img(filename):
                continue
            B_path = os.path.join(dir_B, filename)
            if os.path.exists(B_path):
                B_paths.append(B_path)

    else:
        dir_A = os.path.join(config['dataset']['dataroot'], config['common']['phase'] + 'unpairedA')
        dir_B = os.path.join(config['dataset']['dataroot'], config['common']['phase'] + 'unpairedB')
        if os.path.exists(dir_A) and os.path.exists(dir_B):
            filenames = os.listdir(dir_A)
            if shuffle:
                random.shuffle(filenames)
            for filename in filenames:
                if not check_path_is_img(filename):
                    continue
                A_path = os.path.join(dir_A, filename)
                A_paths.append(A_path)
            filenames = os.listdir(dir_B)
            if shuffle:
                random.shuffle(filenames)
            for filename in filenames:
                if not check_path_is_img(filename):
                    continue
                B_path = os.path.join(dir_B, filename)
                B_paths.append(B_path)


    btoA = config['dataset']['direction'] == 'BtoA'
    input_nc = config['model']['output_nc'] if btoA else config['model']['input_nc']
    output_nc = config['model']['input_nc'] if btoA else config['model']['output_nc']

    transform = Transforms(config, input_grayscale_flag=(input_nc == 1), output_grayscale_flag=(output_nc == 1))
    transform.get_transform_from_config()
    transform.get_transforms().insert(0, ImagePathToImage())
    transform = transform.compose_transforms()

    data['unpaired_A_path'] = A_paths
    data['unpaired_B_path'] = B_paths
    transforms['unpaired'] = transform

def apply_unpaired_transforms(index, data, transforms, return_dict):
    if len(data['unpaired_A_path']) > 0 and len(data['unpaired_B_path']) > 0:
        index_B = random.randint(0, len(data['unpaired_B_path']) - 1)
        return_dict['unpaired_A'], return_dict['unpaired_B'] = transforms['unpaired'] \
            (data['unpaired_A_path'][index], data['unpaired_B_path'][index_B])
        return_dict['unpaired_A_path'] = data['unpaired_A_path'][index]
        return_dict['unpaired_B_path'] = data['unpaired_B_path'][index_B]
