import os
from PIL import Image
import numpy as np
from utils.data_utils import check_create_shuffled_order, check_equal_length

def landmark_path_to_numpy(lmk_path, image_path, image_tensor):
    """Convert an landmark path to the actual landmarks in a numpy array. Also applies scaling to the landmarks
    according to final image' size.

    Parameters:
        lmk_path  --  the landmark file path.
        image_path  --  the original image file path.
        image_tensor  --  the image tensor after all transformations.
    """
    lmk = np.load(lmk_path)
    ow, oh = Image.open(image_path).size
    h, w = image_tensor.size()[1:]
    lmk[:, 0] *= w / ow
    lmk[:, 1] *= h / oh
    return lmk

def add_landmark_data(data, config, paired_data_order):
    A_lmk_paths = []
    B_lmk_paths = []

    if config['dataset']['paired_' + config['common']['phase'] + '_filelist'] != '':
        paired_data_file = open(config['dataset']['paired_' + config['common']['phase'] + '_filelist'], 'r')
        Lines = paired_data_file.readlines()
        paired_data_order = check_create_shuffled_order(Lines, paired_data_order)
        check_equal_length(Lines, paired_data_order, data)
        for i in paired_data_order:
            line = Lines[i]
            if not config['dataset']['use_absolute_datafile']:
                file3 = os.path.join(config['dataset']['dataroot'], line.split(" ")[2]).strip()
                file4 = os.path.join(config['dataset']['dataroot'], line.split(" ")[3]).strip()
            else:
                file3 = line.split(" ")[2].strip()
                file4 = line.split(" ")[3].strip()
            if os.path.exists(file3) and os.path.exists(file4):
                A_lmk_paths.append(file3)
                B_lmk_paths.append(file4)
        paired_data_file.close()
    elif config['dataset']['paired_' + config['common']['phase'] + 'A_folder'] != '' and \
            config['dataset']['paired_' + config['common']['phase'] + 'B_folder'] != '' and \
            os.path.exists(config['dataset']['paired_' + config['common']['phase'] + 'A_lmk_folder']) and \
            os.path.exists(config['dataset']['paired_' + config['common']['phase'] + 'B_lmk_folder']):
        dir_A = config['dataset']['paired_' + config['common']['phase'] + 'A_folder']
        dir_A_lmk = config['dataset']['paired_' + config['common']['phase'] + 'A_lmk_folder']
        dir_B_lmk = config['dataset']['paired_' + config['common']['phase'] + 'B_lmk_folder']
        filenames = os.listdir(dir_A)
        paired_data_order = check_create_shuffled_order(filenames, paired_data_order)
        check_equal_length(filenames, paired_data_order, data)
        for i in paired_data_order:
            filename = filenames[i]
            A_lmk_path = os.path.join(dir_A_lmk, os.path.splitext(filename)[0] + '.npy')
            B_lmk_path = os.path.join(dir_B_lmk, os.path.splitext(filename)[0] + '.npy')
            if os.path.exists(A_lmk_path) and os.path.exists(B_lmk_path):
                A_lmk_paths.append(A_lmk_path)
                B_lmk_paths.append(B_lmk_path)
    else:
        dir_A = os.path.join(config['dataset']['dataroot'], config['common']['phase'] + 'pairedA')
        dir_A_lmk = os.path.join(config['dataset']['dataroot'], config['common']['phase'] + 'pairedA_lmk')
        dir_B_lmk = os.path.join(config['dataset']['dataroot'], config['common']['phase'] + 'pairedB_lmk')
        if os.path.exists(dir_A_lmk) and os.path.exists(dir_B_lmk):
            filenames = os.listdir(dir_A)
            paired_data_order = check_create_shuffled_order(filenames, paired_data_order)
            check_equal_length(filenames, paired_data_order, data)
            for i in paired_data_order:
                filename = filenames[i]
                A_lmk_path = os.path.join(dir_A_lmk, os.path.splitext(filename)[0] + '.npy')
                B_lmk_path = os.path.join(dir_B_lmk, os.path.splitext(filename)[0] + '.npy')
                if os.path.exists(A_lmk_path) and os.path.exists(B_lmk_path):
                    A_lmk_paths.append(A_lmk_path)
                    B_lmk_paths.append(B_lmk_path)
        else:
            print(dir_A_lmk + " or " + dir_B_lmk + " doesn't exist. Skipping landmark data.")

    data['A_lmk_path'] = A_lmk_paths
    data['B_lmk_path'] = B_lmk_paths

    return paired_data_order


def apply_landmark_transforms(index, data, return_dict):
    if len(data['A_lmk_path']) > 0:
        return_dict['A_lmk'] = landmark_path_to_numpy(data['A_lmk_path'][index], data['paired_A_path'][index], return_dict['paired_A'])
        return_dict['B_lmk'] = landmark_path_to_numpy(data['B_lmk_path'][index], data['paired_B_path'][index], return_dict['paired_B'])
        return_dict['A_lmk_path'] = data['A_lmk_path'][index]
        return_dict['B_lmk_path'] = data['B_lmk_path'][index]
