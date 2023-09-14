import os
from utils.util import check_path_is_img
from utils.data_utils import Transforms, check_create_shuffled_order
from utils.augmentation import ImagePathToImage


def add_paired_data(data, transforms, config, paired_data_order):
    A_paths = []
    B_paths = []

    if config['dataset']['paired_' + config['common']['phase'] + '_filelist'] != '':
        paired_data_file = open(config['dataset']['paired_' + config['common']['phase'] + '_filelist'], 'r')
        Lines = paired_data_file.readlines()
        paired_data_order = check_create_shuffled_order(Lines, paired_data_order)
        for i in paired_data_order:
            line = Lines[i]
            if not config['dataset']['use_absolute_datafile']:
                file1 = os.path.join(config['dataset']['dataroot'], line.split(" ")[0]).strip()
                file2 = os.path.join(config['dataset']['dataroot'], line.split(" ")[1]).strip()
            else:
                file1 = line.split(" ")[0].strip()
                file2 = line.split(" ")[1].strip()
            if os.path.exists(file1) and os.path.exists(file2):
                A_paths.append(file1)
                B_paths.append(file2)
        paired_data_file.close()
    elif config['dataset']['paired_' + config['common']['phase'] + 'A_folder'] != '' and \
            config['dataset']['paired_' + config['common']['phase'] + 'B_folder'] != '':
        dir_A = config['dataset']['paired_' + config['common']['phase'] + 'A_folder']
        dir_B = config['dataset']['paired_' + config['common']['phase'] + 'B_folder']
        filenames = os.listdir(dir_A)
        paired_data_order = check_create_shuffled_order(filenames, paired_data_order)
        for i in paired_data_order:
            filename = filenames[i]
            if not check_path_is_img(filename):
                continue
            A_path = os.path.join(dir_A, filename)
            B_path = os.path.join(dir_B, filename)
            if os.path.exists(A_path) and os.path.exists(B_path):
                A_paths.append(A_path)
                B_paths.append(B_path)
    else:
        dir_A = os.path.join(config['dataset']['dataroot'], config['common']['phase'] + 'pairedA')
        dir_B = os.path.join(config['dataset']['dataroot'], config['common']['phase'] + 'pairedB')
        if os.path.exists(dir_A) and os.path.exists(dir_B):
            filenames = os.listdir(dir_A)
            paired_data_order = check_create_shuffled_order(filenames, paired_data_order)
            for i in paired_data_order:
                filename = filenames[i]
                if not check_path_is_img(filename):
                    continue
                A_path = os.path.join(dir_A, filename)
                B_path = os.path.join(dir_B, filename)
                if os.path.exists(A_path) and os.path.exists(B_path):
                    A_paths.append(A_path)
                    B_paths.append(B_path)

    btoA = config['dataset']['direction'] == 'BtoA'
    # get the number of channels of input image
    input_nc = config['model']['output_nc'] if btoA else config['model']['input_nc']
    output_nc = config['model']['input_nc'] if btoA else config['model']['output_nc']

    transform = Transforms(config, input_grayscale_flag=(input_nc == 1), output_grayscale_flag=(output_nc == 1))
    transform.get_transform_from_config()
    transform.get_transforms().insert(0, ImagePathToImage())
    transform = transform.compose_transforms()

    data['paired_A_path'] = A_paths
    data['paired_B_path'] = B_paths

    transforms['paired'] = transform
    return paired_data_order


def apply_paired_transforms(index, data, transforms, return_dict):
    if len(data['paired_A_path']) > 0:
        return_dict['paired_A'], return_dict['paired_B'] = transforms['paired'] \
            (data['paired_A_path'][index], data['paired_B_path'][index])
        return_dict['paired_A_path'] = data['paired_A_path'][index]
        return_dict['paired_B_path'] = data['paired_B_path'][index]
