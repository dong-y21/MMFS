import os
import random
import numpy as np
from utils.augmentation import ImagePathToImage
from utils.data_utils import Transforms, check_img_loaded, check_numpy_loaded


class CustomData(object):

    def __init__(self, config, shuffle=False):
        self.paired_file_groups = []
        self.paired_type_groups = []
        self.len_of_groups = []
        self.landmark_scale = config['dataset']['landmark_scale']
        self.shuffle = shuffle
        self.config = config

        data_dict = config['dataset']['custom_' + config['common']['phase'] + '_data']
        if len(data_dict) == 0:
            self.len_of_groups.append(0)
            return

        for i, group in enumerate(data_dict.values()):  # one example: (0, group_1),  (1, group_2)
            data_types = group['data_types']  # one example: 'image', 'patch'
            data_names = group['data_names']  # one example: 'real_A', 'patch_A'
            file_list = group['file_list']  # one example: "lmt/data/trainA.txt"
            assert(len(data_types) == len(data_names))

            self.paired_file_groups.append({})
            self.paired_type_groups.append({})
            for data_name, data_type in zip(data_names, data_types):
                self.paired_file_groups[i][data_name] = []
                self.paired_type_groups[i][data_name] = data_type

            paired_file = open(file_list, 'rt')
            lines = paired_file.readlines()
            if self.shuffle:
                random.shuffle(lines)
            for line in lines:
                items = line.strip().split(' ')
                if len(items) == len(data_names):
                    ok = True
                    for item in items:
                        ok = ok and os.path.exists(item) and os.path.getsize(item) > 0
                    if ok:
                        for data_name, item in zip(data_names, items):
                            self.paired_file_groups[i][data_name].append(item)
            paired_file.close()

            self.len_of_groups.append(len(self.paired_file_groups[i][data_names[0]]))

        self.transform = Transforms(config)
        self.transform.get_transform_from_config()
        self.transform.get_transforms().insert(0, ImagePathToImage())
        self.transform = self.transform.compose_transforms()

    def get_len(self):
        return max(self.len_of_groups)

    def get_item(self, idx):
        return_dict = {}
        for i in range(len(self.paired_file_groups)):
            inner_idx = idx if idx < self.len_of_groups[i] else random.randint(0, self.len_of_groups[i] - 1)
            img_list = []
            img_k_list = []
            for k, v in self.paired_file_groups[i].items():
                if self.paired_type_groups[i][k] == 'image':
                    # gather images for processing later
                    img_k_list.append(k)
                    img_list.append(v[inner_idx])
                elif self.paired_type_groups[i][k] == 'landmark':
                    # different from images, landmark doesn't use data augmentation. So process them directly here.
                    lmk = np.load(v[inner_idx])
                    lmk[:, 0] *= self.landmark_scale[0]
                    lmk[:, 1] *= self.landmark_scale[1]
                    return_dict[k] = lmk
                return_dict[k + '_path'] = v[inner_idx]

            # transform all images
            if len(img_list) == 1:
                return_dict[img_k_list[0]], _ = self.transform(img_list[0], None)
            elif len(img_list) > 1:
                input1, input2 = img_list[0], img_list[1:]
                output1, output2 = self.transform(input1, input2) # output1 is one image. output2 is a list of images.
                return_dict[img_k_list[0]] = output1
                for j in range(1, len(img_list)):
                    return_dict[img_k_list[j]] = output2[j-1]

        return return_dict

    def split_data_into_bins(self, num_bins):
        bins = []
        for i in range(0, num_bins):
            bins.append([])
        for i in range(0, len(self.paired_file_groups)):
            for b in range(0, num_bins):
                bins[b].append({})
            for dataname, item_list in self.paired_file_groups[i].items():
                if len(item_list) < self.config['dataset']['n_threads']:
                    bins[0][i][dataname] = item_list
                else:
                    num_items_in_bin = len(item_list) // num_bins
                    for j in range(0, len(item_list)):
                        which_bin = min(j // num_items_in_bin, num_bins - 1)
                        if dataname not in bins[which_bin][i]:
                            bins[which_bin][i][dataname] = []
                        else:
                            bins[which_bin][i][dataname].append(item_list[j])
        return bins

    def check_data_helper(self, data):
        all_pass = True
        for paired_file_group in data:
            for k, v in paired_file_group.items():
                if len(v) > 0:
                    for v1 in v:
                        if '.npy' in v1:  # case: numpy array or landmark
                            all_pass = all_pass and check_numpy_loaded(v1)
                        else:  # case: image
                            all_pass = all_pass and check_img_loaded(v1)
        return all_pass
