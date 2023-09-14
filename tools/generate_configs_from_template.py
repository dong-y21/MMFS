import argparse
import copy
import os
import yaml
import sys
sys.path.append('.')
from configs import parse_config

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Config Generator')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    args = parser.parse_args()

    with open(args.input, 'rt') as f:
        template_config = yaml.safe_load(f)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # check template and copy config
    config = copy.deepcopy(template_config)
    group_sizes = []
    for key in config:
        if 'option_group' in config[key]:
            group_sizes.append(len(config[key]['option_group']))
            del config[key]['option_group']
    is_same_size = True
    for i in range(1, len(group_sizes)):
        is_same_size = is_same_size and group_sizes[i] == group_sizes[i - 1]
    assert len(group_sizes) > 0, 'Template config should have at least one option group.'
    assert is_same_size, 'All option groups should have the same size.'

    # generate all configs
    cmds = []
    for idx in range(group_sizes[0]):
        dst_config = copy.deepcopy(config)
        for key in dst_config:
            if 'option_group' in template_config[key]:
                for option, value in template_config[key]['option_group'][idx].items():
                    dst_config[key][option] = value
        dst_config['common']['name'] += f'{idx:02d}'
        dst_fn = dst_config['common']['name']
        dst_path = os.path.join(args.output_folder, dst_fn + '.yaml')
        with open(dst_path, 'wt') as f:
            yaml.safe_dump(dst_config, f)

        cmds.append(f'nohup python3 train.py --cfg_file {dst_path} > stdio/{dst_fn}.log 2>&1 &')

        # try to load config
        try:
            parse_config(dst_path)
        except Exception as e:
            print('\033[1;31m==========\nError:\033[0m', e, '\n\033[1;31m==========\033[0m')
            exit(-1)

    for cmd in cmds:
        print('')
        print(cmd)
