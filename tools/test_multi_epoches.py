import sys
import cv2
import os
import yaml
import numpy as np
from multiprocessing import Process
from tqdm import tqdm

def run_test(param_list, wid, skip):
    for cfg_file, test_folder, ckpt_file, out_dir in param_list:
        if os.path.exists(out_dir):
            if skip:
                print(f'{out_dir} exists, skip it.')
                continue
            else:
                print(f'{out_dir} exists, force to remove.')
                os.system(f'rm -rf {out_dir}')
        cmd = f'python3 test.py --cfg_file {cfg_file} --test_folder {test_folder} --ckpt {ckpt_file} --overwrite_output_dir {out_dir}'
        print(f'Working process {wid} runs: {cmd}')
        os.system(cmd)

def load_image(input_folder, output_folder, ckpt, fn, postfix, resize_to):
    if ckpt == 'blank':
        return np.zeros([resize_to[1], resize_to[0], 3], dtype=np.uint8)
    if ckpt == 'input':
        img = cv2.imread(os.path.join(input_folder, fn.replace(postfix, '')))
    else:
        img = cv2.imread(os.path.join(output_folder, ckpt, fn))
    if not (img.shape[0] == resize_to[1] and img.shape[1] == resize_to[0]):
        img = cv2.resize(img, resize_to)
    return cv2.putText(img, ckpt, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)

if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Usage: python3 test_multi_epochs.py [path to config].'

    # read config
    with open(sys.argv[1], 'rt') as f:
        config = yaml.safe_load(f)
    assert type(config) is dict, 'Loading config file failed.'

    # check config
    gpu_ids = config['gpu_ids']
    skip_existed = config['skip_existed']
    cfg_file = config['cfg_file']
    ckpt_folder = config['ckpt_folder']
    ckpt_layout = config['ckpt_layout']
    test_folder = config['test_folder']
    output_folder = config['output_folder']
    resize_to = config['resize_to']

    with open(cfg_file, 'rt') as f:
        temp_config = yaml.safe_load(f)
    assert type(temp_config) is dict, 'Loading config file failed.'

    postfix = temp_config['testing']['visual_names']
    assert len(postfix) == 1, 'Count of visual names should be 1.'
    postfix = '_' + postfix[0]

    temp_config['testing']['image_format'] = 'input'

    # run test
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    gpu_count = len(gpu_ids)
    param_lists = [ [] for _ in range(gpu_count) ]
    cnt = 0
    for line in ckpt_layout:
        assert len(line) == len(ckpt_layout[0]), 'Each row should have the same count of items.'
        for ckpt in line:
            if not ckpt in ['input', 'blank']:
                ckpt_path = os.path.join(ckpt_folder, ckpt + '.pth')
                assert os.path.exists(ckpt_path), f'Checkpoint not found: {ckpt_path}'
                new_cfg_file = os.path.join(output_folder, ckpt + '.yaml')
                out_dir = os.path.join(output_folder, ckpt)
                with open(new_cfg_file, 'wt') as f:
                    temp_config['common']['gpu_ids'] = [ gpu_ids[cnt % gpu_count] ]
                    yaml.safe_dump(temp_config, f)
                param_lists[cnt % gpu_count].append((new_cfg_file, test_folder, ckpt_path, out_dir))
                cnt += 1
    pool = []
    for i in range(gpu_count):
        p = Process(target=run_test, args=(param_lists[i], i, skip_existed))
        p.start()
        pool.append(p)
    for p in pool:
        p.join()

    # concat image
    print('Begin to concat.')
    out_dir = os.path.join(output_folder, 'concat')
    if os.path.exists(out_dir):
        print(f'{out_dir} exists, force to remove.')
        os.system(f'rm -rf {out_dir}')
    os.makedirs(out_dir)
    files = os.listdir(param_lists[0][0][3])
    for fn in tqdm(files):
        if not os.path.splitext(fn)[1].lower() in ['.jpg', '.jpeg', '.png']:
            continue
        try:
            result = np.vstack([
                np.hstack([
                    load_image(test_folder, output_folder, ckpt, fn, postfix, resize_to) for ckpt in line
                ]) for line in ckpt_layout
            ])
            cv2.imwrite(os.path.join(out_dir, fn.replace(postfix, '')), result)
        except Exception as e:
            print(fn, e)
