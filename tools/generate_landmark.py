import sys
sys.path.append('/share/group_machongyang/limengtian/libycnn2/lib')
import argparse
import os
import cv2
import numpy as np
from multiprocessing import Process
from tqdm import tqdm
from py_ycnn import face_detection

detector = face_detection()
detector.init_face_detection('/share/group_machongyang/limengtian/libycnn2/model/')

def work(lst, output_folder, layout, wid):
    need_A = 'A' in layout
    need_B = 'B' in layout
    if need_A:
        os.mkdir(os.path.join(output_folder, f'lmk_A_{wid:02d}'))
    if need_B:
        os.mkdir(os.path.join(output_folder, f'lmk_B_{wid:02d}'))

    res_list = []
    for i, (A_path, B_path) in tqdm(enumerate(lst)):
        if os.path.isfile(A_path) and os.path.isfile(B_path):
            if need_A:
                A = cv2.imread(A_path)
                A = detector.detectArray(A)
                if len(A) < 1:
                    continue
                A = np.array(A[0], dtype=np.float32)
                A = A[:202].reshape(101, 2)
                np.save(os.path.join(output_folder, f'lmk_A_{wid:02d}', f'{i:06d}.npy'), A)
            if need_B:
                B = cv2.imread(B_path)
                B = detector.detectArray(B)
                if len(B) < 1:
                    continue
                B = np.array(B[0], dtype=np.float32)
                B = B[:202].reshape(101, 2)
                np.save(os.path.join(output_folder, f'lmk_B_{wid:02d}', f'{i:06d}.npy'), B)
            if layout[0] == 'A':
                lmk_A_path = os.path.join(output_folder, f'lmk_A_{wid:02d}', f'{i:06d}.npy')
            else:
                lmk_A_path = os.path.join(output_folder, f'lmk_B_{wid:02d}', f'{i:06d}.npy')
            if layout[1] == 'A':
                lmk_B_path = os.path.join(output_folder, f'lmk_A_{wid:02d}', f'{i:06d}.npy')
            else:
                lmk_B_path = os.path.join(output_folder, f'lmk_B_{wid:02d}', f'{i:06d}.npy')
            res_list.append((A_path, B_path, lmk_A_path, lmk_B_path))

    with open(os.path.join(output_folder, f'lmk_{wid:02d}.txt'), 'wt') as f:
        for A_path, B_path, lmk_A_path, lmk_B_path in res_list:
            f.write(f'{A_path} {B_path} {lmk_A_path} {lmk_B_path}\n')

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Landmark Generator')
    parser.add_argument('--input_list', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--layout', type=str, default='AB')
    parser.add_argument('--worker_count', type=int, default=16)
    args = parser.parse_args()

    # check arguments
    assert os.path.isfile(args.input_list), f'FileNotFound: {args.input_list}.'
    assert not os.path.exists(args.output_folder), 'Output folder exists.'
    assert args.layout in ['AA', 'AB', 'BA', 'BB'], 'Layout error.'
    assert args.worker_count >= 1 and args.worker_count <= 64, 'Worker count error.'

    os.makedirs(args.output_folder)

    # read input list and split to each process
    lsts = [ [] for _ in range(args.worker_count) ]
    with open(args.input_list, 'rt') as f:
        line = f.readline().strip()
        total_cnt = 0
        while line:
            A_path, B_path = line.split(' ')
            lsts[total_cnt % args.worker_count].append((A_path, B_path))
            line = f.readline().strip()
            total_cnt += 1
    print('Total images:', total_cnt)

    # generate landmark
    pool = []
    for i in range(args.worker_count):
        proc = Process(target=work, args=(lsts[i], args.output_folder, args.layout, i))
        proc.start()
        pool.append(proc)
    for proc in pool:
        proc.join()

    # combine lists
    new_fn = os.path.splitext(os.path.split(args.input_list)[1])[0] + '_with_lmk.txt'
    with open(os.path.join(args.output_folder, new_fn), 'wt') as fw:
        for i in range(args.worker_count):
            with open(os.path.join(args.output_folder, f'lmk_{i:02d}.txt'), 'rt') as fr:
                line = fr.readline()
                while line:
                    fw.write(line)
                    line = fr.readline()
