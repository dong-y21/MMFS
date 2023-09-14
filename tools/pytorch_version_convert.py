# 该脚本用于将pytorch 1.6版本的zip模型转换为pytorch、libtorch 1.1/1.3适用的unzipped模型

import os
import argparse
import torch
import functools
import sys
import numpy as np
import cv2
import zipfile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch 1.6 model to 1.3/1.1 model (unzip)')
    parser.add_argument('--output_folder', type=str, default='./', help='Output folder')
    parser.add_argument('--output_filename', type=str, default='output_model.pt', help='Output file name')
    parser.add_argument('--model', default=None, type=str, help='Model path')
    args = parser.parse_args()

    #define model
    print("---load model ---")
    #pytorch version
    torch_version = torch.__version__.split('.')
    print('PyTorch version: ', torch.__version__)
    if int(torch_version[0]) == 1 and int(torch_version[1]) < 6:
        print('Please use PyTorch version >= 1.6 to convert.')
        exit(0)
            
    output_path = args.output_folder
    if output_path[-1] != '/':
        output_path += '/'

    if zipfile.is_zipfile(args.model):
        #load weights
        pretrained_dict = torch.load(args.model)

        torch.save(pretrained_dict, output_path + args.output_filename, _use_new_zipfile_serialization=False)
        print("---export done---")
    else:
        print('The model is not a zip file, it can be handled by PyTorch 1.1/1.3 without version conversion. Please go ahead and use trace_jit.py to export your model.')
        
