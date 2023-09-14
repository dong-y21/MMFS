import argparse
import os

from utils.util import *
from models import create_model
from configs import parse_config
import torch

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Style Master')
    parser.add_argument('--cfg_file', type=str, default='./exp/cycle_gan_cfg.yaml')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument("--output_jit_file", type=str, default='./jit_models/cycle_gan.jit')
    args = parser.parse_args()

    # parse config
    config = parse_config(args.cfg_file)

    config['common']['phase'] = 'test'

    model = create_model(config)      # create a model given opt.model and other options
    model.load_networks(0, ckpt=args.ckpt)

    model.eval()

    dummy_input = torch.rand(1, config['model']['input_nc'], config['testing']['load_size'], config['testing']['load_size'])
    traced_script_module, dummy_output, dummy_output_traced = model.trace_jit(dummy_input)

    if type(dummy_output) is list or type(dummy_output) is tuple:
        diffs = []
        for i in range(len(dummy_output)):
            diffs.append(np.abs(dummy_output[i].detach().numpy() - dummy_output_traced[i].detach().numpy()))
    else:
        diffs = np.abs(dummy_output.detach().numpy() - dummy_output_traced.detach().numpy())

    if not type(diffs) is list:
        diffs = [ diffs ]
    for i in range(len(diffs)):
        avg_diff, max_diff, min_diff = np.mean(diffs[i]), np.max(diffs[i]), np.min(diffs[i])
        print('Network output ', i + 1)
        print("average difference between original and traced model: ", avg_diff)
        print("max difference between original and traced model: ", max_diff)
        print("min difference between original and traced model: ", min_diff)

    traced_script_module.save(args.output_jit_file)
