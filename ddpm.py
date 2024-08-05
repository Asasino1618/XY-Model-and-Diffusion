import argparse
import sys
import os
import random

from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm

import torch
from torch_geometric.data import Data

from ddpm.denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

from utils.transforms import Normalizer
from utils.preprocess import preprocess_ddpm

parser = argparse.ArgumentParser(description='1D DDPM')
parser.add_argument('--date', type=str, required=True, 
                    help='use date as the autoencoder model id, yyyymmddhhmm')
parser.add_argument('--task', type=str, required=True, 
                    help='train or sample')

args = parser.parse_args(sys.argv[1:])

def main():
    global args

    torch.cuda.empty_cache()

    date = args.date
    num_timesteps = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, data_normalizer, prop_normalizer = preprocess_ddpm(date, num_timesteps=num_timesteps)

    seq_length = dataset.seq_length
    
    model = Unet1D(
        dim = seq_length,
        dim_mults = (1, 2, 4, 8),
        channels = 1
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = seq_length,
        timesteps = num_timesteps,
        objective = 'pred_noise'
    )

    # Or using trainer

    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 7000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        num_samples = 256,                # number of samples must have an integer square root
        date = date
    )
    
    if args.task == 'sample':
        trainer.load('fin')
    trainer.train()

    # after a lot of training
    targets_denorm = torch.arange(-1, 5, 0.1)

    for target_denorm in targets_denorm:
        target = prop_normalizer.norm(torch.tensor([target_denorm])).to(device)
        sampled_seq = diffusion.sample(prop=target, batch_size = 64).cpu()
        sampled_seq = data_normalizer.denorm(sampled_seq)

        sampled_path = f'results/sampled/{date}'
        if not os.path.exists(sampled_path):
            os.makedirs(sampled_path)
            print(f'sampled/{date} subfolder not found, new one created')

        for i in range(sampled_seq.shape[0]):
            data = Data(x=sampled_seq[i], cif_id=[f'sample_{target_denorm:.2f}_{i+1}'], prop=torch.tensor([target_denorm]))
            torch.save(data, os.path.join(sampled_path, f'sample-{target_denorm:.2f}-{i+1}.pt'))

if __name__ == '__main__':
    main()