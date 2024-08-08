import torch
from denoising_diffusion_pytorch import denoising_diffusion_pytorch
from torchvision import transforms
import os
import datetime
import argparse
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="XY_M CAL")
parser.add_argument('--N', type=int, default=100, help='Number Generate')
parser.add_argument('--ID', type=str, default=100, help='load ID')
args = parser.parse_args(sys.argv[1:])

def calculate_magnetic_moment(images):
    images = images * 2 * torch.pi
    print(torch.cos(images))
    mx = torch.sum(torch.cos(images), dim=[1, 2, 3])
    my = torch.sum(torch.sin(images), dim=[1, 2, 3])
    print(mx, my)
    return torch.sqrt(mx**2 + my**2)

def main():
    global args
    
    torch.cuda.empty_cache()

    num_timesteps = 1000
    lr = 8e-5
    ts = num_timesteps
    tbs = 16
    tns = 2000
    sat = 200

    model = denoising_diffusion_pytorch.Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = False,
        channels = 1,
    )

    diffusion = denoising_diffusion_pytorch.GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = num_timesteps,    # number of steps
        objective = "pred_noise",
        sampling_timesteps=sat,
    )

    trainer = denoising_diffusion_pytorch.Trainer(
        diffusion,
        'database_for_diffusion/train',
        train_batch_size = tbs,
        train_lr = lr,
        train_num_steps = tns,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = True,              # whether to calculate fid during training
        num_fid_samples = 100,
        date = args.ID,
        save_best_and_latest_only=True,
        val_path="diffusion_val/1eminus1/",
        tem_dict={'high':2.0 * 500.0, 'low':0.1 * 500.0},
    )
    trainer.load(ID=args.ID, milestone='fin')
    Tr = np.linspace(0.1, 2.0, 39)
    m_average = torch.zeros_like(torch.from_numpy(Tr))
    for t in range(len(Tr)):
        sampled_images = diffusion.sample(batch_size=args.N, Temperature=torch.tensor(Tr[t] * 500.0))
        magnetic_moments = calculate_magnetic_moment(sampled_images)
        print(magnetic_moments)
        m_average[t] = magnetic_moments.mean()
        print(m_average[t].item())

    # Plot the results
    plt.plot(Tr, m_average.numpy())
    plt.xlabel('Temperature')
    plt.ylabel('Magnetic Moment (M)')
    plt.title('Magnetic Moment vs Temperature')
    plt.show()
    plt.savefig('M-T.svg')
    plt.close()


if __name__ == '__main__':
    main()