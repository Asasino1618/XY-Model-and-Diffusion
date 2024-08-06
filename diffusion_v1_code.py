import torch
from denoising_diffusion_pytorch import denoising_diffusion_pytorch
from torchvision import transforms
import os
import datetime
import argparse
import sys
import random
import csv
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="XY_M DDPM")
parser.add_argument('--task', type=str, required=True, help='train or sample')
parser.add_argument('--T', type=float, required=True, help='Temperature')
parser.add_argument('--N', type=int, default=100, help='Number Generate')
parser.add_argument('--ID', type=str, default=100, help='load ID')
args = parser.parse_args(sys.argv[1:])

#Temp range:0-2
def log_info(date_str, lr, ts, tbs, tns, sat, Loss, fid, file_name='ddpm_log.csv'):
    # Define the header
    header = ["Model ID",  
              "Learning Rate", "timesteps", "train_batch_size",
              "train_num_steps", "sampling_timesteps", "Loss_avg", "fid"
              ]

    # Format the data
    data = [date_str, lr, ts, tbs, tns, sat, Loss, fid]

    # Check if file exists and write the data
    file_exists = os.path.isfile(file_name)
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)  # write the header if file is new
        writer.writerow(data)

def main():
    global args

    torch.cuda.empty_cache()

    date = datetime.datetime.now()
    date_str = date.strftime('%Y%m%d%H%M')

    num_timesteps = 1000
    Temperature = torch.tensor(args.T, dtype=torch.float32) / 2.0 * num_timesteps#要inference的温度
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
        date = date_str if args.task == 'train' else args.ID,
        save_best_and_latest_only=True,
        val_path="diffusion_val/1eminus1/",
        tem_dict={'high':2.0 * 500.0, 'low':0.1 * 500.0},
    )
    lo = []
    vlo = []
    sep = []
    fi = 0
    if args.task == 'sample':
        trainer.load(ID=args.ID, milestone='fin')
    else:
        lo, vlo, sep, fi = trainer.train(Temperature)
    lo = torch.tensor(lo)
    fi = torch.tensor(fi)
    log_info(date_str, lr, ts, tbs, tns, sat, lo.mean().item(), fi.item(), file_name='ddpm_log.csv')
    x = np.linspace(0, tns, tns)
    plt.figure()
    plt.plot(x, lo, label='Train Loss')
    plt.plot(sep, vlo, label='Validation Loss')
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"results/{date_str}.svg")
    plt.close()
    print("plot complete")

    # print(lo)
    date_str = date_str if args.task == 'train' else args.ID
    sampled_images = diffusion.sample(batch_size = args.N, Temperature=Temperature).cpu()
    transform = transforms.ToPILImage()
    output_dir = f'results_sampled/{date_str}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'results_sampled/{date_str} subfolder not found, new one created')
    for i in range(sampled_images.size(0)):
        img_tensor = sampled_images[i]
        img=transform(img_tensor)
        img.save(os.path.join(output_dir, f'sampled_image_low_{i}.png'))
    print("sampling complete")

if __name__ == '__main__':
    main()

#['high':2.0, 'low':0.1]