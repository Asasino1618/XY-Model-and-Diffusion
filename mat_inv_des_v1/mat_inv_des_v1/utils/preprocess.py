import os
import random
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

import torch
from torch_geometric.data import Data
from cgcnn.data import CIFData
from utils.transforms import Normalizer

from ddpm.denoising_diffusion_pytorch_1d import Dataset1D_cond

def preprocess(data_dir, dataset_ratio=1, rewrite=False, seed=42):
    processed_path = os.path.join(data_dir, 'processed')

    # Create results folder to store models
    if not os.path.exists('results'):
        os.makedirs('results')
        print('results subfolder not found, new one created')

    # Create folder to store processed data
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    
    if (not os.listdir(processed_path)) or rewrite:  # Preprocess and save all the data
        if (not os.listdir(processed_path)):
            print('pre-processed data not found\n')
        # Load dataset
        dataset_pyg = CIFData(data_dir)
        dataset = []
        # Save files
        for data in tqdm(dataset_pyg, desc='Processing data'):
            torch.save(data, os.path.join(processed_path, f'{data.cif_id[0]}.pt'))
            dataset.append(data)
        
    else:  # Directly load the data
        dataset_length = sum(1 for f in os.listdir(processed_path) if f.endswith('.pt'))
        total_size = int(dataset_length * dataset_ratio)
        random.seed(seed)
        data_indices = random.sample(range(1, dataset_length + 1), total_size)

        dataset = [os.path.join(processed_path, f'{i}.pt') 
                   for i in data_indices]
        dataset = list(thread_map(load_file, dataset, desc='Loading data'))
    
    return dataset

def preprocess_ddpm(date, num_timesteps):
    encoder_path = os.path.join('results/', f'CGCNNPyG_{date}.pth')
    assert os.path.exists(encoder_path), 'model with specified ID does not exist'

    latents_path = os.path.join('results/latents', f'{date}')
    assert os.path.exists(latents_path), 'encoded data not found'

    files = [os.path.join(latents_path, f) 
               for f in os.listdir(latents_path) if f.endswith('.pt')]
    data = list(thread_map(load_data, files, desc='Loading data'))
    prop = list(thread_map(load_prop, files, desc='Loading prop'))
    data = torch.stack(data)
    prop = torch.stack(prop)
    
    sample = random.sample(range(len(data)), len(data)//2)
    sampled_data = torch.stack([data[i] for i in sample])
    sampled_prop = torch.stack([prop[i] for i in sample])
    max_data, min_data = torch.amax(sampled_data, dim=0), torch.amin(sampled_data, dim=0)
    max_prop, min_prop = torch.amax(sampled_prop, dim=0), torch.amin(sampled_prop, dim=0)

    data_normalizer = Normalizer(min_data, (max_data - min_data))
    prop_normalizer = Normalizer(min_prop, (max_prop - min_prop) / num_timesteps)

    data_normed = data_normalizer.norm(data).unsqueeze(1)
    prop_normed = prop_normalizer.norm(prop)

    dataset = Dataset1D_cond(data_normed, prop_normed)

    return dataset, data_normalizer, prop_normalizer

def load_file(file_path):
    return torch.load(file_path)

def load_data(file_path):
    return torch.load(file_path).cpu().x.squeeze()

def load_prop(file_path):
    return torch.load(file_path).cpu().prop.squeeze()