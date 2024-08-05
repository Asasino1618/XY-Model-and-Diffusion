import argparse
import sys
import os
import random
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from cgcnn.model import InvSymmetry

from utils.transforms import distances2coords, rotation, distanceloss
from utils.vis import Result, post_log_info

parser = argparse.ArgumentParser(description='Post process of decoded data')
parser.add_argument('data_options', type=str, metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, then other options')
parser.add_argument('--dataset-ratio', default=1, type=float,
                    help='ratio of dataset used for process (default: 1)')
parser.add_argument('--data-rewrite', default=0, type=int,
                    help='pre-process data even existed')
parser.add_argument('--train-ratio', default=0.6, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run (default: 250)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--seed', default=42, type=int, 
                    help='random seed for torch (default: 42)')

parser.add_argument('--act-name', default='ReLU', type=str, 
                    help='activation function (options: ReLU, LeakyReLU, ELU)')
parser.add_argument('--norm', default='batch', type=str,
                    help='normalization (options: batch, group, none)')
parser.add_argument('--n-layer', default=5, type=int,
                    help='number of hidden layers')
parser.add_argument('--h-dim', default=64, type=int, 
                    help='dimension of hidden layers')

parser.add_argument('--rotation-loop', default=1, type=int)

parser.add_argument('--GPU', default=1, type=int)

args = parser.parse_args(sys.argv[1:])

# Set the visible GPU
torch.cuda.set_device(args.GPU)
data_dir = str(*args.data_options)
def data_prepare(path_save, data_dir):
    path = os.path.join(data_dir, 'processed')
    for data_path in tqdm(os.listdir(path), desc='Processing data: '):
        data = torch.load(os.path.join(path, data_path))
        
        cif_id = data.cif_id
        distances = data.distances
        cell = torch.ones(3) * distances[-1]
        
        # Rescale the ground truth coordinates
        coords = data.coords / cell
        
        # Ground truth coordinates are translated geometry center to origin as the translation training set and rotation targets
        coords_rel = coords - torch.mean(coords, dim=0)
        
        # distances2coords function generates misoriented coordinates as the rotation training set
        coords_de = distances2coords(distances.unsqueeze(0)) / cell
        coords_de = coords_de - torch.mean(coords_de, dim=0)
        
        coords_de = coords_de.to(torch.float)
        
        # Create Data objects
        data_rotation = Data(x=coords_de.unsqueeze(0), y=coords_rel.unsqueeze(0), cif_id=cif_id)
        data_translation = Data(x=coords_rel.unsqueeze(0), y=coords.unsqueeze(0), cif_id=cif_id)
        
        # Two sets are stored in two different directories seperately
        torch.save(data_rotation, os.path.join(path_save, 'rotation', f'{cif_id[0]}.pt'))
        torch.save(data_translation, os.path.join(path_save, 'translation', f'{cif_id[0]}.pt'))

def main():
    global args
    path_save = f'./post_process/data'
    
    # Check if the data has been prepared
    if (not os.path.exists(os.path.join(path_save, 'rotation', '1.pt'))) or bool(args.data_rewrite):
        data_prepare(path_save)
    
    date = datetime.now()
    date_str = date.strftime('%Y%m%d%H%M%S%f')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else torch.device('cpu')
    
    print(f'System: {str(sys.version)}\nPyTorch: {str(torch.__version__)}')
    print(f'Device: {device_name}')
    print("Current CUDA device:", torch.cuda.current_device())
    print('-'*60)
    
    # Initialize models
    model_rotation = InvSymmetry(n_atom=5, 
                        n_layer=args.n_layer, 
                        h_dimension=args.h_dim, 
                        norm_type=args.norm, 
                        act_name=args.act_name)
    
    model_translation = InvSymmetry(n_atom=5, 
                        n_layer=args.n_layer, 
                        h_dimension=args.h_dim, 
                        norm_type=args.norm, 
                        act_name=args.act_name)
    
    model_rotation.to(device)
    model_translation.to(device)

    # Load in data
    rotation_path = os.path.join(path_save, 'rotation')
    translation_path = os.path.join(path_save, 'translation')
    dataset_rotation = [torch.load(os.path.join(rotation_path, data_path)) for data_path in os.listdir(rotation_path) if data_path.endswith('.pt')]
    dataset_translation = [torch.load(os.path.join(translation_path, data_path)) for data_path in os.listdir(rotation_path) if data_path.endswith('.pt')]
    
    # Get random indices to shuffle the dataset
    dataset_length = len(dataset_rotation)
    dataset_ratio = args.dataset_ratio
    total_size = int(dataset_length * dataset_ratio)
    
    random.seed(args.seed)
    data_indices = random.sample(range(0, dataset_length), total_size)
    full_set_rotation = [dataset_rotation[i] for i in data_indices]
    full_set_translation = [dataset_translation[i] for i in data_indices]
    
    # Split the dataset into train and val
    train_ratio = args.train_ratio
    train_size = int(train_ratio * total_size)
    
    train_rotation = full_set_rotation[:train_size]
    val_rotation = full_set_rotation[train_size:]
    train_translation = full_set_translation[:train_size]
    val_translation = full_set_translation[train_size:]
    
    # Prepare data loaders
    batch_size = args.batch_size
    train_loader_rotation = DataLoader(train_rotation, batch_size=batch_size, follow_batch=['x', 'y'])
    val_loader_rotation = DataLoader(val_rotation, batch_size=batch_size, follow_batch=['x', 'y'])
    train_loader_translation = DataLoader(train_translation, batch_size=batch_size, follow_batch=['x', 'y'])
    val_loader_translation = DataLoader(val_translation, batch_size=batch_size, follow_batch=['x', 'y'])
    
    # Define loss function as the average distance between adjusted atoms and ground truth
    loss_func = distanceloss
    
    train_val_loop(train_loader_rotation, val_loader_rotation, model_rotation, loss_func, args.epochs, device, date_str, 'Rotation', args.rotation_loop)
    train_val_loop(train_loader_translation, val_loader_translation, model_translation, loss_func, args.epochs, device, date_str, 'Translation')
    

def train_val_loop(train_loader, val_loader, model, loss_func, num_epoch, device, date_str, task, rotation_loop=1):
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Set up learnung rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch)
    
    # Initialize 
    his_train, his_val = [], []
    
    # Training loop
    loss_val_min = float('inf')
    pbar = tqdm(range(num_epoch), desc=f'{task} Loss: 0.0000')
    for epoch in pbar:
        # Single train step
        loss_total = []
        for data in train_loader:
            data.to(device)
            coords_de, coords = data.x, data.y  # decoded coordinates, ground truth coordinates
            
            if task == 'Rotation':
                # Rotation pre-train
                batch_size = coords_de.shape[0]
                pre_theta_norm = torch.rand([batch_size, 3]) * 2 - 1
                pre_R = rotation(pre_theta_norm).to(device)
                pre_coords_de = coords_de @ pre_R
                
                pre_coords_de = pre_coords_de - torch.mean(pre_coords_de, dim=1, keepdim=True)
            else:
                pre_coords_de = coords_de
            
            # Add a small scale of noise to prevent overfitting
            noise = torch.rand_like(pre_coords_de) - 0.5
            pre_coords_de = pre_coords_de + 0.01 * noise
            
            optimizer.zero_grad()
            # Model prediction
            model_out = model(pre_coords_de)
            
            if task == 'Rotation':
                # Output is constrained to [-1, 1] representing [-180, 180]
                model_out = torch.tanh(model_out)
                
                # Construct rotation matrix
                rotation_mat = rotation(model_out)

                # Perform rotation
                coords_mod = pre_coords_de @ rotation_mat

                # Re-align to geometric center
                coords_out = coords_mod - torch.mean(coords_mod, dim=1, keepdim=True)

            else:
                # Perform translation
                model_out = model_out.unsqueeze(1)
                coords_out = coords_de + model_out
            
            # Compute loss
            loss = loss_func(coords, coords_out)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            loss_total.append(loss.item())
        
        # Record training loss
        his_train.append(sum(loss_total) / len(loss_total))
        
        # Single val step
        loss_total = []
        for data in val_loader:
            data.to(device)
            coords_de, coords = data.x, data.y
            with torch.no_grad():
                model.eval()
                
                # Predict rotation angles
                model_out = model(coords_de)

                if task == 'Rotation':
                    for i in range(rotation_loop):
                        # Output is constrained to [-1, 1] representing [-180, 180]
                        model_out = torch.tanh(model_out)
                        # Construct rotation matrix
                        rotation_mat = rotation(model_out)

                        # Perform rotation
                        coords_mod = coords_de @ rotation_mat

                        # Re-align to geometric center
                        coords_de= coords_mod - torch.mean(coords_mod, dim=1, keepdim=True)
                        
                    coords_out = coords_de
                else:
                    # Perform translation
                    model_out = model_out.unsqueeze(1)
                    coords_out = coords_de + model_out
                
                # Compute loss
                loss_val = loss_func(coords, coords_out)
                
                loss_total.append(loss_val.item())
        
        # Record vlidation loss
        his_val.append(sum(loss_total) / len(loss_total))
        
        # Update progress bar description
        pbar.set_description(f'{task} Loss: {his_train[-1]:.4f}')
        
        # Compare current validation loss
        if his_val[-1] < loss_val_min:
            loss_val_min = his_val[-1]
            loss_train_min = his_train[-1]
            torch.save(model.state_dict(), f'./results/model_checkpoint_{date_str}_{task}.pt')
    
    # Save training trajectory
    history = [his_train, his_val]
    
    # Load and then save the best model
    model_name = f'./results/model_checkpoint_{date_str}_{task}.pt'
    model.load_state_dict(torch.load(model_name))
    torch.save(model, f'./results/CGCNN_post_{date_str}_{task}.pth')
    os.remove(f'./results/model_checkpoint_{date_str}_{task}.pt')
    
    # Create plot
    vis = Result()
    report = vis.simplified_report(args, history, date=date_str)
    report.savefig(f'./results/CGCNN_post_{date_str}_{task}.pdf')
    
    # Save training log
    if task == 'Rotation':
        task += f' loop = {rotation_loop}'
    post_log_info(date_str, args, model, loss_train_min, loss_val_min, note=task, file_name='postprocess_log.csv')

if __name__ == '__main__':
    main()