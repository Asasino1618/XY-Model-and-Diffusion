import argparse
import sys
import os
import random
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from cgcnn.data import get_train_val_test_loader

from cgcnn.model import CGCNN

from utils.vis import Result, log_info
from utils.preprocess import preprocess
from utils.transforms import Normalizer, loss_function, mae_function

parser = argparse.ArgumentParser(description='CGCNN with PyG')
parser.add_argument('data_options', type=str, metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, then other options')
parser.add_argument('--dataset-ratio', default=1, type=float,
                    metavar='N', help='ratio of dataset used for process (default: 1)')
parser.add_argument('--data-rewrite', default=0, type=int,
                    metavar='RW', help='pre-process data even existed')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run (default: 250)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--seed', default=42, type=int, metavar='W', 
                    help='random seed for torch (default: 42)')
parser.add_argument('--performance', default=1, type=int, metavar='Bool', 
                    help='pooling method (default: performance)')
parser.add_argument('--self-loop', default=0, type=int, metavar='Bool', 
                    help='add self loop in the graph (default: False)')
parser.add_argument('--element-emb', default=0, type=int, metavar='Bool', 
                    help='add element embedding to coords decoder (default: False)')
parser.add_argument('--encoder-act-name', default='ReLU', type=str, metavar='ACT', 
                    help='activation of encoder (options: ReLU, LeakyReLU, ELU)')
parser.add_argument('--decoder-act-name', default='LeakyReLU', type=str, metavar='ACT', 
                    help='activation of decoder MLPs (options: ReLU, LeakyReLU, ELU)')
parser.add_argument('--encoder-norm', default='batch', type=str, metavar='norm', 
                    help='normalization of encoder (options: batch, group, none)')
parser.add_argument('--decoder-norm', default='batch', type=str, metavar='norm', 
                    help='normalization of decoder (options: batch, group, none)')

train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=0.6, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')

valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.2, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.2)')

test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.2, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.2)')

parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=32, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

vis_group = parser.add_mutually_exclusive_group()
vis_group.add_argument('--log', default=1, type=int, metavar='L', 
                       help='log y axis (default: True)')

args = parser.parse_args(sys.argv[1:])

def main():
    global args

    date = datetime.now()
    date_str = date.strftime('%Y%m%d%H%M%S%f')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else torch.device('cpu')
    
    print(f'System: {str(sys.version)}\nPyTorch: {str(torch.__version__)}')
    print(f'Device: {device_name}')
    print('-'*60)
    
    data_dir = str(*args.data_options)
    dataset = preprocess(data_dir, 
                         dataset_ratio=args.dataset_ratio, 
                         rewrite=bool(args.data_rewrite), 
                         seed=args.seed
                         )

    if args.dataset_ratio > 1: 
        print('Dataset usage ratio cannot exceed 1, forced to 1')
        args.dataset_ratio = 1
    
    train_loader, val_loader, test_loader = get_train_val_test_loader(dataset, 
                                                                      batch_size=args.batch_size, 
                                                                      train_ratio=args.train_ratio, 
                                                                      val_ratio=args.val_ratio, 
                                                                      test_ratio=args.test_ratio
                                                                      )

    # Prepare the target normalizers
    sample = random.sample(range(len(dataset)), len(dataset)//2)

    props = torch.tensor([dataset[i].y for i in sample])
    prop_mean, prop_std = torch.mean(props), torch.std(props)

    coords = torch.stack([dataset[i].distances for i in sample]).to(device)
    coords_mean, coords_std = torch.mean(coords, dim=0), torch.std(coords, dim=0)
    
    prop_normalizer = Normalizer(prop_mean, prop_std)
    element_normalizer = Normalizer(0, 1)
    coords_normalizer = Normalizer(coords_mean, coords_std)

    normalizers = (prop_normalizer, coords_normalizer, element_normalizer)

    # Build model
    data = next(iter(dataset))
    orig_atom_fea_len = data.x.shape[-1]
    nbr_fea_len = 64
    num_atoms = data.x.shape[0]

    model = CGCNN(normalizers, 
                  orig_atom_fea_len, 
                  nbr_fea_len,
                  atom_fea_len=args.atom_fea_len, 
                  n_conv=args.n_conv, 
                  h_fea_len=args.h_fea_len, 
                  n_h=args.n_h, 
                  performance=bool(args.performance), 
                  num_atoms=num_atoms, 
                  element_emb=bool(args.element_emb), 
                  encoder_act_name=args.encoder_act_name, 
                  decoder_act_name=args.decoder_act_name, 
                  en_norm=args.encoder_norm, 
                  de_norm=args.decoder_norm
                  )
    
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_epoch = args.epochs

    # Set up the cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch)

    train_history = []
    val_history = []

    loss_min = float('inf')
    print('-'*60)
    print('Training in process:')
    
    pbar = tqdm(range(num_epoch), desc='Loss: 0.0000')
    for epoch in pbar:
        # Train
        # Return a 1 x 4 torch.tensor of [Loss, MAE1, MAE2, ACC]
        train_his = train_val_test_step(model, 
                                        loader=train_loader, 
                                        optimizer=optimizer,
                                        scheduler=scheduler,  
                                        normalizers=normalizers, 
                                        is_train=True, 
                                        is_test=False, 
                                        device=device, 
                                        date_str=date_str)
        
        train_history.append(train_his)

        # Validate
        # Return a 1 x 4 torch.tensor of [Loss, MAE1, MAE2, ACC]
        val_his = train_val_test_step(model, 
                                      loader=val_loader, 
                                      optimizer=optimizer, 
                                      scheduler=scheduler,
                                      normalizers=normalizers, 
                                      is_train=False, 
                                      is_test=False, 
                                      device=device, 
                                      date_str=date_str)
        
        val_history.append(val_his)

        loss_current = val_his[0]
        pbar.set_description(f'Loss: {loss_current:.4f}')

        if loss_current < loss_min:
            loss_min = loss_current
            # Save model checkpoint if the current loss is lower
            torch.save(model.state_dict(), f'results/model_checkpoint_{date_str}.pt')
        
    train_history = torch.stack(train_history)
    val_history = torch.stack(val_history)
    
    # Test
    # Return a tuple of (predictions[N x 2], targets[N x 2], maes[1 x 2])
    results = train_val_test_step(model, 
                                  loader=test_loader, 
                                  optimizer=optimizer, 
                                  scheduler=scheduler,
                                  normalizers=normalizers, 
                                  is_train=False, 
                                  is_test=True, 
                                  device=device, 
                                  date_str=date_str)
    
    history = (torch.t(train_history), torch.t(val_history))
    
    torch.save(model, f'results/CGCNNPyG_{date_str}.pth')
    os.remove(f'results/model_checkpoint_{date_str}.pt')

    vis = Result()
    vis.report(args, history, results=results, log=bool(args.log), date=date_str)

def train_val_test_step(model, 
                        loader, 
                        optimizer, 
                        scheduler, 
                        normalizers, 
                        is_train, 
                        is_test, 
                        device, 
                        date_str):
    tasks = ('regression', 'regression', 'classification')  # prop, coords, element

    if is_test:
        # Load the best model parameters
        model_name = f'results/model_checkpoint_{date_str}.pt'
        model.load_state_dict(torch.load(model_name))

    if is_train:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(is_train):
        loss_total = 0
        maes_total = torch.zeros(len(tasks))

        targets_out = [torch.empty(0,) for _ in range(2)]
        outputs_denormed_out = targets_out

        for data in loader:
            data.to(device)
            targets = (data.y, data.distances.view(-1, 11), data.elements)
            
            num_atoms = data.elements.shape[0] // data.y.shape[0]

            optimizer.zero_grad()

            # Normalize targets
            targets_normed = [normalizer.norm(target) for normalizer, target in zip(normalizers, targets)]
            
            # Compute outputs
            outputs_denormed = model(data)
            outputs_normed = [normalizer.norm(output) for normalizer, output in zip(normalizers, outputs_denormed)]

            # Commpute losses and maes
            losses = (loss_function(output, target_normed, task) 
                      for output, target_normed, task in zip(outputs_normed, targets_normed, tasks))

            maes = torch.tensor([mae_function(output_denormed, target, task, num_atoms) 
                                 for output_denormed, target, task in zip(outputs_denormed, targets, tasks)])
            
            # Multi-task loss
            loss_mt = sum(loss for loss in losses)
            
            if is_train:
                loss_mt.backward()
                optimizer.step()
                scheduler.step()
            
            loss_total += loss_mt.item()
            maes_total += torch.tensor([mae.item() for mae in maes])

            if is_test:
                targets_out = [torch.cat((empty, target.cpu()), dim=0) 
                               for empty, target in zip(targets_out, targets[:2])]
                outputs_denormed_out = [torch.cat((empty, output_denormed.cpu()), dim=0) 
                                        for empty, output_denormed in zip(outputs_denormed_out, outputs_denormed[:2])]
                
    loss_avg = torch.tensor(loss_total / len(loader)).unsqueeze(0)
    maes_avg = maes_total / len(loader)

    if is_test:
        print('-'*60)
        print(f'Test on the best model: Loss {loss_avg.squeeze():.4f}\t'
              f'Prediction MAE {maes_avg[0]:.3f}\t'
              f'Coordinates Decoding MAE {maes_avg[1]:.4f}\t'
              f'ACC {maes_avg[2]:.3f}')
        
        global args
        log_info(date_str, 
                 args, 
                 model, 
                 loss_avg, 
                 maes_avg, 
                 file_name='train_log.csv')
        
        out =  (outputs_denormed_out, 
                targets_out, 
                maes_avg)

    else:
        out =  torch.cat((loss_avg, maes_avg), dim=0)
    
    return out

import os

if __name__ == '__main__':
    main()