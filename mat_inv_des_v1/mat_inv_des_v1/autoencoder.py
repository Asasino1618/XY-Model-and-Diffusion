import argparse
import sys
import os
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

import torch

from torch_geometric.data import Data, DataLoader

from ase.io import write

from mendeleev import element

from utils.preprocess import preprocess, load_file
from utils.transforms import Normalizer, prob2one_hot, atoms2comp, tensors2atoms, distances2coords, rotation

parser = argparse.ArgumentParser(description='CGCNN Autoencoder')
parser.add_argument('data_options', type=str, metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, then other options')
parser.add_argument('--date', type=str, required=True, 
                    help='use date as the model id, yyyymmddhhmm')
parser.add_argument('--task', type=str, required=True, 
                    help='prediction, encoding, or decoding')
parser.add_argument('--rot-id', required=False, type=str)
parser.add_argument('--trans-id', required=False, type=str)
parser.add_argument('--rot-loop', type=int, default=5)

args = parser.parse_args(sys.argv[1:])

# Set up task choice
task_dic = {
    'predicting': (True, True),
    'encoding': (True, False),
    'decoding': (False, True)
}

def main():
    global args
    global task_dic

    date_str = args.date
    task = args.task
    encode, decode = iter(task_dic[task])
    
    print(f'Model ID: {date_str}\t Task: {task}')
    print('-'*60)

    assert os.path.exists('results'), 'results subfolder not found, a pretrained model is required'

    # Load trained model
    assert os.path.exists(f'results/CGCNNPyG_{date_str}.pth'), f'model {date_str} does not exist'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(f'results/CGCNNPyG_{date_str}.pth')
    model.to(device)
    model.eval()
    
    if encode & (not decode):  # Encoding mode, the output would contain a latent vector
        # Load dataset
        data_dir = str(*args.data_options)
        dataset = preprocess(data_dir)

        # Empty list to store latent tensors
        latents = []

        # Enumerate dataset and encode cif into 1D latent
        for data in tqdm(dataset, desc='encoding'):
            data.to(device)

            latent = autoencoder(data, model, encode, decode)
            latents.append(latent)

        # Check target folder
        latent_path = f'results/latents/{date_str}'
        if not os.path.exists(latent_path):
            os.makedirs(latent_path)
            print(f'latents/{date_str} subfolder not found, new one created')
        
        # Save latent tensor with a unique cif ID as filename
        thread_map(lambda latent: save_pt(latent, path=latent_path), latents, desc='Writing files')
        print(f'process completed, files saved at {latent_path}')
    
    if decode:  # If decoder is ON, the output would contain an ase.Atoms object
        
        assert args.rot_id, 'The model ID of rotation postprocess is required'
        assert args.trans_id, 'The model ID of translation postprocess is required'
        assert os.path.exists(f'./results/CGCNN_post_{args.rot_id}_Rotation.pth'), 'Pre-trained rotation model not found'
        assert os.path.exists(f'./results/CGCNN_post_{args.trans_id}_Translation.pth'), 'Pre-trained translation model not found'
        
        # Empty list to store resullts
        results = []

        if encode:  # Predicting mode
            # Load dataset
            data_dir = str(*args.data_options)
            dataset = preprocess(data_dir)
            for data in tqdm(dataset, desc='predicting'):
                data.to(device)
                result = autoencoder(data, model, encode, decode)
                results.append(result)
        
        else:  # Decoding mode, directly decoding from latents
            # Load and form dataset
            data_dir = str(*args.data_options)
            dataset = [os.path.join(data_dir, f) 
                       for f in os.listdir(data_dir) if f.endswith('.pt')]
            dataset = list(thread_map(load_file, dataset, desc='Loading data'))

            for data in tqdm(dataset, desc='decoding'):
                data.to(device)
                result = autoencoder(data, model, encode, decode)
                results.append(result)
        
        # Post-processing
        model_rot = torch.load(f'./results/CGCNN_post_{args.rot_id}_Rotation.pth').cuda().eval()
        model_trans = torch.load(f'./results/CGCNN_post_{args.trans_id}_Translation.pth').cuda().eval()
        structures = []
        for data in tqdm(results, desc='Post-proceessing'):
            structures.append(postprocess(model_rot, model_trans, data, loop=args.rot_loop))
        
        # Check target folder
        decoded_path = os.path.join(data_dir, 'decoded')
        if not os.path.exists(decoded_path):
            os.makedirs(decoded_path)
            print(f'decoded subfolder not found, new one created')

        # Save decoded CIF files with the same name as input latents
        thread_map(lambda result: save_cif(result, path=decoded_path), structures, desc='Writing files')
        print(f'process completed, files saved at {decoded_path}')
    
@torch.no_grad()
def autoencoder(data, model, encode=True, decode=True):
    out = model(data, encode, decode)
    
    if decode: 
        out = [tensor.cpu() for tensor in out]

    if decode:
        out[2] = torch.argmax(prob2one_hot(out[2]), dim=1)

    if encode & decode:
        out = Data(cif_id=data.cif_id, y=out[0], distances=out[1], elements=out[2], task='prediction')
    elif encode:
        out = Data(cif_id=data.cif_id, x=out, y=data.y, task='encoding')
    else:
        out = Data(cif_id=data.cif_id, y=out[0], distances=out[1], elements=out[2], target=data.y, task='decoding')

    return out

# Returns a ase.Atoms object for cif file writing
@torch.no_grad()
def postprocess(model_rot, model_trans, data, loop=5):
    cif_id=data.cif_id.item()
    formula = data.elements.numpy()
    prop = data.y
    coords = distances2coords(data.distances)
    cell = torch.eye(3) * data.distances.squeeze()[-1].item()  # 3x3 identity tensor
    lattice_para = torch.diag(cell)  # [3] vector

    # Normalize to relative coordinates for model input
    coords_rel = (coords / lattice_para).cuda().unsqueeze(0)
    coords_rel = coords_rel - torch.mean(coords_rel, dim=1, keepdim=True)

    # Predict rotation angles
    for i in range(loop):
        theta = model_rot(coords_rel)
        theta_norm = torch.tanh(theta) * (1 - 0.099 * i)
        R = rotation(theta_norm)
        coords_rot = coords_rel @ R
        coords_rel = coords_rot - torch.mean(coords_rot, dim=1, keepdim=True)

    # Predict translation vector
    delta = model_trans(coords_rel)
    coords_recon = (coords_rel + delta).squeeze().detach()
    
    return tensors2atoms(formula, lattice_para, coords_recon, prop, cif_id)

def save_pt(latent, path):
    torch.save(latent, os.path.join(path, f'{latent.cif_id[0]}.pt'))

def save_cif(atom, path):
    cif_id = atom.cif_id
    
    file_name = os.path.join(path, f'{cif_id}')

    # Write CIF file
    write(file_name + '.cif', atom)
    
    # Write txt file with property
    with open(file_name + '.txt', 'w') as file:
        file.write(f'{atom.prop}\n')

if __name__ == '__main__':
    main()