import numpy as np

import torch
import torch.nn.functional as F

from sklearn.manifold import MDS

from mendeleev import element
from ase import Atoms

class AtomProps():
    def __init__(self):
        self.atomic_symbols = ['H', 'He', 
                        'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 
                        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 
                        'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
                        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 
                        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']

        atomic_radius = [0.0000, 0.4043, 0.5106, 0.3404, 0.2553, 0.1915, 0.1702, 0.1489, 0.1064,
                        0.5745, 0.6596, 0.5319, 0.4255, 0.3617, 0.3191, 0.3191, 0.3191, 0.1957,
                        0.8298, 0.6596, 0.5745, 0.4894, 0.4681, 0.4894, 0.4894, 0.4894, 0.4681,
                        0.4681, 0.4681, 0.4681, 0.4468, 0.4255, 0.3830, 0.3830, 0.3830, 0.5149,
                        0.8936, 0.7447, 0.6596, 0.5532, 0.5106, 0.5106, 0.4681, 0.4468, 0.4681,
                        0.4894, 0.5745, 0.5532, 0.5532, 0.5106, 0.5106, 0.4894, 0.4894, 0.5149,
                        1.0000, 0.8085, 0.7234, 0.6809, 0.6809, 0.6809, 0.6809, 0.6809, 0.6809,
                        0.6596, 0.6383, 0.6383, 0.6383, 0.6383, 0.6383, 0.6383, 0.6383, 0.5532,
                        0.5106, 0.4681, 0.4681, 0.4468, 0.4681, 0.4681, 0.4681, 0.5319, 0.7021,
                        0.6596, 0.5745, 0.7021, 0.5149, 0.5149]
        self.atomic_radius = torch.tensor(atomic_radius).cuda()

atom_prop = AtomProps()

# Convert edge indices matrix into adjacency list
def idx2adj_list(idx):
    num_nodes, num_edges = idx.shape[0], idx.shape[1]
    row, column = torch.arange(num_nodes), torch.ones(num_edges).reshape(-1, 1)
    start = (row*column).T.reshape(1, -1)
    end = torch.from_numpy(idx.reshape(1, -1))
    adj_list = torch.cat([start, end])
    adj_list = adj_list.to(dtype=torch.long)

    return adj_list

# Convert atomic numbers into one-hot vectors
def atom2one_hot(atomic_numbers):
    one_hot = torch.zeros(len(atomic_numbers), 86) # 7 periods in total
    for i, atom_number in enumerate(atomic_numbers):
        one_hot[i, atom_number] = 1

    return one_hot

def loss_function(output, target, task):
    if task == 'classification':
        return F.cross_entropy(output, target)
    elif task == 'regression':
        return F.mse_loss(output, target, reduction='mean')
    
def mae_function(output, target, task, num_atoms):
    if task == 'classification':
        return accuracy(output, target, num_atoms)
    elif task == 'regression':
        return F.l1_loss(output, target, reduction='mean')

# Convert probability output into 1-hot vector
def prob2one_hot(tensor):
    max_indices = torch.argmax(tensor, dim=1)
    one_hot_matrix = torch.zeros_like(tensor)
    one_hot_matrix[torch.arange(tensor.size(0)), max_indices] = 1
    return one_hot_matrix

# Compress atom 1-hot vectors into one 86D vector, where the numbers indicate the composition
def atoms2comp(atom_tensor, num_atoms):
    composition = atom_tensor.view(-1, num_atoms, atom_tensor.size(1)).sum(dim=1)
    return composition

# Compute composition decoding accuracy
def accuracy(output, label, num_atoms):
    output = atoms2comp(prob2one_hot(output), num_atoms)
    label = atoms2comp(label, num_atoms)

    not_match = (output != label).sum(dim=1) 
    acc = (not_match == torch.zeros_like(not_match)).sum().item() / len(not_match)
    return acc

# Convert 1x86 composition vector into chemical formula
def comp2formula(comp):
# Constructing the chemical formula
    chemical_formula = ""

    for atomic_num, count in enumerate(comp, start=1):
        if count > 0:
            # Get the element symbol using Mendeleev
            atom = element(atomic_num)
            # Append the element symbol and its count (if more than 1) to the formula
            chemical_formula += atom.symbol + (str(int(count)) if count > 1 else "")

    return chemical_formula

def tensors2atoms(formula, lattice, coords, props, cif_id, atomic_symbols=atom_prop.atomic_symbols):
    # Convert PyTorch tensors to numpy arrays
    lattice_np = lattice.cpu().numpy()
    coords_np = coords.cpu().numpy()

    # Map negative values back to zeros
    min_values = coords_np.min(axis=0)
    translation = np.where(min_values < 0, -min_values, 0)
    translated_coords = coords_np + translation

    # Create the lattice cell
    lattice_cell = np.diag(lattice_np)

    # To ensure efficiency
    # the atomic numbers will not be converted into symbols in the decoder
    formula = [atomic_symbols[int(atom.item())-1] for atom in formula]

    # Create the Atoms object
    atoms = Atoms(symbols=formula, 
                  scaled_positions=translated_coords, 
                  cell=lattice_cell, 
                  pbc=True
                  )
    atoms.prop = props.item()
    atoms.cif_id = cif_id

    return atoms

def atoms2radius(tensor, num_atoms, atomic_radius=atom_prop.atomic_radius):
    index = torch.argmax(tensor, dim=1)
    radius = atomic_radius[index]

    return radius.view(-1, num_atoms)

idx = np.zeros((5, 5), dtype=int)
id = 0 
for i in range(5): 
    for j in range(i + 1, 5):
        idx[i, j] = id
        id += 1 
idx = idx + idx.T  # Mirror the upper part to the lower part
np.fill_diagonal(idx, -1)

def distances2coords(distances, indices=idx):
    if distances.shape != distances.T.shape:
        distances = distances.squeeze().cpu().numpy()
        distances = distances[indices]
    np.fill_diagonal(distances, 0)
    
    # Classical Multidimensional Scaling
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=0, normalized_stress='auto')
    coordinates = mds.fit_transform(distances)
    
    return torch.from_numpy(coordinates).to(dtype=torch.float)

class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean
    
def get_distances(coordinates):
    coords = np.tile(coordinates, (coordinates.shape[0], 1, 1))
    diff = coords - coords.transpose(1, 0, 2)
    distances = np.sqrt(np.sum(diff**2, axis=2))
    distances_flattened = np.triu(distances)
    distances_flattened = distances_flattened[distances_flattened != 0]

    return distances, distances_flattened

def rotation(theta_norm):
    # Convert degrees to radians
    pi = torch.tensor(torch.pi)
    theta = theta_norm * pi
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    # Expand dimensions for batch processing
    cos = cos.unsqueeze(-1)
    sin = sin.unsqueeze(-1)

    # Define rotation matrices for each batch element
    R_x = torch.stack([
        torch.stack([torch.ones_like(cos[:, 0]), torch.zeros_like(cos[:, 0]), torch.zeros_like(cos[:, 0])], dim=-1),
        torch.stack([torch.zeros_like(cos[:, 0]), cos[:, 0], -sin[:, 0]], dim=-1),
        torch.stack([torch.zeros_like(cos[:, 0]), sin[:, 0], cos[:, 0]], dim=-1)
    ], dim=-2)

    R_y = torch.stack([
        torch.stack([cos[:, 1], torch.zeros_like(cos[:, 1]), sin[:, 1]], dim=-1),
        torch.stack([torch.zeros_like(cos[:, 1]), torch.ones_like(cos[:, 1]), torch.zeros_like(cos[:, 1])], dim=-1),
        torch.stack([-sin[:, 1], torch.zeros_like(cos[:, 1]), cos[:, 1]], dim=-1)
    ], dim=-2)

    R_z = torch.stack([
        torch.stack([cos[:, 2], -sin[:, 2], torch.zeros_like(cos[:, 2])], dim=-1),
        torch.stack([sin[:, 2], cos[:, 2], torch.zeros_like(cos[:, 2])], dim=-1),
        torch.stack([torch.zeros_like(cos[:, 2]), torch.zeros_like(cos[:, 2]), torch.ones_like(cos[:, 2])], dim=-1)
    ], dim=-2)

    # Matrix multiplication for rotation for each batch element
    R = torch.matmul(torch.matmul(R_z, R_y), R_x)
    
    return R.squeeze()
        
def distanceloss(coordinates_recon, coordinates):
    diff = coordinates_recon - coordinates
    return torch.mean(torch.sqrt(torch.sum(diff**2, dim=-1)))