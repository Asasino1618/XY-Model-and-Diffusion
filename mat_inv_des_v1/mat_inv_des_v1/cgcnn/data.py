import csv
import functools
import json
import os
import warnings
import random

import numpy as np
from pymatgen.core.structure import Structure
from ase.io import read
from CifFile import ReadCif

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from utils.transforms import idx2adj_list, atom2one_hot, get_distances

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    def __init__(self, root_dir,  
                 max_num_nbr=12, 
                 radius=8, 
                 dmin=0, 
                 step=0.2):
        
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        
        self._indices = None

        adj_matrix = torch.tensor([[6, 8, 4, 4, 4], 
                                   [8, 6, 2, 2, 2], 
                                   [4, 2, 6, 4, 4], 
                                   [4, 2, 6, 4, 4], 
                                   [4, 2, 4, 6, 4]])
        self.edge_indices = adj_matrix.nonzero(as_tuple=False)
        self.weights = adj_matrix[self.edge_indices[:, 0], self.edge_indices[:, 1]]
        self.adj_list = self.edge_indices.repeat_interleave(self.weights, dim=0).t()

    def len(self):
        return len(self.id_prop_data)
    
    def get(self, idx):
        return self.__getitem__(idx)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, prop = self.id_prop_data[idx]

        # Get crystal structure from CIF file
        file_path = os.path.join(self.root_dir, cif_id+'.cif')

        # The Structure object is loaded by pymatgen
        crystal = Structure.from_file(file_path)
        
        # The coordinates are directly read in to avoid pbc
        cif_data = ReadCif(file_path)
        for block in cif_data:
            raw_sites = np.array([atom for atom in block.GetLoop('_atom_site_type_symbol')])
            
        coords_frac = np.array(raw_sites[:, 3:6], dtype=float)  # Fractional coordinates, 5x3 array
        
        # In some cases the sequence of atoms in Structure is different from cif file
        # Therefore, indices of sequence in the cif file are needed
        atoms_index = np.array(raw_sites[:, 1])
        
        # And a dictionary of coordinates as well
        sites_dict = {site.label: site for site in crystal}

        # Rearrange coordinates according to cif sequence 
        sites_list = [sites_dict[site] for site in atoms_index]
        crystal = Structure.from_sites(sites_list)
        
        # Distances between atoms as invariant edge features
        lattice_para = np.array(crystal.lattice.abc)  # Read lattice parameters, 1x3 array
        coords = coords_frac * lattice_para  # Absolute coordinates, 5x3 array
        
        # Compute distances between atoms, 5x5 array
        distances, distances_flattened = get_distances(coords)
        distances_flattened = np.append(distances_flattened, np.mean(lattice_para))
        
        # Add cubic lattice parameter as diag due to pbc
        distances = distances + np.eye(distances.shape[-1]) * np.mean(lattice_para)  
        distances = torch.from_numpy(distances).to(dtype=torch.float)
        
        # This method of edge feature construction performs worse, thus abendoned
        edge_features = distances[self.edge_indices[:, 0], self.edge_indices[:, 1]].repeat_interleave(self.weights, dim=0)

        # Atomic types
        atomic_numbers = [site.specie.number for site in crystal.sites]
        elements = atom2one_hot(atomic_numbers)
        
        # Atomic features
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        # The distances between each atom and its neighbors are computed within .get_all_neighbors()
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
                
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)

        nbr_fea = nbr_fea.reshape(-1, 1)
        
        atom_fea = torch.from_numpy(atom_fea).to(dtype=torch.float)
        nbr_fea = torch.from_numpy(nbr_fea).to(dtype=torch.float)
        nbr_fea_idx = idx2adj_list(nbr_fea_idx)

        cif_id=torch.tensor([int(cif_id)], dtype=torch.int)
        prop = torch.tensor([float(prop)], dtype=torch.float)
        lattice_para = torch.from_numpy(lattice_para).to(dtype=torch.float)
        coords = torch.tensor(coords, dtype=torch.float)
        distances_flattened = torch.from_numpy(distances_flattened).to(torch.float)

        data = Data(x=atom_fea, 
                    edge_index=nbr_fea_idx, 
                    edge_attr=nbr_fea, 
                    y=prop, 
                    cif_id=cif_id,
                    elements=elements,
                    distances=distances_flattened, 
                    coords=coords
                    )

        return data

def get_train_val_test_loader(dataset,  
                              batch_size=256, 
                              train_ratio=0.6, 
                              val_ratio=0.2, 
                              test_ratio=0.2):
    
    assert sum([train_ratio, val_ratio, test_ratio]) == 1, "train, val, test ratios must sum to 1"
    
    total_size = int(len(dataset))
    
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    # Split the dataset
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

