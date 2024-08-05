import os
import sys
import argparse
import csv
import itertools

from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from pymatgen.io.cif import CifParser
from pymatgen.analysis.structure_matcher import StructureMatcher

import smact
from smact.screening import pauling_test

parser = argparse.ArgumentParser(description='evaluation')
parser.add_argument('--origin', type=str, required=True, 
                    help='Path to original data')
parser.add_argument('--target', type=str, required=True,  
                    help='Path to data needs to be evaluated')
parser.add_argument('--notes', type=str, default=None)
parser.add_argument('--match', type=int, default=1, 
                    help='Whether to mathch two provided dataset')

args = parser.parse_args()

def main():
    global args
    
    path_origin = args.origin
    path_target= args.target
    
    do_match = bool(args.match)
    if do_match:
        assert num_origin == num_target; "Data amounts mismatch, unavailable for matching!"
        # Initialize StructureMatcher
        matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)  # Criteria from DP-CDVAE
        match = []
        dis_rms = []
    
    # Wasserstein distances
    print('Computing Wasserstein distances: \n')
    prop_origin = pd.read_csv(os.path.join(path_origin, 'id_prop.csv'), header=None)[1]
    prop_target = pd.read_csv(os.path.join(path_target, 'id_prop.csv'), header=None)[1]
    prop_dis = wasserstein_distance(prop_origin, prop_target)
    
    cif_origin = sorted([file for file in os.listdir(path_origin) if file.endswith('.cif')])
    cif_target = sorted([file for file in os.listdir(path_target) if file.endswith('.cif')])
    
    structure_origin = [CifParser(os.path.join(path_origin, cif)).parse_structures(primitive=True)[0] for cif in cif_origin]
    structure_target = [CifParser(os.path.join(path_target, cif)).parse_structures(primitive=True)[0] for cif in cif_target]
    
    density_origin = [structure.density for structure in structure_origin]
    density_target = [structure.density for structure in structure_target]
    density_dis = wasserstein_distance(density_origin, density_target)
    
    atoms_origin = [structure.num_sites for structure in structure_origin]
    atoms_target = [structure.num_sites for structure in structure_target]
    atoms_dis = wasserstein_distance(atoms_origin, atoms_target)
    
    elem_origin = np.array([[site.specie.Z for site in structure] for structure in structure_origin]).flatten()
    elem_target = np.array([[site.specie.Z for site in structure] for structure in structure_target]).flatten()
    elem_dis = wasserstein_distance(elem_origin, elem_target)

    # Match or/and verify structures
    num_origin = len(cif_origin)
    num_target = len(cif_target)
        
    val_stru = []
    val_comp = []

    for idx in tqdm(range(num_target), desc="Verifying structures"):
        # Load target structure
        structure_b = structure_target[idx]
        
        # Varify cell validity
        val_stru.append(validity_stru(structure_b))
        val_comp.append(validity_comp(structure_b))

        if do_match:
            # Read in ground truth structure
            structure_a = structure_origin[idx]
            
            # Compare the structures
            matched = matcher.fit(structure_a, structure_b)
            match.append(matched)
            if matched:  # Compute normalized RMS distances
                dis_rms.append(np.sqrt(np.sum((np.array(structure_a.cart_coords) - np.array(structure_b.cart_coords)) ** 2, axis=1)) 
                                               / np.cbrt(structure_a.volume / structure_a.num_sites))
    
    if do_match:
        match_rate = str(f'{sum(match) * 100 / len(match):.2f}')
        avg_dis_rms = str(f'{np.average(dis_rms):.4f}')
    else:
        match_rate = 'N/A'
        avg_dis_rms = 'N/A'
    
    val_stru = sum(val_stru) * 100 / len(val_stru)
    val_comp = sum(val_comp) * 100 / len(val_comp)
    
    print(f'Match Rate: {match_rate} %\t', 
          f'Average Distance RMS: {avg_dis_rms}\t', 
          f'Structural Validity: {val_stru:.2f} %\t', 
          f'Compositional Validity: {val_comp:.2f} %\n', 
          f'Property distance: {prop_dis:.4f}\t', 
          f'Density distance: {density_dis:.4f}\t',
          f'Atoms distance: {atoms_dis:.4f}\t', 
          f'Elements distance: {elem_dis:.4f}\t')
    
    # Define the header
    header = ["Ground Truth Data", "Evaluated Data", "Match Rate", "Average Distance RMS", 
              "Val Stru.", "Val Comp.", 
              "d Property", "d Density", "d Atoms", "d Elements",
              "Notes"]

    # Format the data
    info = [path_origin, path_target, match_rate, avg_dis_rms, 
            val_stru, val_comp, 
            prop_dis, density_dis, atoms_dis, elem_dis, 
            args.notes]

    # Check if file exists and write the data
    file_name = 'evaluation_log.csv'
    file_exists = os.path.isfile(file_name)
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)  # write the header if file is new
        writer.writerow(info)


def validity_comp(structure,
                  use_pauling_test=True,
                  include_alloys=True):
    ele_dict = structure.composition.get_el_amt_dict()
    elem_symbols = tuple(symbol for symbol in ele_dict)
    count = [int(ele_dict[symbol]) for symbol in ele_dict]
    
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(
                        tuple([elem_symbols, ox_states, ratio]))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    if len(compositions) > 0:
        return True
    else:
        return False
    
def validity_stru(structure, cutoff=0.5):
    all_nbrs = structure.get_all_neighbors(cutoff, include_index=True)
    return not bool(sum([nbr != [] for nbr in all_nbrs]))

if __name__ == '__main__':
    main()