import os
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

from pymatgen.core.structure import Structure
from ase.formula import Formula
from mendeleev import element
from CifFile import ReadCif

from plotly.colors import qualitative
import plotly.graph_objects as go

from utils.transforms import distances2coords

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

class Result():
    def __init__(self):
        self.type = ['Loss', 'ACC']
        self.task = ['Prediction', 'Decoding']
        self.legend = ['Training', 'Validation']
        self.unit = ['eV/atom', 'Å']

    def trajectory(self, history, type='Loss', task='Prediction', legend='Training', log=True, is_acc=False):
        epoch = np.arange(self.num_epoch)
            
        if log:
            plt.semilogy(epoch, history, label=f'{legend}')
        else:
            plt.plot(epoch, history, label=f'{legend}')
        
        if is_acc:
            plt.ylim([0, 1])
        
        plt.xlabel('Epochs')
        plt.ylabel(type)
        plt.legend()
    
    def simplified_report(self, args, history, date):
        self.num_epoch = args.epochs
        fig = plt.figure(figsize=(15, 10))
        
        for i in range(2):
            his = history[i]
            self.trajectory(his, type=self.type[0], legend=self.legend[i], log=False)
        plt.tight_layout()
        
        return fig

    def report(self, args, history, date, results=None, log=True):
        self.num_epoch = args.epochs
        fig = plt.figure(figsize=(10, 15))
        
        if results != None:
            for i in range(2):
                plt.subplot2grid((3,2), (0,i))
                result = (results[j][i] for j in range(3))
                self.histogram(result, self.task[i], unit=self.unit[i])
        
        plt.subplot2grid((3,2),(1,0),colspan=2)
        for j in range(2):
            his = history[j][0]
            self.trajectory(his, type=self.type[0], legend=self.legend[j], log=log)
        
        plt.subplot2grid((3,2),(2,0),colspan=2)
        for j in range(2):
            his = history[j][3]
            self.trajectory(his, type=self.type[1], legend=self.legend[j], log=False, is_acc=True)
        
        plt.tight_layout()
        fig.savefig(f'results/CGCNNPyG_{date}.pdf')

    def histogram(self, result, task, unit):
        props, predictions, pred_mae = iter(result)
        try:
            props = torch.mean(props, dim=1)
            predictions = torch.mean(predictions, dim=1)
        except:
            pass
        
        plt.scatter(props, predictions, s=0.5, alpha=0.5)

        min_val = min(np.hstack([props, predictions]))
        max_val = max(np.hstack([props, predictions]))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, alpha=0.5)

        plt.title(f'{task} MAE = {pred_mae:.3f} ({unit})')

        plt.xlim([min_val, max_val])
        plt.ylim([min_val, max_val])

        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')

    def info(self, args, date):
        text = f"""
                date: {date}, lr = {args.lr}, wd = {args.weight_decay}, atom_fea_len: {args.atom_fea_len}, h_fea_len: {args.h_fea_len}, n_conv: {args.n_conv}, n_h: {args.n_h}, performance: {bool(args.performance)} 
                """
        return text

class CrystalStructure():
    def __init__(self):
        self.edges = [
            (0, 1), (1, 4), (4, 2), (2, 0),  # Base
            (0, 3), (1, 5), (2, 6), (4, 7),  # Sides
            (3, 5), (5, 7), (7, 6), (6, 3)   # Top
        ]
        self.color_scale = qualitative.Plotly
        self.color_map = {num: self.color_scale[i % len(self.color_scale)] 
                          for i, num in enumerate(range(86))}

    def create_sphere(self, center, radius, n_points=20):
        phi = np.linspace(0, 2 * np.pi, n_points)
        theta = np.linspace(0, np.pi, n_points)
        phi, theta = np.meshgrid(phi, theta)
        x = center[0] + radius * np.sin(theta) * np.cos(phi)
        y = center[1] + radius * np.sin(theta) * np.sin(phi)
        z = center[2] + radius * np.cos(theta)
        return x, y, z
    
    def plot_from_pt(self, file):
        data = torch.load(file)
        formula = data.elements.argmax(dim=1).numpy()
        coords = data.coords.numpy()
        cell = np.eye(3) * data.distances[-1].item()
        prop = data.y.item()
        
        self.plot(formula, coords, cell, prop)
        
    def plot_from_cif(self, file, prop=None):
        crystal = Structure.from_file(file)
        cif_data = ReadCif(file)
        for block in cif_data:
            raw_sites = np.array([atom for atom in block.GetLoop('_atom_site_type_symbol')])
            
        coords_frac = np.array(raw_sites[:, 3:6], dtype=float)  # Fractional coordinates, 5x3 array

        atoms_index = np.array(raw_sites[:, 1])
        sites_dict = {site.label: site for site in crystal}
        sites_list = [sites_dict[site] for site in atoms_index]
        crystal = Structure.from_sites(sites_list)
        
        cell = np.eye(3) * crystal.lattice.abc
        coords = coords_frac * crystal.lattice.abc
        formula = np.array([site.specie.number for site in crystal.sites])

        self.plot(formula, coords, cell, prop)
    
    def plot_from_data(self, data):
        formula = data.re[2].numpy()
        prop = data.re[0].item()
        coords = distances2coords(data.re[1])  # Absolute coordinates
        cell = np.eye(3) * data.re[1].squeeze()[-1].item()  # 3x3 identity tensor
        
        self.plot(formula, coords, cell, prop)

    def plot(self, formula, coords, cell, prop=None):  # cell should be a 3x3 diagonal matrix
        corners = np.array([np.zeros(3),  # Origin
                            cell[0],  # a
                            cell[1],  # b
                            cell[2],  # c
                            cell[0] + cell[1],
                            cell[0] + cell[2],
                            cell[1] + cell[2],
                            cell[0] + cell[1] + cell[2]])  # a + b + c
        corners = corners / 2

        atom_radius = np.array([element(int(atom)).covalent_radius/100 for atom in formula])

        fig = go.Figure()

        # Add edges to the plot
        for start, end in self.edges:
            line = go.Scatter3d(
                x=[corners[start][0], corners[end][0]],
                y=[corners[start][1], corners[end][1]],
                z=[corners[start][2], corners[end][2]],
                mode='lines',
                line=dict(color='black', width=1),
                hoverinfo='skip',
                showlegend=False
            )
            fig.add_trace(line)
        
        # Add atoms to the plot
        symbols = []
        for i, number in enumerate(formula):
            radius = atom_radius[i]
            color = self.color_map[number]
            center = coords[i]
            x, y, z = self.create_sphere(center, radius)
            symbol = element(int(number)).symbol
            symbols.append(symbol)
            
            sphere = go.Mesh3d(
                x=x.flatten(), y=y.flatten(), z=z.flatten(),
                alphahull=0,
                opacity=0.2,
                color=color
            )
            fig.add_trace(sphere)

            # Add atomic symbol
            fig.add_trace(go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                text=[symbol],
                mode='text',
                textposition='middle center',
                hoverinfo='skip',
                showlegend=False
            ))
        composition = Formula(''.join([symbols[i] for i in [1, 0, 2, 3, 4]])).format('reduce')
        # Define layout with hidden axis labels and ticks
        if prop:
            title = f'{composition}, property: {prop:.4f}'
        else:
            title = f'{composition}'
        layout = go.Layout(
            scene=dict(
                xaxis=dict(
                    title=f'{cell[0][0]:.3f} Å', 
                    showticklabels=False,  # Hide tick labels
                    tickvals=[]  # No tick values
                ),
                yaxis=dict(
                    title=f'{cell[1][1]:.3f} Å',
                    showticklabels=False,
                    tickvals=[]
                ),
                zaxis=dict(
                    title=f'{cell[2][2]:.3f} Å',
                    showticklabels=False,
                    tickvals=[]
                )
            ),
            title=title
        )

        fig.update_layout(layout)
        fig.show()

def log_info(date_str, args, model, loss_avg, maes_avg, file_name='train_log.csv'):
    # Define the header
    header = ["Model ID", "Dataset", "Dataset Ratio", "Atom Feature", 
              "Hidden Feature", "Convolution Layer", "Hidden Layer", 
              "Encoder Activation", "Decoder MLPs Activation", 
              "Encoder Normalization", "Decoder Normalization", 
              "Learning Rate", "Weight Decay", "Epoch", "Batch Size", 
              "Loss", "Property Prediction MAE",  
              "Coordinates Decoding MAE", "Composition Decoding ACC", "Notes"
              ]

    # Format the data
    data = [date_str, *args.data_options, args.dataset_ratio, model.atom_fea_len, 
            model.h_fea_len, model.n_conv, model.n_h, 
            args.encoder_act_name, args.decoder_act_name, 
            args.encoder_norm, args.decoder_norm, 
            args.lr, args.weight_decay, args.epochs, args.batch_size, 
            f"{loss_avg.squeeze():.4f}", f"{maes_avg[0]:.3f}", 
            f"{maes_avg[1]:.3f}", f"{maes_avg[2]:.3f}", args.element_emb]

    # Check if file exists and write the data
    file_exists = os.path.isfile(file_name)
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)  # write the header if file is new
        writer.writerow(data)
        
def post_log_info(date_str, args, model, train_loss, val_loss, note, file_name='postprocess_log.csv'):
    # Define the header
    header = ["Model ID", "Dataset Ratio", "Train Ratio"
              "Hidden Feature", "Hidden Layer", 
              "Activation", "Normalization", 
              "Learning Rate", "Weight Decay", "Epoch", "Batch Size", 
              "Training Loss", "Validation Loss", "Notes"
              ]

    # Format the data
    data = [date_str, args.dataset_ratio, args.train_ratio, 
            model.h_dimension, model.n_layer, 
            args.act_name, args.norm, 
            args.lr, args.weight_decay, args.epochs, args.batch_size, 
            f"{train_loss:.4f}", f"{val_loss:.4f}", note]

    # Check if file exists and write the data
    file_exists = os.path.isfile(file_name)
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)  # write the header if file is new
        writer.writerow(data)