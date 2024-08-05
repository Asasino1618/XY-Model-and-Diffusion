import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool

from utils.transforms import atoms2radius

class AutoDiscretization(nn.Module):
    def __init__(self, embedding_dim=64, num_bins=100):
        super().__init__()
        
        # Parameters
        self.embedding_dim = embedding_dim
        self.num_bins = num_bins

        # Layers
        self.linear1 = nn.Linear(1, num_bins)
        self.leaky_relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(num_bins, num_bins)
        
        # Scaling mixture factor (alpha)
        self.alpha = nn.Parameter(torch.rand(1))

        # Lookup table T
        self.lut = nn.Parameter(torch.rand(num_bins, embedding_dim))

    def forward(self, v):
        # Linear transformation
        v1 = self.linear1(v)

        # Leaky ReLU activation
        v2 = self.leaky_relu(v1)

        # Cross-layer projection
        v3 = self.linear2(v2) + self.alpha * v2

        # Softmax for normalization
        v4 = F.softmax(v3, dim=-1)

        # Weighted combination of embeddings
        e = v4 @ self.lut

        return e

class CGConv(MessagePassing):
    def __init__(self, 
                 atom_fea_len, 
                 nbr_fea_len, 
                 nbr_emb_bin=100, 
                 act_name='ReLU', 
                 norm_type='batch'
                 ):
        super(CGConv, self).__init__(aggr='add')
        
        act_funcs = {
            'ReLU': F.relu, 
            'LeakyReLU': F.leaky_relu, 
            'ELU': F.elu
        }

        assert act_name in act_funcs, f"Unsupported activation function: {act_name}"

        self.activation = act_funcs[act_name]

        norm_layers = {
            'batch': lambda in_fea_len: nn.BatchNorm1d(in_fea_len),
            'group': lambda in_fea_len: nn.GroupNorm(in_fea_len // 8, in_fea_len),
            'none': lambda _: nn.Identity()
        }

        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        
        self.fea_emb = AutoDiscretization(embedding_dim=nbr_fea_len, num_bins=nbr_emb_bin)
        
        self.fc_full = torch.nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len,
                                       2 * self.atom_fea_len)

        self.bn1 = norm_layers[norm_type](2 * atom_fea_len)
        self.bn2 = norm_layers[norm_type](atom_fea_len)
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, x_i, edge_attr):
        # x_j: atom_nbr_fea
        # x_i: atom_in_fea
        # edge_attr: nbr_fea
        edge_attr = self.fea_emb(edge_attr)
        
        # Concatenate the features
        total_nbr_fea = torch.cat([x_i, x_j, edge_attr], dim=1)
        
        # Apply linear transformation
        total_gated_fea = self.fc_full(total_nbr_fea)

        # Batch normalization
        total_gated_fea = self.bn1(total_gated_fea)
        
        # Split the features
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=1)
        
        # Apply activation functions
        nbr_filter = self.activation(nbr_filter)
        nbr_core = self.activation(nbr_core)
        
        # Element-wise multiplication
        return nbr_filter * nbr_core

    def update(self, aggr_out, x):
        # aggr_out: nbr_sumed
        # x: atom_in_fea
        # Batch normalization
        nbr_sumed = self.bn2(aggr_out)
        
        # Apply activation function
        out = F.relu(x + nbr_sumed)
        return out

class AttentionLayer(nn.Module):
    def __init__(self, orig_atom_fea_len):
        super().__init__()
        self.attention = nn.Parameter(torch.ones(orig_atom_fea_len))
    
    def forward(self, data):
        return data * self.attention
    
class CGCEncoder(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, 
                 performance=True, 
                 act_name='ReLU', 
                 norm_type='batch'
                 ):
        super().__init__()

        # Attention layer
        self.attention = AttentionLayer(orig_atom_fea_len)

        # Node feature embedding
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        # n_conv convolution layers
        self.convs = nn.ModuleList([CGConv(atom_fea_len=atom_fea_len, 
                                           nbr_fea_len=nbr_fea_len, 
                                           act_name=act_name, 
                                           norm_type=norm_type) 
                                           for _ in range(n_conv)])
        
        # Pooling layer
        self.performance = performance
        self.n_conv = n_conv
        if self.performance:
            self.pre_pooling = nn.ModuleList([nn.Linear(atom_fea_len, atom_fea_len), nn.Softmax(dim=1)])

    def forward(self, data):
        atom_fea, edge_index, nbr_fea, crys_idx, batch = data.x, data.edge_index, data.edge_attr, data.cif_id, data.batch
    
        batchsize = len(crys_idx)
        num_atom = atom_fea.shape[0] // batchsize

        # Attention layer
        atom_fea = self.attention(atom_fea)

        # Feature embedding layer        
        atom_fea = self.embedding(atom_fea)

        if self.performance:
            atom_feas = atom_fea
        
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, edge_index, nbr_fea)
            if self.performance:
                atom_feas = torch.cat((atom_feas, atom_fea), 0)
        
        # Pool to form crystal features
        if self.performance:
            for module in self.pre_pooling: # Linear transformation and activation of each atom features
                atom_feas = module(atom_feas)

            atom_feas = atom_feas.view(self.n_conv+1, batchsize, num_atom, -1)
            atom_feas = torch.sum(atom_feas, dim=0)
            crys_fea = torch.sum(atom_feas, dim=1) # Atom features in each layer are summed together to form crystal feature
        else:
            crys_fea = global_mean_pool(atom_fea, batch)
        
        out = crys_fea

        return out

class MLP(nn.Module):
    def __init__(self, in_fea_len=64, 
                 h_fea_len=32, 
                 n_h=2, 
                 out_fea_len=1, 
                 act_name='LeakyReLU', 
                 norm_type='none'
                 ):
        super().__init__()

        # Set up activation function for MLP

        act_funcs = {
            'ReLU': nn.ReLU(), 
            'LeakyReLU': nn.LeakyReLU(), 
            'ELU': nn.ELU()
        }

        assert act_name in act_funcs, f"Unsupported activation function: {act_name}"

        self.activation = act_funcs[act_name]

        norm_layers = {
            'batch': lambda in_fea_len: nn.BatchNorm1d(in_fea_len),
            'group': lambda in_fea_len: nn.GroupNorm(in_fea_len // 8, in_fea_len),
            'none': lambda _: nn.Identity()
        }

        # n_h linear transformation layers after pooling
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_fea_len, h_fea_len))
        self.layers.append(norm_layers[norm_type](h_fea_len))
        self.layers.append(self.activation)
        
        if n_h > 1:
            for _ in range(1, n_h):
                self.layers.append(nn.Linear(h_fea_len, h_fea_len))
                self.layers.append(norm_layers[norm_type](h_fea_len))
                self.layers.append(self.activation)

        # Output layer for prediction
        self.fc_out = nn.Linear(h_fea_len, out_fea_len)
    
    def forward(self, hidden_fea):
        # Linear transformation of crystal features for lattice parameter decoding
        for layer in self.layers:
            hidden_fea = layer(hidden_fea)

        # Output of prediction and decoding
        out = self.fc_out(hidden_fea)

        return out
    
class Classifier(nn.Module):
    def __init__(self, in_fea_len=64, 
                 h_fea_len=32, 
                 n_h=1, 
                 num_atoms=5, 
                 num_classes=86):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_classes = num_classes

        # n_h linear transformation layers after pooling
        self.n_h = n_h
        if n_h > 1:
            self.layers = nn.ModuleList()

            self.layers.append(nn.Linear(in_fea_len*num_atoms, h_fea_len*num_atoms))
            self.layers.append(nn.ReLU())
        
        if n_h > 2:
            for _ in range(1, n_h - 1):
                self.layers.append(nn.Linear(h_fea_len*num_atoms, h_fea_len*num_atoms))
                self.layers.append(nn.ReLU())

        # Output layer for prediction
        if n_h == 1:
            h_fea_len = in_fea_len
        self.fc_out = nn.Linear(h_fea_len*num_atoms, num_classes * num_atoms)

    def forward(self, hidden_fea):
        batch_size = hidden_fea.shape[0]
        # Repeat crystal latent
        hidden_fea = hidden_fea.repeat(1, self.num_atoms)

        # Linear transformation of crystal features for lattice parameter decoding
        if self.n_h > 1:
            for layer in self.layers:
                hidden_fea = layer(hidden_fea)

        # Output of prediction and decoding
        out = self.fc_out(hidden_fea)
        out = out.view(-1, self.num_classes)
        out = F.log_softmax(out, dim=1)
        return out
    
class CGCNN(nn.Module):
    def __init__(self, normalizers, 
                 orig_atom_fea_len, 
                 nbr_fea_len,
                 atom_fea_len=64, n_conv=3, 
                 h_fea_len=32, n_h=1, 
                 performance=True, 
                 num_atoms=5, 
                 element_emb=False, 
                 encoder_act_name='ReLU', 
                 decoder_act_name='LeakyReLU', 
                 en_norm='batch', 
                 de_norm='batch'
                 ):
        
        super(CGCNN, self).__init__()

        self.atom_fea_len = atom_fea_len
        self.h_fea_len = h_fea_len
        self.n_conv = n_conv
        self.n_h = n_h

        self.encoder_act_name = encoder_act_name
        self.decoder_act_name = decoder_act_name

        # Define encoder
        self.encoder = CGCEncoder(orig_atom_fea_len, nbr_fea_len, atom_fea_len, n_conv, performance, 
                                  act_name=encoder_act_name, norm_type=en_norm)

        # Define decoders
        self.decoder_prop = MLP(atom_fea_len, h_fea_len, n_h, out_fea_len=1, 
                                act_name=decoder_act_name, norm_type=de_norm)
        self.decoder_element = Classifier(atom_fea_len, h_fea_len, n_h, num_atoms, num_classes=86)
        
        self.element_emb = element_emb

        if self.element_emb:
            dim_emb = 32
            self.element_embedding = MLP(num_atoms, 32, 1, dim_emb, act_name=decoder_act_name)
            self.decoder_coords = MLP((atom_fea_len+dim_emb)*11, h_fea_len*11, n_h, 
                                      out_fea_len=11, 
                                      act_name=decoder_act_name, norm_type=de_norm)
        else:
            self.decoder_coords = MLP(atom_fea_len*11, h_fea_len*11, n_h, 
                                      out_fea_len=11, 
                                      act_name=decoder_act_name, norm_type=de_norm)

        # Define normalizers
        self.normalizers = normalizers

        self.num_atoms = num_atoms

    def forward(self, data, encode=True, decode=True):
        if encode:
            # Encoding graph into latent vector
            crys_fea = self.encoder(data)
        else:
            crys_fea = data.x

        if decode:
            # Decoding latent using MLPs
            out_prop = self.decoder_prop(crys_fea)
            out_element = self.decoder_element(crys_fea)

            if self.element_emb:
                element_fea = atoms2radius(out_element, self.num_atoms)
                element_fea = (element_fea / self.atom_fea_len).cuda()
                element_emb = self.element_embedding(element_fea)
                in_coords = torch.cat((crys_fea, element_emb), dim=1).repeat(1, 11)
            else:
                in_coords = crys_fea.repeat(1, 11)

            out_coords = self.decoder_coords(in_coords)

            out = (torch.squeeze(out_prop), 
                   out_coords, 
                   out_element
                   )
            # Denormalize outputs
            out = [normalizer.denorm(output) for normalizer, output in zip(self.normalizers, out)]
        else:
            out = crys_fea
            
        return out

class InvSymmetry(nn.Module):
    def __init__(self, 
                 n_atom=5, 
                 n_layer=5, 
                 h_dimension=64, 
                 norm_type='batch', 
                 act_name='LeakyReLU'):
        super().__init__()
        
        act_funcs = {
            'ReLU': nn.ReLU(), 
            'LeakyReLU': nn.LeakyReLU(), 
            'ELU': nn.ELU()
        }
        
        self.activation = act_funcs[act_name]
        
        self.n_atom = n_atom
        self.n_layer = n_layer
        self.h_dimension = h_dimension
        
        norm_layers = {
            'batch': lambda in_fea_len: nn.BatchNorm1d(in_fea_len),
            'group': lambda in_fea_len: nn.GroupNorm(in_fea_len // 8, in_fea_len),
            'none': lambda _: nn.Identity()
        }
        
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(n_atom * 3, h_dimension))
        self.layers.append(norm_layers[norm_type](h_dimension))
        self.layers.append(self.activation)
        
        if n_layer > 1:
            for _ in range(1, n_layer):
                self.layers.append(nn.Linear(h_dimension, h_dimension))
                self.layers.append(norm_layers[norm_type](h_dimension))
                self.layers.append(self.activation)
                
        self.fc_out = nn.Linear(h_dimension, 3)
        
    def forward(self, coordinates): # Take an input of Nx3 tensor
        batch_size = coordinates.shape[0]
        coords = coordinates.view(batch_size, -1)
        
        for layer in self.layers:
            coords = layer(coords)

        out = self.fc_out(coords)

        return out
