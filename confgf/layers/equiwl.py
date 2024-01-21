# coding=utf-8
import sys, os

from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing

from .common import MeanReadout, SumReadout, MultiLayerPerceptron
from .gin import GINEConv
from .gat import Transformer_layer
from .wl import TwoFDisInit, TwoFDisLayer

#import DisGNN code
from .DisGNN.layers.Mol2Graph import Mol2Graph
from .DisGNN.utils.loss_fns import loss_fn_map
from .DisGNN.utils.activation_fns import activation_fn_map
from .DisGNN.utils.EMA import ExponentialMovingAverage #TODO: insert EMA in trainging

from .DisGNN import kDisGNN


from .DisGNN.scripts.script_utils import get_cfgs

class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class EquiLayer(MessagePassing):

    def __init__(self, eps: float = 0., train_eps: bool = False,
                 activation="softplus", **kwargs):
        super(EquiLayer, self).__init__(aggr='add', **kwargs)
        self.initial_eps = eps

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None       

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            # assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor: 
        if self.activation:
            return self.activation(x_j + edge_attr) # self.activation(edge_attr)(x_j-x_i)
        else:
            # return x_j + edge_attr
            return edge_attr

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class GradientGEGNNWL(torch.nn.Module):

    def __init__(self, hidden_dim, hidden_coff_dim=64, num_convs=3, activation="softplus", readout="sum", short_cut=False, concat_hidden=False, color_steps=1):
        super(GradientGEGNNWL, self).__init__()
        
        
        '''
            get hparams
        '''
        model_name = "2FDis"
        self.model_name = model_name
        dataset_name = "qm9"
        self.dataset_name = dataset_name
        data_name = "ethanol" #default is ethanol, doesnt make a difference
        self.data_name = data_name
        config_path = "/home/snirhordan/ClofNetOrig/confgen/confgf/layers/DisGNN/config/{}_{}.yaml".format(model_name, dataset_name) #TODO ERASE NAMES!
        specific_config_path = "hparams/specific/{}_{}_specific.yaml".format(model_name, dataset_name)
        config = get_cfgs(config_path, None, None, data_name)
        
        #print("-"*20)
        #print(config)
        #print("-"*20)
        
        model_config = config.model_config

        self.hidden_dim = hidden_dim
        # self.num_convs = num_convs
        self.num_layers = 2
        self.num_convs = 2
        self.short_cut = short_cut
        self.num_head = 8
        self.dropout = 0.1
        self.concat_hidden = concat_hidden
        self.hidden_coff_dim = hidden_coff_dim
        self.dist_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None 
        self.radial_mlp = nn.Linear(2 *self.hidden_dim, self. hidden_dim)
        # # of steps
        self.color_steps = color_steps
        #initical coloring
        self.init_color = TwoFDisInit(ef_dim=self.hidden_dim, k_tuple_dim=self.hidden_dim)  
        #color update steps
        self.interaction_layers = nn.ModuleList()
        for _ in range(color_steps):
            self.interaction_layers.append(
                    TwoFDisLayer(
                        hidden_dim=self.hidden_dim,
                        )
                    )
        
        # self.conv_modules = nn.ModuleList()
        self.transformers = nn.ModuleList()
        self.equi_modules = nn.ModuleList()
        self.dynamic_mlp_modules = nn.ModuleList()
        for _ in range(self.num_layers):
            trans_convs = nn.ModuleList()
            for i in range(self.num_convs):
                trans_convs.append(
                    Transformer_layer(self.num_head, self.hidden_dim, dropout=self.dropout, activation=activation)
                )
            # self.conv_modules.append(convs)
            self.transformers.append(trans_convs)
            self.equi_modules.append(EquiLayer(activation=False))
            self.dynamic_mlp_modules.append(
                nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_coff_dim),
                nn.Softplus(),
                nn.Linear(self.hidden_coff_dim, 1)) # scarlar output
            )
        self.model = kDisGNN(
            z_hidden_dim=model_config.z_hidden_dim,
            ef_dim=model_config.ef_dim,
            rbf=model_config.rbf,
            max_z=model_config.max_z,
            rbound_upper=model_config.rbound_upper,
            rbf_trainable=model_config.rbf_trainable,
            activation_fn=activation_fn_map[model_config.activation_fn_name],
            k_tuple_dim=model_config.k_tuple_dim,
            block_num=model_config.block_num,
            pooling_level=model_config.get("pooling_level"),
            e_mode=model_config.get("e_mode"),
            qm9=True,
            model_name=self.model_name,
            use_mult_lin=model_config.get("use_mult_lin"),
            data_name=self.data_name,
            interaction_residual=model_config.get("interaction_residual"),
            )
    def init_row_col(self, n_nodes):
    #Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        return edges
        

    def coord2basis(self, data):
        coord_diff = data.pert_pos[data.edge_index[0]] - data.pert_pos[data.edge_index[1]]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_cross = torch.cross(data.pert_pos[data.edge_index[0]], data.pert_pos[data.edge_index[1]])

        #norm = torch.sqrt(radial) + 1
        #coord_diff = coord_diff / norm
        cross_norm = torch.sqrt(torch.sum((coord_cross)**2, 1).unsqueeze(1)) + 1
        coord_cross = coord_cross / cross_norm
        
        coord_vertical = torch.cross(coord_diff, coord_cross)

        return coord_diff, coord_cross, coord_vertical
        
    def run_wl(self, data):
        """data is torch geometric Data object and return edge-wise WL coloring"""
        row, col = init_row_col(coord_dist.size(1)) #arg is number of nodes
        coord_dist = data.pert_pos.unsqueeze(2) - data.pert_pos.unsqueeze(1) # (B, N, N, 3)
        coord_dist = torch.norm(coord_dist.clone(), dim=-1, keepdim=True) # (B, N, N, 1)
        rbf_coord_dist = self.dist_gaussian_fourier(coord_dist.reshape(-1, 1)).reshape(coord_dist.size(0), coord_dist.size(1), coord_dist.size(2), self.hidden_dim) # (B, N, N, 2*hidden_dim)
        rbf_coord_dist = self.radial_mlp(torch.clone(rbf_coord_dist)).reshape(coord_dist.size(0), coord_dist.size(1), coord_dist.size(2), self.hidden_dim) # (B, N, N, hidden_dim) 
        kemb = self.init_color(rbf_coord_dist)
        for i in range(self.color_steps):
            kemb += self.interaction_layers[i](
                        kemb=kemb.clone(),
                        )# (B, N ,N, hidden_nf)
        #return to reg shape and create three lists of index list to query result from wl
        batch   = torch.floor_divide(row, coord_dist.size(1))
        rowidx  = torch.remainder(row, coord_dist.size(1))
        colidx  = torch.remainder(col, coord_dist.size(1))
        
        #assert same sizes
        return kemb[batch, rowidx, colidx]
        

    
    def forward(self, data, node_attr, edge_attr):
        """
        Input:
            data: (torch_geometric.data.Data): batched graph
            node_attr: node feature tensor with shape (num_node, hidden)
            edge_attr: edge feature tensor with shape (num_edge, hidden)
        Output:
            node_attr
            graph feature
        """


        #sys.exit('indexing successful') 
        #update edge attr
        #edge_attr =  edge_attr.clone() + wl_edge_feat
        
        #node feature convolution    
        hiddens = []
        conv_input = node_attr # (num_node, hidden)
        


        for module_idx, convs in enumerate(self.transformers):
            for conv_idx, conv in enumerate(convs):
                hidden = conv(data.edge_index, conv_input, edge_attr)
                if conv_idx < len(convs) - 1 and self.activation is not None:
                    hidden = self.activation(torch.clone(hidden), inplace=True)
                assert hidden.shape == conv_input.shape                
                if self.short_cut and hidden.shape == conv_input.shape:
                    hidden = hidden.clone() + conv_input.clone()

                hiddens.append(hidden)
                conv_input = hidden

            if self.concat_hidden:
                node_feature = torch.cat(hiddens, dim=-1)
            else:
                node_feature = hiddens[-1]

            h_row, h_col = node_feature[data.edge_index[0]], node_feature[data.edge_index[1]] # (num_edge, hidden)
            edge_feature = torch.cat([h_row*h_col, edge_attr], dim=-1) # (num_edge, 2 * hidden)
            ## generate gradient
            
            dynamic_coff = self.dynamic_mlp_modules[module_idx](edge_feature)
            coord_diff, coord_cross, coord_vertical = self.coord2basis(data)
            #basis_mix = dynamic_coff[:, :1] * coord_diff + dynamic_coff[:, 1:2] * coord_cross + dynamic_coff[:, 2:3] * coord_vertical
            scalar_mult_coord_diffs = coord_diff * dynamic_coff
            
            if module_idx == 0:
                gradient = self.equi_modules[module_idx](node_feature, data.edge_index, scalar_mult_coord_diffs)
                
            else:
                gradient = gradient.clone() + self.equi_modules[module_idx](node_feature, data.edge_index, scalar_mult_coord_diffs)
            
        return {
            "node_feature": node_feature,
            "gradient": gradient
        }

