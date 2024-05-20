import torch
from torch import nn
from models.gcl import GCL, E_GCL, E_GCL_vel, GCL_rf_vel
import os, sys

from models.wl import TwoFDisInit, TwoFDisLayer
from models.basis_layers import rbf_class_mapping



class GNN(nn.Module):
    def __init__(self, input_dim, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, attention=0, recurrent=False):
        super(GNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        #self.add_module("gcl_0", GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=1, act_fn=act_fn, attention=attention, recurrent=recurrent))
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=1, act_fn=act_fn, attention=attention, recurrent=recurrent))

        self.decoder = nn.Sequential(nn.Linear(hidden_nf, hidden_nf),
                              act_fn,
                              nn.Linear(hidden_nf, 3))
        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_nf))
        self.to(self.device)


    def forward(self, nodes, edges, edge_attr=None):
        h = self.embedding(nodes)
        #h, _ = self._modules["gcl_0"](h, edges, edge_attr=edge_attr)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
        #return h
        return self.decoder(h)


def get_velocity_attr(loc, vel, rows, cols):
    #return  torch.cat([vel[rows], vel[cols]], dim=1)

    diff = loc[cols] - loc[rows]
    norm = torch.norm(diff, p=2, dim=1).unsqueeze(1)
    u = diff/norm
    va, vb = vel[rows] * u, vel[cols] * u
    va, vb = torch.sum(va, dim=1).unsqueeze(1), torch.sum(vb, dim=1).unsqueeze(1)
    return va

class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.LeakyReLU(0.2), n_layers=4, coords_weight=1.0):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        #self.reg = reg
        ### Encoder
        #self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=True, coords_weight=coords_weight))
        self.to(self.device)


    def forward(self, h, x, edges, edge_attr, vel=None):
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            #if vel is not None:
                #vel_attr = get_velocity_attr(x, vel, edges[0], edges[1])
                #edge_attr = torch.cat([edge_attr0, vel_attr], dim=1).detach()
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        return x


class EGNN_vel(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_edge_nf, hidden_node_nf, hidden_coord_nf,
                 device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, recurrent=False,
                 norm_diff=False, tanh=False, num_vectors=1,
                 update_vel=False, color_steps=2, ef_dim=3, mixed=False, shared_wl=False, wl_dim=32):
        super(EGNN_vel, self).__init__()
        self.hidden_edge_nf  = hidden_edge_nf
        self.hidden_node_nf  = hidden_node_nf
        self.hidden_coord_nf = hidden_coord_nf
        hidden_nf            = self.hidden_edge_nf
        self.device = device
        self.n_layers = n_layers
        self.update_vel = update_vel
        self.init_color = TwoFDisInit(ef_dim=ef_dim, k_tuple_dim=wl_dim, activation_fn=act_fn)
        self.init_color_mixed_first = TwoFDisInit(ef_dim=ef_dim , k_tuple_dim=wl_dim, activation_fn=act_fn)
        self.init_color_mixed = TwoFDisInit(ef_dim=ef_dim*(num_vectors)**2 , k_tuple_dim=wl_dim, activation_fn=act_fn)
        # interaction layers
        self.interaction_layers = nn.ModuleList()
        for _ in range(color_steps):
            self.interaction_layers.append(
                    TwoFDisLayer(
                        hidden_dim=wl_dim,
                        activation_fn=act_fn,
                        )
                    )
        #self.reg = reg
        ### Encoder
        #self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_node_nf)
        self.add_module("gcl_%d" % 0, E_GCL_vel(self.hidden_node_nf, self.hidden_node_nf, self.hidden_edge_nf, self.hidden_node_nf, self.hidden_coord_nf, edges_in_d=in_edge_nf, act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent, norm_diff=norm_diff, tanh=tanh, num_vectors_out=num_vectors, color_steps=color_steps, ef_dim=ef_dim, mixed=mixed, shared_wl=shared_wl, init_color=self.init_color, init_color_mixed=self.init_color_mixed, interaction_layers=self.interaction_layers, init_color_mixed_first=self.init_color_mixed_first, wl_dim=wl_dim))#
        for i in range(1, n_layers - 1):
            self.add_module("gcl_%d" % i, E_GCL_vel(self.hidden_node_nf, self.hidden_node_nf, self.hidden_edge_nf, self.hidden_node_nf, self.hidden_coord_nf, edges_in_d=in_edge_nf, act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent, norm_diff=norm_diff, tanh=tanh, num_vectors_in=num_vectors, num_vectors_out=num_vectors, color_steps=color_steps, ef_dim=ef_dim, mixed=mixed, shared_wl=shared_wl, init_color=self.init_color, init_color_mixed=self.init_color_mixed, interaction_layers=self.interaction_layers, wl_dim=wl_dim))
        self.add_module("gcl_%d" % (n_layers - 1), E_GCL_vel(self.hidden_node_nf, self.hidden_node_nf,self.hidden_edge_nf, self.hidden_node_nf, self.hidden_coord_nf, edges_in_d=in_edge_nf, act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent, norm_diff=norm_diff, tanh=tanh, num_vectors_in=num_vectors, last_layer=True, color_steps=color_steps, ef_dim=ef_dim, mixed=mixed, shared_wl=shared_wl, init_color=self.init_color, init_color_mixed=self.init_color_mixed, interaction_layers=self.interaction_layers, wl_dim=wl_dim))
        self.to(self.device)


    def forward(self, h, x, edges, vel, edge_attr, use_traj=False):
        h = self.embedding(h)
        wl_edge_feat_prop=None
        n_nodes=5
        x = x.reshape(-1, n_nodes, 3)
        centroid = torch.mean(x, dim=1, keepdim=True)
        x_center = (x - centroid).reshape(-1, 3)
        if use_traj:
            x_traj = [x_center.clone()]
        for i in range(0, self.n_layers):
            h, x_center, _, new_vel, wl_edge_feat_prop = self._modules["gcl_%d" % i](h, edges, x_center, vel, wl_edge_feat_prop=wl_edge_feat_prop, edge_attr=edge_attr)
            if self.update_vel:
                vel = new_vel
            if use_traj:
                x_traj.append(x_center.clone())
        if use_traj:
            x = x_center.squeeze(2).reshape(-1, n_nodes, 3) + centroid
            x = x.reshape(-1, 3)
            return x, x_traj
        else:
            x = x_center.squeeze(2).reshape(-1, n_nodes, 3) + centroid
            x = x.reshape(-1, 3)
            return x

class RF_vel(nn.Module):
    def __init__(self, hidden_nf, edge_attr_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4):
        super(RF_vel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        #self.reg = reg
        ### Encoder
        #self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL_rf_vel(nf=hidden_nf, edge_attr_nf=edge_attr_nf, act_fn=act_fn))
        self.to(self.device)


    def forward(self, vel_norm, x, edges, vel, edge_attr):
        for i in range(0, self.n_layers):
            x, _ = self._modules["gcl_%d" % i](x, vel_norm, vel, edges, edge_attr)
        return x

class Baseline(nn.Module):
    def __init__(self, device='cpu'):
        super(Baseline, self).__init__()
        self.dummy = nn.Linear(1, 1)
        self.device = device
        self.to(self.device)

    def forward(self, loc):
        return loc

class Linear(nn.Module):
    def __init__(self, input_nf, output_nf, device='cpu'):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_nf, output_nf)
        self.device = device
        self.to(self.device)

    def forward(self, input):
        return self.linear(input)

class Linear_dynamics(nn.Module):
    def __init__(self, device='cpu'):
        super(Linear_dynamics, self).__init__()
        self.time = nn.Parameter(torch.ones(1)*0.7)
        self.device = device
        self.to(self.device)

    def forward(self, x, v):
        return x + v*self.time