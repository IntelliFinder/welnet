import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from confgf import utils, layers

#import DisGNN code
from ..layers.DisGNN.layers.Mol2Graph import Mol2Graph
from ..layers.DisGNN.utils.loss_fns import loss_fn_map
from ..layers.DisGNN.utils.activation_fns import activation_fn_map
from ..layers.DisGNN.utils.EMA import ExponentialMovingAverage #TODO: insert EMA in trainging


from ..layers.DisGNN import kDisGNN


from ..layers.DisGNN.scripts.script_utils import get_cfgs


class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class EquiDistanceScoreMatch(torch.nn.Module):

    def __init__(self, config):
        super(EquiDistanceScoreMatch, self).__init__()
        model_name = "2FDis"
        self.model_name = model_name
        dataset_name = "qm9"
        self.dataset_name = dataset_name
        data_name = "ethanol" #default is ethanol, doesnt make a difference
        self.data_name = data_name
        config_path = "your-path/confgen/confgf/layers/DisGNN/config/{}_{}_3_blocks_256_72.yaml".format(model_name, dataset_name)
        specific_config_path = "hparams/specific/{}_{}_specific.yaml".format(model_name, dataset_name)
        config2 = get_cfgs(config_path, None, None, data_name)
        
                
        print("-"*20)
        print(config2)
        print("-"*20)
                
        model_config = config2.model_config

        self.config = config
        
        
        self.anneal_power = self.config.train.anneal_power
        self.hidden_dim = self.config.model.hidden_dim
        self.order = self.config.model.order
        self.noise_type = self.config.model.noise_type

        self.node_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.edge_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.dist_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1)
        self.input_mlp = layers.MultiLayerPerceptron(2 * self.hidden_dim, [self.hidden_dim], activation=self.config.model.mlp_act)
        self.dist_mlp = layers.MultiLayerPerceptron(1, [self.hidden_dim], activation=self.config.model.mlp_act)
        self.coff_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1)
        self.dist_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1)
        self.coff_mlp = nn.Linear(4 * self.hidden_dim, self. hidden_dim)
        self.radial_mlp = nn.Linear(2 *self.hidden_dim, self. hidden_dim)
        self.project = layers.MultiLayerPerceptron(2 * self.hidden_dim + 2, [self.hidden_dim, self.hidden_dim], activation=self.config.model.mlp_act)
        self.project_radial = layers.MultiLayerPerceptron(self.hidden_dim, [self.hidden_dim, self.hidden_dim], activation=self.config.model.mlp_act)
        #initical coloring
        #self.init_color = layers.TwoFDisInit(ef_dim=self.hidden_dim, k_tuple_dim=self.hidden_dim)  
        #color update steps
        #self.interaction_layers = nn.ModuleList()
        #color_steps = 3 
        #for _ in range(color_steps):
        #    self.interaction_layers.append(
        #            layers.TwoFDisLayer(
        #                hidden_dim=self.hidden_dim,
        #                )
        #            )
        self.wl = kDisGNN(
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
        #print(self.wl)
                    
        self.model = layers.GradientGEGNNWL(hidden_dim=self.hidden_dim, hidden_coff_dim=128, \
                                 num_convs=self.config.model.num_convs, \
                                 activation=self.config.model.gnn_act, \
                                 readout="sum", short_cut=self.config.model.short_cut, \
                                 concat_hidden=self.config.model.concat_hidden)
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_noise_level)), dtype=torch.float32)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False) # (num_noise_level)
        """
        Techniques from "Improved Techniques for Training Score-Based Generative Models"
        1. Choose sigma1 to be as large as the maximum Euclidean distance between all pairs of training data points.
        2. Choose sigmas as a geometric progression with common ratio gamma, where a specific equation of CDF is satisfied.
        3. Parameterize the Noise Conditional Score Networks with f_theta_sigma(x) =  f_theta(x) / sigma
        """

    
    @torch.no_grad()
    # extend the edge on the fly, second order: angle, third order: dihedral
    def extend_graph(self, data: Data, order=3):

        def binarize(x):
            return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

        def get_higher_order_adj_matrix(adj, order):
            """
            Args:
                adj:        (N, N)
                type_mat:   (N, N)
            """
            adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                        binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

            for i in range(2, order+1):
                adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
            order_mat = torch.zeros_like(adj)

            for i in range(1, order+1):
                order_mat += (adj_mats[i] - adj_mats[i-1]) * i

            return order_mat

        num_types = len(utils.BOND_TYPES)

        N = data.num_nodes
        adj = to_dense_adj(data.edge_index).squeeze(0)
        adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

        type_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).squeeze(0)   # (N, N)
        type_highorder = torch.where(adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order))
        assert (type_mat * type_highorder == 0).all()
        type_new = type_mat + type_highorder

        new_edge_index, new_edge_type = dense_to_sparse(type_new)
        _, edge_order = dense_to_sparse(adj_order)

        data.bond_edge_index = data.edge_index  # Save original edges
        data.edge_index, data.edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N) # modify data
        edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N) # modify data
        data.is_bond = (data.edge_type < num_types)
        assert (data.edge_index == edge_index_1).all()

        return data
    # @torch.no_grad()
    def get_distance(self, data: Data):
        pos = data.pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1) # (num_edge, 1)
        data.edge_length = d
        return data   

    # @torch.no_grad()
    def get_perturb_distance(self, data: Data, p_pos):
        pos = p_pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1) # (num_edge, 1)
        return d  

    def get_pred_distance(self, data: Data, p_pos):
        pos = p_pos
        row, col = data.edge_index
        d = torch.sqrt(torch.sum((pos[row] - pos[col])**2, dim=-1) + 0.0001)
        # d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1) # (num_edge, 1)
        return d    

    def coord2basis(self, data):
        coord_diff = data.pert_pos[data.edge_index[0]] - data.pert_pos[data.edge_index[1]]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_cross = torch.cross(data.pert_pos[data.edge_index[0]], data.pert_pos[data.edge_index[1]])

        norm = torch.sqrt(radial) + 1
        coord_diff = coord_diff / norm
        cross_norm = torch.sqrt(torch.sum((coord_cross)**2, 1).unsqueeze(1)) + 1
        coord_cross = coord_cross / cross_norm
        
        coord_vertical = torch.cross(coord_diff, coord_cross)

        return coord_diff, coord_cross, coord_vertical, radial

    @torch.no_grad()
    def get_angle(self, data: Data, p_pos):
        pos = p_pos
        row, col = data.edge_index
        pos_normal = pos.clone().detach()
        pos_normal_norm = pos_normal.norm(dim=-1).unsqueeze(-1)
        pos_normal = pos_normal / (pos_normal_norm + 1e-5)
        cos_theta = torch.sum(pos_normal[row] * pos_normal[col], dim=-1, keepdim=True)
        sin_theta = torch.sqrt(1 - cos_theta ** 2)
        node_angles = torch.cat([cos_theta, sin_theta], dim=-1)
        return node_angles

    @torch.no_grad()
    def get_score(self, data: Data, d, sigma):
        """
        Input:
            data: torch geometric batched data object
            d: edge distance, shape (num_edge, 1)
            sigma: noise level, tensor (,)
        Output:
            log-likelihood gradient of distance, tensor with shape (num_edge, 1)         
        """
        
        # generate common features
        node_attr = self.node_emb(data.atom_type) # (num_node, hidden)
        edge_attr = self.edge_emb(data.edge_type) # (num_edge, hidden)
        d_emb = self.dist_gaussian_fourier(d)
        d_emb = self.input_mlp(d_emb) # (num_edge, hidden)
        edge_attr = d_emb * edge_attr # (num_edge, hidden)
        
        #TODO : use this d in the distance matrix?
        
        #construct WL features
        wl_output = self.wl(data)

        B    = wl_output.size(1)
        row  = data.edge_index[0]
        col  = data.edge_index[1]
        batch   = torch.floor_divide(row, B)
        rowidx  = torch.remainder(row, B)
        colidx  = torch.remainder(col, B)
            
        wl_edge_feat = wl_output[batch, rowidx, colidx]
        
        edge_attr = edge_attr + wl_edge_feat
        
        # construct geometric features
        row, col = data.edge_index[0], data.edge_index[1] # check if roe and col is right?
        coord_diff, coord_cross, coord_vertical, radial = self.coord2basis(data) # [E, 3]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1) # [E, 3]
        r_i, r_j = data.pert_pos[row], data.pert_pos[col] # [E, 3]
        # [E, 3, 3] x [E, 3, 1]
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1) # [E, 3]
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1) # [E, 3]
        coff_mul = coff_i * coff_j # [E, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True)
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True)
        pesudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + 1e-5) / (coff_j_norm + 1e-5)
        pesudo_sin = torch.sqrt(1 - pesudo_cos**2)
        psudo_angle = torch.cat([pesudo_sin, pesudo_cos], dim=-1)
        embed_i = self.get_embedding(coff_i) # [E, C]
        embed_j = self.get_embedding(coff_j) # [E, C]
        edge_embed = torch.cat([psudo_angle, embed_i, embed_j], dim=-1)
        edge_embed = self.project(edge_embed)
        edge_embed = self.get_dist_embedding(radial)
        edge_embed = self.project_radial(torch.clone(edge_embed))
        #edge_attr = edge_attr + edge_embed

        output = self.model(data, node_attr, edge_attr)
        scores = output["gradient"] * (1. / sigma) # f_theta_sigma(x) =  f_theta(x) / sigma, (num_edge, 1)
        return scores
    def get_embedding(self, coff_index):
        coff_embeds = []
        for i in [0, 2]: # if i=1, then x=0
            coff_embeds.append(self.coff_gaussian_fourier(coff_index[:, i:i+1])) #[E, 2C]
        coff_embeds = torch.cat(coff_embeds, dim=-1) # [E, 6C]
        coff_embeds = self.coff_mlp(coff_embeds)
        
        return coff_embeds
        
    def get_dist_embedding(self, radial):
        radial = self.coff_gaussian_fourier(torch.clone(radial)) #[E, 2C]
        radial = self.radial_mlp(torch.clone(radial))
        return radial

    def forward(self, data):
        """
        Input:
            data: torch geometric batched data object
        Output:
            loss
        """
        # a workaround to get the current device, we assume all tensors in a model are on the same device.
        self.device = self.sigmas.device
        data = self.extend_graph(data, self.order)
        ## enable input gradient
        input_x = data.pos
        input_x.requires_grad = True #why?

        data = self.get_distance(data)

        assert data.edge_index.size(1) == data.edge_length.size(0) #TODO problem?
        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]        

        # sample noise level
        noise_level = torch.randint(0, self.sigmas.size(0), (data.batch_size,), device=self.device) # (num_graph)
        used_sigmas = self.sigmas[noise_level] # (num_graph)
        used_sigmas = used_sigmas[node2graph].unsqueeze(-1) # (num_nodes, 1)

        if self.noise_type == 'rand':
            coord_noise = torch.randn_like(data.pos) 
        else:
            raise NotImplementedError('noise type must in [distance_symm, distance_rand]')
  
        assert coord_noise.shape == data.pos.shape
        perturbed_pos = data.pos + coord_noise * used_sigmas 
        data.pert_pos = perturbed_pos 
        perturbed_d = self.get_perturb_distance(data, perturbed_pos)
        target = -1 / (used_sigmas ** 2) * (perturbed_pos - data.pos)

        # generate common features
        node_attr = self.node_emb(data.atom_type) # (num_node, hidden)
        edge_attr = self.edge_emb(data.edge_type) # (num_edge, hidden)
        d_emb = self.dist_gaussian_fourier(perturbed_d)
        d_emb = self.input_mlp(d_emb) # (num_edge, hidden)
        edge_attr = d_emb * edge_attr # (num_edge, hidden)
        
        #generate wl feat
                #get wl edge features
        wl_output = self.wl(data)

        B    = wl_output.size(1)
        row  = data.edge_index[0]
        col  = data.edge_index[1]
        batch   = torch.floor_divide(row, B)
        rowidx  = torch.remainder(row, B)
        colidx  = torch.remainder(col, B)
            
        wl_edge_feat = wl_output[batch, rowidx, colidx]
        
        edge_attr = edge_attr + wl_edge_feat
        
        # construct geometric features
        #row, col = data.edge_index[0], data.edge_index[1] # check if roe and col is right?
        #coord_diff, coord_cross, coord_vertical, radial = self.coord2basis(data) # [E, 3]
        #edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1) # [E, 3]
        #r_i, r_j = data.pert_pos[row], data.pert_pos[col] # [E, 3]
        #coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1) # [E, 3]
        #coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1) # [E, 3]
        #coff_mul = coff_i * coff_j # [E, 3]
        #coff_i_norm = coff_i.norm(dim=-1, keepdim=True)
        #coff_j_norm = coff_j.norm(dim=-1, keepdim=True)
        #pesudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + 1e-5) / (coff_j_norm + 1e-5)
        #pesudo_sin = torch.sqrt(1 - pesudo_cos**2)
        #psudo_angle = torch.cat([pesudo_sin, pesudo_cos], dim=-1)
        #embed_i = self.get_embedding(coff_i) # [E, C]
        #embed_j = self.get_embedding(coff_j) # [E, C]
        #edge_embed = torch.cat([psudo_angle, embed_i, embed_j], dim=-1)
        #edge_embed = self.project(edge_embed)
        #edge_embed = self.get_dist_embedding(radial)
        #edge_embed = self.project_radial(torch.clone(edge_embed))
        #edge_attr = edge_attr + edge_embed

        # estimate scores
        output = self.model(data, node_attr, edge_attr)
        scores = output["gradient"] * (1. / used_sigmas)
        loss_pos =  0.5 * torch.sum((scores - target) ** 2, -1) * (used_sigmas.squeeze(-1) ** self.anneal_power) # (num_edge)
        loss_pos = scatter_mean(loss_pos, node2graph) # (num_graph)
        
        loss_dict = {
            'position': loss_pos.mean(),
            'distance': torch.Tensor([0]).to(loss_pos.device),
        }
        return loss_dict
