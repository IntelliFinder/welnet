import os, sys

from torch import nn
import torch

from .wl import TwoFDisInit, TwoFDisLayer
from models.basis_layers import rbf_class_mapping


class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


class GCL_basic(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self):
        super(GCL_basic, self).__init__()


    def edge_model(self, source, target, edge_attr):
        pass

    def node_model(self, h, edge_index, edge_attr):
        pass

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        edge_feat = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_feat)
        return x, edge_feat



class GCL(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_nf=0, act_fn=nn.ReLU(), bias=True, attention=False, t_eq=False, recurrent=True):
        super(GCL, self).__init__()
        self.attention = attention
        self.t_eq=t_eq
        self.recurrent = recurrent
        input_edge_nf = input_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_nf + edges_in_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf, bias=bias),
            act_fn)
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf, bias=bias),
                act_fn,
                nn.Linear(hidden_nf, 1, bias=bias),
                nn.Sigmoid())


        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, output_nf, bias=bias))

        #if recurrent:
            #self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, edge_attr):
        edge_in = torch.cat([source, target], dim=1)
        if edge_attr is not None:
            edge_in = torch.cat([edge_in, edge_attr], dim=1)
        out = self.edge_mlp(edge_in)
        if self.attention:
            att = self.att_mlp(torch.abs(source - target))
            out = out * att
        return out

    def node_model(self, h, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        out = torch.cat([h, agg], dim=1)
        out = self.node_mlp(out)
        if self.recurrent:
            out = out + h
            #out = self.gru(out, h)
        return out


class GCL_rf(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, nf=64, edge_attr_nf=0, reg=0, act_fn=nn.LeakyReLU(0.2), clamp=False):
        super(GCL_rf, self).__init__()

        self.clamp = clamp
        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi = nn.Sequential(nn.Linear(edge_attr_nf + 1, nf),
                                 act_fn,
                                 layer)
        self.reg = reg

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        if self.clamp:
            m_ij = torch.clamp(m_ij, min=-100, max=100)
        return m_ij

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_attr, row, num_segments=x.size(0))
        x_out = x + agg - x*self.reg
        return x_out


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_edge_nf, hidden_node_nf, hidden_coord_nf,edges_in_d=0,
                nodes_att_dim=0, act_fn=nn.ReLU(),recurrent=True, coords_weight=1.0,
                attention=False, clamp=False, norm_diff=False, tanh=False,
                num_vectors_in=1, num_vectors_out=1, last_layer=False, wl_dim=32):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.num_vectors_in = num_vectors_in
        self.num_vectors_out = num_vectors_out
        self.last_layer = last_layer
        edge_coords_nf = 1
        wl_diff = (wl_dim != hidden_edge_nf)

        print((1-int(wl_diff))*hidden_edge_nf + int(wl_diff)*wl_dim)
        print(int(wl_diff)*wl_dim)
        #import sys
        #sys.exit("printed wl dim true/false")
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + num_vectors_in + edges_in_d + (1-int(wl_diff))*hidden_edge_nf + int(wl_diff)*wl_dim, hidden_edge_nf),
            act_fn,
            nn.Linear(hidden_edge_nf, hidden_edge_nf),
            act_fn)
        print(self.edge_mlp)
        
        self.edge_mlp_prop = nn.Sequential(
            nn.Linear(input_edge + num_vectors_in + edges_in_d + 2*(1-int(wl_diff))*hidden_edge_nf + 2*int(wl_diff)*wl_dim, hidden_edge_nf),
            act_fn,
            nn.Linear(hidden_edge_nf, hidden_edge_nf),
            act_fn)
        print(self.edge_mlp)
            
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_edge_nf + input_nf + nodes_att_dim, hidden_node_nf),
            act_fn,
            nn.Linear(hidden_node_nf, output_nf))

        layer = nn.Linear(hidden_coord_nf, num_vectors_in * num_vectors_out, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_edge_nf, hidden_coord_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_edge_nf, 1),
                nn.Sigmoid())

        #if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, radial, wl_colors, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial, wl_colors], dim=1)
        else:
            out = torch.cat([source, target, radial, wl_colors, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out
        
    def edge_model_prop(self, source, target, radial, wl_colors, prev_colors, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial, wl_colors, prev_colors], dim=1)
        else:
            out = torch.cat([source, target, radial, wl_colors, prev_colors, edge_attr], dim=1)
        out = self.edge_mlp_prop(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, radial, edge_feat):
        row, col = edge_index
        coord_matrix = self.coord_mlp(edge_feat).view(-1, self.num_vectors_in, self.num_vectors_out)
        if coord_diff.dim() == 2:
            coord_diff = coord_diff.unsqueeze(2)
            coord = coord.unsqueeze(2).repeat(1, 1, self.num_vectors_out)
        # coord_diff = coord_diff / radial.unsqueeze(1)
        trans = torch.einsum('bij,bci->bcj', coord_matrix, coord_diff)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        if self.last_layer:
            coord = coord.mean(dim=2, keepdim=True) + agg * self.coords_weight
        else:
            coord += agg * self.coords_weight
        return coord


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        if radial.dim() == 3:
            radial = radial.squeeze(1)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, radial, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr


class E_GCL_vel(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """


    def __init__(self, input_nf, output_nf, hidden_edge_nf, hidden_node_nf, hidden_coord_nf,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0,
                 attention=False, norm_diff=False, tanh=False, num_vectors_in=1, num_vectors_out=1, last_layer=False, color_steps=2, ef_dim=3, mixed=False,  shared_wl=False, init_color=None, init_color_mixed=None, init_color_mixed_first=None, interaction_layers=None, wl_dim=64, prop=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_edge_nf, hidden_node_nf, hidden_coord_nf,
                       edges_in_d=edges_in_d, nodes_att_dim=nodes_att_dim, act_fn=act_fn, recurrent=recurrent,
                       coords_weight=coords_weight, attention=attention, norm_diff=norm_diff, tanh=tanh,
                       num_vectors_in=num_vectors_in, num_vectors_out=num_vectors_out, last_layer=last_layer, wl_dim=wl_dim)
        self.shared_wl      = shared_wl
        self.num_vectors_in = num_vectors_in
        self.norm_diff      = norm_diff
        self.mixed          = mixed
        self.prop           = prop
        self.coord_mlp_vel  = nn.Sequential(
            nn.Linear(input_nf, hidden_coord_nf),
            act_fn,
            nn.Linear(hidden_coord_nf, num_vectors_in * num_vectors_out))
        self.color_mlp = nn.Sequential(
            nn.Linear(hidden_edge_nf*num_vectors_in, hidden_edge_nf),
            act_fn,
            nn.Linear(hidden_edge_nf, hidden_edge_nf),
            act_fn)
            
        rbound_upper                = 10
        hidden_nf                   = hidden_edge_nf
        self.color_steps            = color_steps
        self.ef_dim=ef_dim
        if shared_wl:
            self.init_color         = init_color
            self.init_color_mixed   = init_color_mixed
            if (num_vectors_in == 1):
              self.init_color_mixed = init_color_mixed_first
            self.interaction_layers = interaction_layers
        else:
            self.init_color = TwoFDisInit(ef_dim=ef_dim, k_tuple_dim=wl_dim, activation_fn=act_fn)
            self.init_color_mixed = TwoFDisInit(ef_dim=ef_dim*(num_vectors_in)**2 , k_tuple_dim=wl_dim, activation_fn=act_fn)
            # interaction layers
            self.interaction_layers = nn.ModuleList()
            for _ in range(color_steps):
                self.interaction_layers.append(
                        TwoFDisLayer(
                            hidden_dim=wl_dim,
                            activation_fn=act_fn,
                            )
                        )

        self.rbf_fn = rbf_class_mapping["nexpnorm"](
                    num_rbf=ef_dim, 
                    rbound_upper=rbound_upper, 
                    rbf_trainable=False,
                )
                
        
    def wl(self, edge_index, clouds: list=[]):
        row, col = edge_index
        #return to reg shape and create three lists of index list to query result from wl
        batch   = torch.floor_divide(row, 5)
        rowidx  = torch.remainder(row, 5)
        colidx  = torch.remainder(col, 5)
        colors  = []
        for j in range(len(clouds)):
          # apply WL to batch
          B = clouds[j].size(0) // 5
          coord = clouds[j].reshape(B, 5, 3)
          coord_dist = coord.unsqueeze(2) - coord.unsqueeze(1) # (B, N, N, 3)
          coord_dist = torch.norm(coord_dist.clone(), dim=-1, keepdim=True) # (B, N, N, 1)
          rbf_coord_dist = self.rbf_fn(coord_dist.reshape(-1, 1)).reshape(B, 5, 5,self.ef_dim) # (B, N, N, ef_dim)
          
  
          #print(self.rbf_fn(coord_dist.reshape(-1, 1)).size())
          #run wl
          kemb = self.init_color(rbf_coord_dist) #do init color one? with only 4 features?
          for i in range(self.color_steps):
              kemb += self.interaction_layers[i](
                          kemb=kemb.clone(),
                          )   # (B, N ,N, hidden_nf)
          colors.append(kemb[batch, rowidx, colidx])
          
          #assert same sizes
        colors = torch.cat(colors, dim=-1)
        colors = self.color_mlp(colors) 
        
        #apply function
        return  colors  
                  
    def wl_mixed(self, edge_index, clouds: list=[]):
        row, col = edge_index
        #return to reg shape and create three lists of index list to query result from wl
        batch   = torch.floor_divide(row, 5)
        rowidx  = torch.remainder(row, 5)
        colidx  = torch.remainder(col, 5)
        colors  = []
        B = clouds[0].size(0) // 5
        for i in range(len(clouds)):
            for j in range(len(clouds)):
                # apply WL to batch
                coord1 = clouds[i].reshape(B, 5, 3)
                coord2 = clouds[j].reshape(B, 5, 3)
                coord_dist = coord1.unsqueeze(2) - coord2.unsqueeze(1) # (B, N, N, 3)
                coord_dist = torch.norm(coord_dist.clone(), dim=-1, keepdim=True) # (B, N, N, 1)
                rbf_coord_dist = self.rbf_fn(coord_dist.reshape(-1, 1)).reshape(B, 5, 5,self.ef_dim) # (B, N, N, ef_dim)
                colors.append(rbf_coord_dist)
  
        colors = torch.cat(colors, dim=-1)
        kemb = self.init_color_mixed(colors) 
        for i in range(self.color_steps):
            kemb += self.interaction_layers[i](
                        kemb=kemb.clone(),
                        )   # (B, N ,N, hidden_nf)
        #apply function
        return  kemb[batch, rowidx, colidx]  
           

    def forward(self, h, edge_index, coord, vel, wl_edge_feat_prop=None, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        if coord_diff.dim() > 2:
          clouds = [ coord_diff[:,:, i] for i in range(self.num_vectors_in)]
        else:
          clouds = [coord_diff]
        if self.mixed:
            wl_edge_feat = self.wl_mixed(edge_index, clouds) 
        else:
            wl_edge_feat = self.wl(edge_index, clouds)
        edge_feat = self.edge_model(h[row], h[col], radial, wl_edge_feat, edge_attr)
        if self.prop and wl_edge_feat_prop!=None:
          edge_feat = self.edge_model_prop(h[row], h[col], radial, wl_edge_feat, wl_edge_feat_prop, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, radial, edge_feat)

        coord_vel_matrix = self.coord_mlp_vel(h).view(-1, self.num_vectors_in, self.num_vectors_out)
        if vel.dim() == 2:
            vel = vel.unsqueeze(2)
        coord += torch.einsum('bij,bci->bcj', coord_vel_matrix, vel)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr, vel, wl_edge_feat
 
class GCL_rf_vel(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """
    def __init__(self,  nf=64, edge_attr_nf=0, act_fn=nn.LeakyReLU(0.2), coords_weight=1.0):
        super(GCL_rf_vel, self).__init__()
        self.coords_weight = coords_weight
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(1, nf),
            act_fn,
            nn.Linear(nf, 1))

        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        #layer.weight.uniform_(-0.1, 0.1)
        self.phi = nn.Sequential(nn.Linear(1 + edge_attr_nf, nf),
                                 act_fn,
                                 layer,
                                 nn.Tanh()) #we had to add the tanh to keep this method stable

    def forward(self, x, vel_norm, vel, edge_index, edge_attr=None):
        row, col = edge_index
        edge_m = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_m)
        x += vel * self.coord_mlp_vel(vel_norm)
        return x, edge_attr

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        return m_ij

    def node_model(self, x, edge_index, edge_m):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_m, row, num_segments=x.size(0))
        x_out = x + agg * self.coords_weight
        return x_out


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_sum_vec(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1), data.size(2))
    segment_ids = segment_ids.unsqueeze(-1).unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, data.size(1), data.size(2))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result 

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1), data.size(2))
    segment_ids = segment_ids.unsqueeze(-1).unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, data.size(1), data.size(2))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)