import torch.nn as nn
from .layers.Mol2Graph import Mol2Graph
from .utils.loss_fns import loss_fn_map
from .utils.activation_fns import activation_fn_map
from .utils.EMA import ExponentialMovingAverage

import torch
from .twoFDis.twoFDis_init import TwoFDisInit
from .twoFDis.twoFDis_interaction import TwoFDisLayer
#from layers.twoFDis.twoFDis_output import TwoOrderOutputBlock, TwoOrderDipOutputBlock, TwoOrderElcOutputBlock

from .layers.basic_layers import Residual, Dense

import sys

init_layer_dict = {
    "2FDis": TwoFDisInit,
}

interaction_layer_dict = {
    "2FDis": TwoFDisLayer,
}

#output_layer_dict = {
#    "2FDis": TwoOrderOutputBlock,
#    "2FDisDip": TwoOrderDipOutputBlock,
#    "2FDisElc": TwoOrderElcOutputBlock,
#    "3EDis": ThreeOrderOutputBlock,
#    "2EDis": TwoOrderOutputBlock,
#    "3FDis": ThreeOrderOutputBlock
#}

class kDisGNN(nn.Module):
    def __init__(self, 
        z_hidden_dim,
        ef_dim,
        rbf,
        max_z,
        rbound_upper,
        rbf_trainable,
        activation_fn,
        k_tuple_dim,
        block_num,
        pooling_level,
        e_mode,
        qm9,
        model_name,
        data_name,
        use_mult_lin,
        interaction_residual,
    ):
        super().__init__()
        
        
        # Transform Molecule to Graph
        self.M2G = Mol2Graph(
            z_hidden_dim=z_hidden_dim,
            ef_dim=ef_dim,
            rbf=rbf,
            max_z=max_z,
            rbound_upper=rbound_upper,
            rbf_trainable=rbf_trainable
        )
        

        
        # initialize tuples
        init_layer = init_layer_dict[model_name]
        self.init_layer = init_layer(
            z_hidden_dim=z_hidden_dim,
            ef_dim=ef_dim,
            k_tuple_dim=k_tuple_dim,
            activation_fn=activation_fn,
        )
        
        # interaction layers
        self.interaction_layers = nn.ModuleList()
        if interaction_residual:
            self.interaction_residual_layers = nn.ModuleList()
        interaction_layer = interaction_layer_dict[model_name]
        for _ in range(block_num):
            self.interaction_layers.append(
                    interaction_layer(
                        hidden_dim=k_tuple_dim,
                        activation_fn=activation_fn,
                        e_mode=e_mode,
                        ef_dim=ef_dim,
                        use_mult_lin=use_mult_lin,
                        )
                    )
            if interaction_residual:
                self.interaction_residual_layers.append(
                    Residual(
                        mlp_num=2,
                        hidden_dim=k_tuple_dim,
                        activation_fn=activation_fn
                    )
                )
        #dim reduction to edge feature size
        self.dim_reduce = Dense(in_features=k_tuple_dim, out_features=288, bias=True)
        # output layers
        #self.output_layers = output_layer(
        #    hidden_dim=k_tuple_dim,
        #    activation_fn=activation_fn,
        #    pooling_level=pooling_level
        #    )
        
        
        self.predict_force = not qm9
        self.interaction_residual = interaction_residual


    def forward(self, batch_data):
        #print(batch_data.to_dict())
        #torch.save(batch_data.to_dict(), 'dict.pt')
        #sys.exit()
        if self.predict_force: #False for qm9
            batch_data.pos.requires_grad_(True)
        
        B = batch_data.num_graphs
        
        z, pos = batch_data.atom_type.reshape(B, -1), batch_data.pert_pos.reshape(B, -1, 3)
        
        # Molecule to Graphs
        emb1, ef = self.M2G(z, pos)  
        kemb = self.init_layer(emb1, ef)

        # interaction
        for i in range(len(self.interaction_layers)):
            kemb = self.interaction_layers[i](
                kemb=kemb.clone(),
                ef=ef
                ) + kemb.clone()
            if self.interaction_residual:
                kemb = self.interaction_residual_layers[i](kemb)

        #reduce to output size
        #B, N = kemb.shape[0], kemb.shape[1]
        #self.rbf_fn(el.reshape(-1, 1)).reshape(B, N, N, -1) # (B, N, N, ef_dim)
        kemb = self.dim_reduce(kemb.clone())
        # output
        #scores = self.output_layers(
        #    kemb=kemb,
        #    pos=pos,
        #    z=z
        #    )
        
        # normalize
        #pred_energy = scores * self.global_y_std + self.global_y_mean
        
        #if self.predict_force:
        #    pred_force = -torch.autograd.grad(
        #        [torch.sum(pred_energy)], 
        #        [batch_data.pos],
        #        retain_graph=True,
        #       create_graph=True
        #        )[0]
        #    batch_data.pos.requires_grad_(False)
        #    
        #    return pred_energy, pred_force
        
        return kemb