import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal 

class TwoFDisInit(nn.Module):
    def __init__(self,
                 ef_dim: int,   
                 k_tuple_dim: int,
                 activation_fn: nn.Module = nn.SiLU(),
                 **kwargs
                 ):
        super().__init__()


        self.ef_lin = nn.Linear(ef_dim, k_tuple_dim, bias=False)


        self.pattern_embedding = nn.Embedding(
            num_embeddings=3,
            embedding_dim=k_tuple_dim,
            padding_idx=0
        )
        
        self.mix_lin = nn.Sequential(
            nn.Linear(k_tuple_dim, k_tuple_dim),
            activation_fn,
            nn.Linear(k_tuple_dim, k_tuple_dim),
            activation_fn,
            nn.Linear(k_tuple_dim, k_tuple_dim)
        )


    def forward(self,
                ef: torch.Tensor
                ):
        
        ef0 = self.ef_lin(ef.clone())
        
        ef_mixed = ef0 # (B, N, N, k_tuple_dim)
        

        B = ef_mixed.shape[0]
        N = ef_mixed.shape[1]
        
        idx = torch.arange(N)
        tuple_pattern = torch.ones(size=(B, N, N), dtype=torch.int64, device=ef_mixed.device)
        tuple_pattern[:, idx, idx] = 2
        tuple_pattern = self.pattern_embedding(tuple_pattern) # (B, N, N, k_tuple_dim) .type(torch.LongTensor).to(tuple_pattern.get_device())
        
        emb2 = ef_mixed * tuple_pattern 
        
        emb2 = self.mix_lin(emb2) + emb2 #residual nn
        
        return emb2
        

class TwoFDisLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 activation_fn: nn.Module = nn.SiLU(),
                 **kwargs
                 ):
        super().__init__()
        
        self.emb_lins = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    activation_fn,
                    nn.Linear(hidden_dim, hidden_dim),
                    activation_fn
                ) for _ in range(3)
            ] 
        )
        


        self.output_lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, 
                kemb: torch.Tensor,
                **kwargs
                ):
        '''
            kemb: (B, N, N, hidden_dim)
        '''
        
        
        self_message, kemb_0, kemb_1 = [self.emb_lins[i](kemb.clone()) for i in range(3)]
        
        kemb_0, kemb_1 = (kemb_0.permute(0, 3, 1, 2), kemb_1.permute(0, 3, 1, 2))
        
        kemb_multed = torch.matmul(kemb_0, kemb_1).permute(0, 2, 3, 1)

        kemb_out = self.output_lin(self_message * kemb_multed) + (self_message * kemb_multed)
        
        return kemb_out




    
    