from torch.utils.data import DataLoader
from yacs.config import load_cfg
from typing import List, Optional
import sys
sys.path.append(".")

def get_cfgs(original_config_path: str, merge_list: Optional[dict] = None, 
               specific_config_path: Optional[str] = None, data_name: Optional[str] = None):
    
    # load config
    with open(original_config_path, "r") as f:
        cfg = load_cfg(f)

    if specific_config_path is not None:
        assert data_name is not None
        with open(specific_config_path, "r") as f:
            s_cfg = load_cfg(f).get(data_name)
        if s_cfg is not None:
            cfg.merge_from_other_cfg(s_cfg)
        else:
            print(f"No specific config for {data_name}")
    
    if merge_list is not None:
        cfg.merge_from_list(merge_list)

        
    return cfg









    
    
