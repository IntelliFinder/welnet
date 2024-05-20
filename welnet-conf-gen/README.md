
# WeLNet

## Reproduce the Molecular Conformation GEOM-QM9 results


### Build environment

If you don't have mamba installed you can replace mamba with conda in the commands.

```   
mamba create -n welnet-conf-gen python=3.9 -y
mamba activate welnet-conf-gen

mamba install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia


mamba install -y scikit-learn pandas decorator ipython networkx tqdm matplotlib
mamba install -y -c conda-forge easydict
pip install pyyaml wandb

pip install torch_geometric

# Optional dependencies:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

pip install rdkit==2022.9.1 rdkit

pip install yacs

```

###Important :
If you recieve the Error ModuleNotFoundError: No module named 'torch._six'

Please do the following:

In /your-root-to/miniconda-3/miniconda3/envs/welnet-conf-gen/lib/python3.9/site-packages/torch_geometric/data/dataloader.py

Replace:
#from torch._six import container_abcs, string_classes, int_classes

with:
import torch
import re
#from torch._six import container_abcs, string_classes, int_classes
import collections.abc as container_abcs


and code should work.


Please replace "your-path" in the code with your local path.


### Conformation Generation
Equilibrium conformation generation targets on predicting stable 3D structures from 2D molecular graphs. As in Clofnet, we follow [ConfGF](https://arxiv.org/abs/2105.03902), we evaluate  WeLNet on the GEOM-QM9 and GEOM-Drugs datasets ([Axelrod & Gomez-Bombarelli, 2020](https://arxiv.org/abs/2006.05531)) as well as the ISO17 dataset ([Sch¨utt et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)). For the score-based generation framework, we build our algorithm on the code by [ClofNet](https://github.com/mouthful/ClofNet), which is in turn based on the public codebase of [ConfGF](https://github.com/DeepGraphLearning/ConfGF). 

#### Dataset 
* **Offical Dataset**: The offical raw GEOM dataset is avaiable [[here]](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF).

* **Preprocessed dataset**: We use the preprocessed datasets (GEOM, ISO17) published by ConfGF([[google drive folder]](https://drive.google.com/drive/folders/10dWaj5lyMY0VY4Zl0zDPCa69cuQUGb-6?usp=sharing)).

#### Train
```
python -u script/train.py --config_path ./config/qm9_clofnet.yml
```
#### Generation
```
python -u script/gen.py --config_path ./config/qm9_clofnet.yml --generator EquiGF --eval_epoch [epoch] --start 0 --end 200
```

We provided a trained checkpoint. You can run the above line of code with --eval_epoch 342 and generate molecules.
Then copy the generated file path into the --input in the Evaluation line below.

#### Evaluation
```
python -u script/get_task1_results.py --input /root/to/generation --core 10 --threshold 0.5
```