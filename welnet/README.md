
# WeLNet

## Reproduce the N-Body dynamics task results:


### Build environment

```   
conda create -n welnet python=3.9 -y
conda activate welnet
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install wandb

```

### Create Dataset

```
cd n_body_system/dataset
python -u generate_dataset.py --num-train 10000 --seed 43 --sufix small
```


#### Run the N-Body experiment

```
python -u main_nbody.py --exp_name test_centralize_5smallwldim --model egnn_vel --max_training_samples 3000 --lr 1e-3 --nf 128 \
--ef 9 --shared True --mixed True --silu True --count 5 --clip 1.5 --num_vectors 5 --wl_dim 32 --prop False --epochs 8000 --decay 0.5
```
