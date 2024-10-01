# EUREKHA: Enhancing User Representation for Key Hackers Identification in Underground Forums

## Requirements
Run following command to create environment for reproduction (for cuda 10.2):
```
conda env create -f eurekha.yaml
conda activate eurekha
pip install torch==1.12.0+cu102 torchvision==0.13.0+cu102 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu102
```
For ```pyg_lib```, ```torch_cluster```, ```torch_scatter```, ```torch_sparse``` and ```torch_spline_conv```, please download [here](https://data.pyg.org/whl/torch-1.12.0%2Bcu102.html) and install locally.
```
pip install pyg_lib-0.1.0+pt112cu102-cp39-cp39-linux_x86_64.whl torch_cluster-1.6.0+pt112cu102-cp39-cp39-linux_x86_64.whl torch_scatter-2.1.0+pt112cu102-cp39-cp39-linux_x86_64.whl torch_sparse-0.6.16+pt112cu102-cp39-cp39-linux_x86_64.whl torch_spline_conv-1.2.1+pt112cu102-cp39-cp39-linux_x86_64.whl
```
## Data preparation
It is assumed that you have a dataset in a csv format. 

## Training
Run the following commands to train:
```
main.py --project_name eurekha --experiment_name hackforums --dataset hackforums --device 0 --LM_pretrain_epochs 2.2  --batch_size_LM 16 --LM_model bert --GNN_model rgcn 
```
