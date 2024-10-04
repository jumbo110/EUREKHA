# EUREKHA: Enhancing User Representation for Key Hackers Identification in Underground Forums

Underground forums serve as hubs for cybercriminal activities, offering a space for anonymity and evasion of conventional online oversight. In these hidden communities, malicious actors collaborate to exchange illicit knowledge, tools, and tactics, driving a range of cyber threats—from hacking techniques to the sale of stolen data, malware, and zero-day exploits. Identifying the key instigators (i.e., key hackers), behind these operations is essential but remains a complex challenge. This paper presents a novel method called EUREKHA (Enhancing User Representation for Key Hacker Identification in Underground Forums), designed to identify these key hackers by modeling each user as a textual sequence. This sequence is processed through a large language model (LLM) for domain-specific adaptation, with LLMs acting as feature extractors. These extracted features are then fed into Graph Neural Network (GNN) to model user structural relationships, significantly improving identification accuracy.
Furthermore, we employ BERTopic (Bidirectional Encoder Representations from Transformers Topic Modeling) to extract personalized topics from user-generated content, enabling multiple textual representations per user and optimizing the selection of the most representative sequence. Our study demonstrates that fine-tuned LLMs outperform state-of-the-art methods in identifying key hackers. Additionally, when combined with GNNs, our model achieves significant improvements, resulting in approximately 6\% and 10\% increases in accuracy and F1-score, respectively, over existing methods. EUREKHA was tested on the dataset, and we provide open-source access to our code.

## I will publish the code once the paper is accepted.
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
