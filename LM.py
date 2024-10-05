from transformers import AutoModelForMaskedLM, AutoModel, AutoModelForSeq2SeqLM
import torch.nn as nn
from torch_geometric.nn.models import MLP
import torch


class LM_Model(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.LM_model_name = model_config['lm_model']

        model_map = {
            'roberta': 'roberta-base',
            'bert': 'bert-base-uncased',
            'albert': 'albert-base-v2',
            'xlnet': 'xlnet-base-cased',
        }

        if self.LM_model_name in model_map:
            self.LM = AutoModel.from_pretrained(model_map[self.LM_model_name])
        else:
            raise ValueError(f"Unsupported LM model: {self.LM_model_name}")
        
        # Define the classifier using an MLP
        self.classifier = MLP(
            in_channels=self.LM.config.hidden_size,
            hidden_channels=model_config['classifier_hidden_dim'],
            out_channels=2,
            num_layers=model_config['classifier_n_layers'],
            act=model_config['activation']
        )

    def forward(self, tokenized_tensors):
        
        out = self.LM(output_hidden_states=True, **tokenized_tensors)['hidden_states'][-1]

        # Mean pooling over the last hidden state
        embedding = out.mean(dim=1)

        # Return both the embedding and the classifier output
        return embedding, self.classifier(embedding)