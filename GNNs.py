import torch.nn as nn
from torch_geometric.nn import GCNConv, RGCNConv, GATv2Conv,GATConv
import torch
from torch_geometric.nn.models import MLP

class GCN(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = model_config['gnn_hidden_dim']
        self.n_layers = model_config['gnn_n_layers']
        self.convs = nn.ModuleList([])
        self.linear_in = nn.Linear(model_config['lm_input_dim'], self.hidden_dim)
  
        for i in range(self.n_layers):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))

        self.dropout = nn.Dropout(model_config['dropout'])
        
        self.activation_name = model_config['activation'].lower()
        if self.activation_name == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif self.activation_name == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_name == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError('Please choose activation function from "leakyrelu", "relu" or "elu".')
        
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, 2)

    def forward(self, x, edge_index, edge_type):
        x = self.linear_in(x)
        x = self.dropout(x)
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.activation(x)
        x = self.linear_pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear_out(x)



class RGCN(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = model_config['gnn_hidden_dim']
        self.n_layers = model_config['gnn_n_layers']
        self.convs = nn.ModuleList([])
        self.linear_in = nn.Linear(model_config['lm_input_dim'], self.hidden_dim)
  
        for i in range(self.n_layers):
            self.convs.append(RGCNConv(self.hidden_dim, self.hidden_dim, model_config['n_relations']))

        self.dropout = nn.Dropout(model_config['dropout'])
        
        self.activation_name = model_config['activation'].lower()
        if self.activation_name == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif self.activation_name == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_name == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError('Please choose activation function from "leakyrelu", "relu" or "elu".')
        
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, 2)

    def forward(self, x, edge_index, edge_type):
        x = self.linear_in(x)
        x = self.dropout(x)
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_type)
            x = self.activation(x)
        x = self.linear_pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear_out(x)


class GAT(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = model_config['gnn_hidden_dim']
        self.n_layers = model_config['gnn_n_layers']
        self.heads = model_config['att_heads']  # Number of attention heads
        
        self.convs = nn.ModuleList([])
        self.linear_in = nn.Linear(model_config['lm_input_dim'], self.hidden_dim)
  
        for i in range(self.n_layers):
            if i == self.n_layers - 1:  # Final layer should not use multiple heads
                self.convs.append(GATConv(self.hidden_dim * self.heads, self.hidden_dim, heads=1))
            else:
                self.convs.append(GATConv(self.hidden_dim, self.hidden_dim, heads=self.heads))

        self.dropout = nn.Dropout(model_config['dropout'])
        
        self.activation_name = model_config['activation'].lower()
        if self.activation_name == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif self.activation_name == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_name == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError('Please choose activation function from "leakyrelu", "relu" or "elu".')
        
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, 2)

    def forward(self, x, edge_index, edge_type):
        x = self.linear_in(x)
        x = self.dropout(x)
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.linear_pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear_out(x)

class GATv2(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = model_config['gnn_hidden_dim']
        self.n_layers = model_config['gnn_n_layers']
        self.heads = model_config['att_heads']  # Number of attention heads
        
        self.convs = nn.ModuleList([])
        self.linear_in = nn.Linear(model_config['lm_input_dim'], self.hidden_dim)
  
        for i in range(self.n_layers):
            if i == self.n_layers - 1:  # Final layer should not use multiple heads
                self.convs.append(GATv2Conv(self.hidden_dim * self.heads, self.hidden_dim, heads=1))
            else:
                self.convs.append(GATv2Conv(self.hidden_dim, self.hidden_dim, heads=self.heads))

        self.dropout = nn.Dropout(model_config['dropout'])
        
        self.activation_name = model_config['activation'].lower()
        if self.activation_name == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif self.activation_name == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_name == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError('Please choose activation function from "leakyrelu", "relu" or "elu".')
        
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, 2)

    def forward(self, x, edge_index, edge_type):
        x = self.linear_in(x)
        x = self.dropout(x)
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.linear_pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear_out(x)
    