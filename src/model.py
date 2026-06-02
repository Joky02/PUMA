import os

import torch
import torch_geometric
from torch_geometric.nn import MessagePassing, GraphNorm
import gzip
import pickle
import numpy as np
import time
from omegaconf import DictConfig
import torch.nn as nn
from torch_geometric.utils import softmax



class GNNPredictor(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.embd_size = config.embd_size
        self.depth = config.depth
        cons_nfeats = 4
        var_nfeats = 6

        
        self.cons_embedding = nn.Sequential(
            nn.BatchNorm1d(cons_nfeats),
            nn.Linear(cons_nfeats, self.embd_size),
            nn.ReLU(),
            nn.Linear(self.embd_size, self.embd_size),
            nn.ReLU(),
        )


        self.var_embedding = nn.Sequential(
            nn.BatchNorm1d(var_nfeats),
            nn.Linear(var_nfeats, self.embd_size),
            nn.ReLU(),
            nn.Linear(self.embd_size, self.embd_size),
            nn.ReLU(),
        )

        self.conv_v_to_c_layers = nn.Sequential()
        self.conv_c_to_v_layers = nn.Sequential()
        self.graph_norm_v_to_c_layers = nn.Sequential()
        self.graph_norm_c_to_v_layers = nn.Sequential()
        
        for _ in range(self.depth):
            self.conv_v_to_c_layers.append(BipartiteGraphConvolution(config)) # TODO
            self.graph_norm_v_to_c_layers.append(GraphNorm(self.embd_size))
            self.conv_c_to_v_layers.append(BipartiteGraphConvolution(config)) # TODO
            self.graph_norm_c_to_v_layers.append(GraphNorm(self.embd_size))


        self.output_norm = GraphNorm((self.depth + 1) * self.embd_size)
        self.vars_output_layer = torch.nn.Sequential(
            nn.Linear((self.depth + 1) * self.embd_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1) #, bias=True),
        )
        
        self.cons_output_layer = torch.nn.Sequential(
            nn.Linear((self.depth + 1) * self.embd_size, self.embd_size),
            nn.ReLU(),
            nn.Linear(self.embd_size, 1) #, bias=True),
        )

    def forward(self, graph):
        constraint_features = graph.constraint_features.float()
        edge_indices = graph.edge_index
        edge_features = graph.edge_attr.float()
        variable_features = graph.variable_features.float()
        
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features_batch = graph.constraint_features_batch if hasattr(graph, "constraint_features_batch") else None
        variable_features_batch = graph.variable_features_batch if hasattr(graph, "variable_features_batch") else None
        
        vars_outputs = [variable_features]
        cons_outputs = [constraint_features]
        for i in range(self.depth):            

            constraint_features = self.conv_v_to_c_layers[i](
                variable_features, reversed_edge_indices, edge_features, constraint_features
            )
            variable_features = self.conv_c_to_v_layers[i](
                constraint_features, edge_indices, edge_features, variable_features
            )
            
            constraint_features = self.graph_norm_v_to_c_layers[i](
                constraint_features, constraint_features_batch)
            variable_features = self.graph_norm_c_to_v_layers[i](
                variable_features, variable_features_batch)
            
            vars_outputs.append(variable_features)
            cons_outputs.append(constraint_features)

        variable_features = torch.cat(vars_outputs, dim=-1)
        vars_out = self.vars_output_layer(variable_features).squeeze(-1)

        return vars_out, None

class BipartiteGraphConvolution(MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self, config):
        super().__init__("mean") 
        self.embd_size = config.embd_size

        self.feature_module_final = nn.Sequential(
            nn.Linear(self.embd_size * 2 + 1, self.embd_size),
            nn.ReLU(),
            nn.Linear(self.embd_size, self.embd_size),
        )

        self.post_conv_module = nn.Sequential(
            nn.BatchNorm1d(self.embd_size)
        )

        self.output_module = nn.Sequential(
            nn.Linear(2 * self.embd_size, self.embd_size),
            nn.ReLU(),
            nn.Linear(self.embd_size, self.embd_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """

        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )

        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )


    def message(self, node_features_i, node_features_j, edge_features):

        output = self.feature_module_final(
            torch.cat([node_features_i, edge_features, node_features_j], dim=-1)
        )

        return output

    
class BipartiteGATConvolution(MessagePassing):
    """
    Graph Attention Network (GAT) layer designed for bipartite graphs.
    """
    def __init__(self, config: DictConfig, heads: int = 4, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.embd_size = config.embd_size
        self.heads = heads
        self.head_dim = self.embd_size // heads
        assert self.embd_size % heads == 0, "Embedding size must be divisible by number of heads"

        # Linear layers W_q, W_k, W_v
        self.lin_q = nn.Linear(self.embd_size, self.embd_size)
        self.lin_k = nn.Linear(self.embd_size, self.embd_size)
        self.lin_v = nn.Linear(self.embd_size, self.embd_size)

        # Attention vector a
        self.attn_vector = nn.Parameter(torch.Tensor(1, self.heads, self.head_dim * 2 + 1)) # +1 for edge_features

        # Output module
        self.output_module = nn.Sequential(
            nn.Linear(2 * self.embd_size, self.embd_size),
            nn.ReLU(),
            nn.Linear(self.embd_size, self.embd_size),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_q.weight)
        nn.init.xavier_uniform_(self.lin_k.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)
        nn.init.xavier_uniform_(self.attn_vector)

    def forward(self, left_features, edge_indices, edge_features, right_features):
        aggregated_message = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )

        # propagate returns aggregated_message as [num_right_nodes, embd_size]
        aggregated_message = aggregated_message.view(-1, self.embd_size)
        
        output = self.output_module(
            torch.cat([aggregated_message, right_features], dim=-1)
        )
        return output

    def message(self, node_features_i, node_features_j, edge_features, index, ptr, size_i):
        q_features_i = self.lin_q(node_features_i).view(-1, self.heads, self.head_dim)
        k_features_j = self.lin_k(node_features_j).view(-1, self.heads, self.head_dim)
        v_features_j = self.lin_v(node_features_j).view(-1, self.heads, self.head_dim)
        
        attention_features = torch.cat([
            q_features_i, 
            k_features_j,
            edge_features.unsqueeze(1).repeat(1, self.heads, 1)
        ], dim=-1)
        
        attn_logits = (attention_features * self.attn_vector).sum(dim=-1)
        attn_logits = torch.nn.functional.leaky_relu(attn_logits)
        
        attn_weights = softmax(attn_logits, index, ptr, size_i)
        
        message_3d = v_features_j * attn_weights.unsqueeze(-1)

        # Flatten the 3D message [num_edges, heads, head_dim] into
        # [num_edges, heads * head_dim] (= [num_edges, embd_size]) before returning.
        return message_3d.view(-1, self.heads * self.head_dim)

class GATPredictor(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.embd_size = config.embd_size
        self.depth = config.depth
        self.heads = config.get('num_heads', 4) # number of GAT heads (default 4)

        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 6

        self.cons_embedding = nn.Sequential(
            nn.BatchNorm1d(cons_nfeats),
            nn.Linear(cons_nfeats, self.embd_size),
            nn.ReLU(),
            nn.Linear(self.embd_size, self.embd_size),
            nn.ReLU(),
        )

        self.edge_embedding = nn.Sequential(
            nn.BatchNorm1d(edge_nfeats),
        )

        self.var_embedding = nn.Sequential(
            nn.BatchNorm1d(var_nfeats),
            nn.Linear(var_nfeats, self.embd_size),
            nn.ReLU(),
            nn.Linear(self.embd_size, self.embd_size),
            nn.ReLU(),
        )
        
        # Store layers in ModuleLists
        self.conv_v_to_c_layers = nn.ModuleList()
        self.conv_c_to_v_layers = nn.ModuleList()
        self.graph_norm_v_to_c_layers = nn.ModuleList()
        self.graph_norm_c_to_v_layers = nn.ModuleList()
        
        for _ in range(self.depth):
            self.conv_v_to_c_layers.append(BipartiteGATConvolution(config, heads=self.heads))
            self.graph_norm_v_to_c_layers.append(GraphNorm(self.embd_size))
            self.conv_c_to_v_layers.append(BipartiteGATConvolution(config, heads=self.heads))
            self.graph_norm_c_to_v_layers.append(GraphNorm(self.embd_size))

        # Output layer
        self.vars_output_layer = torch.nn.Sequential(
            nn.Linear((self.depth + 1) * self.embd_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.cons_output_layer = torch.nn.Sequential(
            nn.Linear((self.depth + 1) * self.embd_size, self.embd_size),
            nn.ReLU(),
            nn.Linear(self.embd_size, 1)
        )

    def forward(self, graph):
        constraint_features = graph.constraint_features
        edge_indices = graph.edge_index
        edge_features = graph.edge_attr
        variable_features = graph.variable_features
        
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # Embedding
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features_batch = graph.constraint_features_batch if hasattr(graph, "constraint_features_batch") else None
        variable_features_batch = graph.variable_features_batch if hasattr(graph, "variable_features_batch") else None
        
        vars_outputs = [variable_features]
        cons_outputs = [constraint_features]
        
        # GNN Layers
        for i in range(self.depth):            
            constraint_features = self.conv_v_to_c_layers[i](
                variable_features, reversed_edge_indices, edge_features, constraint_features
            )
            variable_features = self.conv_c_to_v_layers[i](
                constraint_features, edge_indices, edge_features, variable_features
            )
            
            constraint_features = self.graph_norm_v_to_c_layers[i](
                constraint_features, constraint_features_batch)
            variable_features = self.graph_norm_c_to_v_layers[i](
                variable_features, variable_features_batch)
            
            vars_outputs.append(variable_features)
            cons_outputs.append(constraint_features)

        # Output
        variable_features = torch.cat(vars_outputs, dim=-1)
        vars_out = self.vars_output_layer(variable_features).squeeze(-1)

        # cons_out is not needed here, but the layer must exist
        return vars_out, None

class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)


    def process_sample(self, filepath):
        BGFilepath = filepath
        TensorFilepath = filepath.replace("samples", "tensors")
        with open(BGFilepath, "rb") as f:
            graph = pickle.load(f)
            
        with open(TensorFilepath, "rb") as f:
            Tensors = pickle.load(f)
      
        return graph, Tensors


    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """

        BGFilePath = self.sample_files[index]
        TensorFilepath = BGFilePath.replace("samples", "tensors")
        
        with open(BGFilePath, "rb") as f:
            BG = pickle.load(f)
            
        with open(TensorFilepath, "rb") as f:
            Tensors = pickle.load(f)

        constraint_features, edge_indices, edge_features, variable_features = BG

        graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features)
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        
        graph.A = Tensors[0]
        graph.b = Tensors[1]
        graph.c = Tensors[2]
        
        return graph

class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features=None,
        edge_indices=None,
        edge_features=None,
        variable_features=None,

    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)