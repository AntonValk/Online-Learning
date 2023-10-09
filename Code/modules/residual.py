# Libraries requied
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim

import torch as t
from modules.custom_layers import Embedding, FCBlock
from modules.custom_layers import FCBlockNorm
from typing import Tuple


class SingleStageResidualNet(t.nn.Module):
    """
    Fully-connected residual architechture with many categorical inputs wrapped in embeddings
    """

    def __init__(self,
                 num_blocks_enc: int, num_layers_enc: int, layer_width_enc,
                 num_blocks_stage: int, num_layers_stage: int, layer_width_stage: int,
                 dropout: float, size_in: int, size_out: int,
                 embedding_dim: int, embedding_size: int, embedding_num: int, layer_norm: bool = True, eps: float = 1e-6):
        super().__init__()

        self.layer_width_enc = layer_width_enc

        self.embeddings = [Embedding(num_embeddings=embedding_size, embedding_dim=embedding_dim) for _ in
                           range(embedding_num)]

        # IMPORTANT NOTE: We don't use LayerNorm in these blocks as this would inject inputs without taking into account their weights
        self.encoder_blocks = [FCBlock(num_layers=num_layers_enc, layer_width=layer_width_enc, dropout=dropout,
                                             size_in=size_in + embedding_dim * embedding_num, size_out=layer_width_enc)]
        self.encoder_blocks += [FCBlock(num_layers=num_layers_enc, layer_width=layer_width_enc, dropout=dropout,
                                              size_in=layer_width_enc, size_out=layer_width_enc) for _ in range(num_blocks_enc - 1)]

        if layer_norm:
            self.stage_blocks = [FCBlockNorm(num_layers=num_layers_stage, layer_width=layer_width_stage, dropout=dropout,
                                                size_in=layer_width_enc, size_out=size_out, eps=eps)] + \
                                 [FCBlockNorm(num_layers=num_layers_stage, layer_width=layer_width_stage, dropout=dropout,
                                                size_in=layer_width_stage, size_out=size_out, eps=eps) for _ in range(num_blocks_stage - 1)]
        else:
            self.stage_blocks = [FCBlock(num_layers=num_layers_stage, layer_width=layer_width_stage,
                                              dropout=dropout, size_in=layer_width_enc, size_out=size_out)] + \
                                 [FCBlock(num_layers=num_layers_stage, layer_width=layer_width_stage,
                                              dropout=dropout, size_in=layer_width_stage, size_out=size_out) for _ in range(num_blocks_stage - 1)]

        self.model = t.nn.ModuleList(
            self.encoder_blocks + self.stage_blocks + self.embeddings)

    def encode(self, x: t.Tensor, weights: t.Tensor, *args) -> t.Tensor:
        """
                x the continuous input : BxNxF
                weights the weight of each input BxN
                e the categorical inputs BxNxC
                """

        weights = weights.unsqueeze(2)
        weights_sum = weights.sum(dim=1, keepdim=True)

        ee = [x]
        for i, v in enumerate(args):
            ee.append(self.embeddings[i](v))
        backcast = t.cat(ee, dim=-1)

        encoding = 0.0
        for i, block in enumerate(self.encoder_blocks):
            backcast, e = block(backcast)
            encoding = encoding + e

            # weighted average
            prototype = (encoding * weights).sum(dim=1, keepdim=True) / weights_sum

            backcast = backcast - prototype / (i + 1.0)
            backcast = t.relu(backcast)

        pose_embedding = (encoding * weights).sum(dim=1, keepdim=True) / weights_sum
        pose_embedding = pose_embedding.squeeze(1)
        return pose_embedding

    def decode(self, pose_embedding: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast = pose_embedding
        stage_forecast = 0.0
        for block in self.stage_blocks:
            backcast, f = block(backcast)
            stage_forecast = stage_forecast + f

        return stage_forecast

    def forward(self, x: t.Tensor, weights: t.Tensor, *args) -> Tuple[t.Tensor, t.Tensor]:
        """
        x the continuous input : BxNxF
        weights the weight of each input BxN
        e the categorical inputs BxNxC
        """

        pose_embedding = self.encode(x, weights, *args)
        return self.decode(pose_embedding)



# Aux-Drop (OGD) code
class AuxDrop_OGD(nn.Module):
    def __init__(self, features_size, max_num_hidden_layers = 5, qtd_neuron_per_hidden_layer = 100, n_classes = 2, aux_layer = 3, 
                 n_neuron_aux_layer = 100, batch_size=1, n_aux_feat = 3, n=0.01, dropout_p=0.5):
        super(AuxDrop_OGD, self).__init__()

        self.features_size = features_size 
        self.max_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.aux_layer = aux_layer 
        self.n_neuron_aux_layer = n_neuron_aux_layer
        self.n_aux_feat = n_aux_feat
        self.n_classes = n_classes
        self.p = dropout_p
        self.batch_size = batch_size
        self.n = n

        # Stores hidden layers
        self.hidden_layers = []

        self.hidden_layers.append(
            nn.Linear(self.features_size, self.qtd_neuron_per_hidden_layer, bias=True))
        for i in range(self.max_layers - 1):
            # The input to the aux_layer is the outpout coming from its previous layer, i.e., "qtd_neuron_per_hidden_layer" and the 
            # number of auxiliary features, "n_aux_feat".
            if i+2 == self.aux_layer:
                self.hidden_layers.append(
                    nn.Linear(self.n_aux_feat + self.qtd_neuron_per_hidden_layer, self.n_neuron_aux_layer, bias=True))
            elif i + 1 == self.aux_layer:
                self.hidden_layers.append(
                    nn.Linear(self.n_neuron_aux_layer, self.qtd_neuron_per_hidden_layer, bias=True))
            else:
                self.hidden_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer, bias=True))
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        self.output_layer = nn.Linear(self.qtd_neuron_per_hidden_layer, 
            self.n_classes, bias=True)

        self.loss_fn = nn.CrossEntropyLoss()
        
        self.prediction = []
        self.loss_array = []
    
    def forward(self, X, aux_feat, aux_mask):

        X = torch.from_numpy(X).float()
        aux_feat = torch.from_numpy(aux_feat).float()
        aux_mask = torch.from_numpy(aux_mask).float()

        # Forward pass of the first hidden layer. Apply the linear transformation and then relu. The output from the relu is the input
        # passed to the next layer.
        inp = F.relu(self.hidden_layers[0](X))

        for i in range(1, self.max_layers):
            # Forward pass to the Aux layer.
            if i==self.aux_layer-1:
                # Input to the aux layer will be the output from its previous layer and the incoming auxiliary inputs.
                inp = F.relu(self.hidden_layers[i](torch.cat((aux_feat, inp), dim=1)))
                # Based on the incoming aux data, the aux inputs which do not come gets included in the dropout probability.
                # Based on that we calculate the probability to drop the left over neurons in auxiliary layer.
                
                aux_p = (self.p * self.n_neuron_aux_layer - (aux_mask.size()[1] - torch.sum(aux_mask)))/(self.n_neuron_aux_layer 
                            - aux_mask.size()[1])
                binomial = torch.distributions.binomial.Binomial(probs=1-aux_p)
                non_aux_mask = binomial.sample([1, self.n_neuron_aux_layer - aux_mask.size()[1]])
                mask = torch.cat((aux_mask, non_aux_mask), dim = 1)
                inp = inp*mask*(1.0/(1-self.p))
            else:
                inp = F.relu(self.hidden_layers[i](inp))
        
        out = F.softmax(self.output_layer(inp), dim=1)

        return out