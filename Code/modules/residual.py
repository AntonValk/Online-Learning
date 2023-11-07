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
    Fully-connected residual architechture with OGD online learning backbone
    """

    def __init__(self,
                 num_blocks_enc: int, num_layers_enc: int, layer_width_enc,
                 num_blocks_stage: int, num_layers_stage: int, layer_width_stage: int,
                 dropout: float, size_in: int, size_out: int,
                 embedding_dim: int, embedding_size: int, embedding_num: int, layer_norm: bool = True, eps: float = 1e-6, lr=1e-3):
        super().__init__()

        self.layer_width_enc = layer_width_enc
        
        self.embeddings = [Embedding(num_embeddings=embedding_size, embedding_dim=embedding_dim) for _ in
                           range(embedding_num)]
        
        self.n=lr

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

        self.model = t.nn.ModuleList(self.encoder_blocks + self.stage_blocks + self.embeddings)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.prediction = []
        self.loss_array = []

    def encode(self, x: t.Tensor, *args) -> t.Tensor:
        """
        x the continuous input : BxF
        e the categorical inputs BxC
        """

        backcast = x
        encoding = 0.0
        for i, block in enumerate(self.encoder_blocks):
            backcast, e = block(backcast)
            encoding = encoding + e

            # weighted average
            prototype = (encoding).sum(dim=1, keepdim=True)

            backcast = backcast - prototype / (i + 1.0)
            backcast = t.relu(backcast)

        # full_embedding = (encoding).sum(dim=1, keepdim=True)
        # full_embedding = full_embedding.squeeze(1)
        full_embedding = encoding#.squeeze(1)
        return full_embedding

    def decode(self, full_embedding: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast = full_embedding
        stage_forecast = 0.0
        for block in self.stage_blocks:
            backcast, f = block(backcast)
            stage_forecast = stage_forecast + f
        return stage_forecast

    def forward(self, x: t.Tensor, *args) -> Tuple[t.Tensor, t.Tensor]:
        """
        x the continuous input : BxF
        e the categorical inputs BxC
        """

        X = t.from_numpy(x).float()
        # aux_feat = args[0]
        # aux_mask = args[1]
        aux_feat = t.from_numpy(args[0]).float()
        aux_mask = t.from_numpy(args[1]).float()
        x = t.cat([X, aux_feat * aux_mask], axis=1)
        full_embedding = self.encode(x)
        return t.softmax(self.decode(full_embedding), dim=1)

    def validate_input_X(self, data):
        if len(data.shape) != 2:
            raise Exception(
                "Wrong dimension for this X data. It should have only two dimensions."
            )

    def validate_input_Y(self, data):
        if len(data.shape) != 1:
            raise Exception(
                "Wrong dimension for this Y data. It should have only one dimensions."
            )

    def partial_fit(self, X_data, aux_data, aux_mask, Y_data, show_loss=False):
        self.validate_input_X(X_data)
        self.validate_input_X(aux_data)
        self.validate_input_Y(Y_data)

        optimizer = optim.SGD(self.parameters(), lr=self.n)
        optimizer.zero_grad()
        y_pred = self.forward(X_data, aux_data, aux_mask)
        self.prediction = [y_pred]
        loss = self.loss_fn(y_pred, torch.tensor(Y_data, dtype=torch.long))
        self.loss_array = [loss.item()]
        loss.backward()
        optimizer.step()

        # if show_loss:
        #     print("Loss is: ", loss)


# Aux-Drop (OGD) code
class AuxDrop_OGD(nn.Module):
    def __init__(
        self,
        features_size,
        max_num_hidden_layers=5,
        qtd_neuron_per_hidden_layer=100,
        n_classes=2,
        aux_layer=3,
        n_neuron_aux_layer=100,
        batch_size=1,
        n_aux_feat=3,
        n=0.01,
        dropout_p=0.5,
    ):
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
            nn.Linear(self.features_size, self.qtd_neuron_per_hidden_layer, bias=True)
        )
        for i in range(self.max_layers - 1):
            # The input to the aux_layer is the outpout coming from its previous layer, i.e., "qtd_neuron_per_hidden_layer" and the
            # number of auxiliary features, "n_aux_feat".
            if i + 2 == self.aux_layer:
                self.hidden_layers.append(
                    nn.Linear(
                        self.n_aux_feat + self.qtd_neuron_per_hidden_layer,
                        self.n_neuron_aux_layer,
                        bias=True,
                    )
                )
            elif i + 1 == self.aux_layer:
                self.hidden_layers.append(
                    nn.Linear(
                        self.n_neuron_aux_layer,
                        self.qtd_neuron_per_hidden_layer,
                        bias=True,
                    )
                )
            else:
                self.hidden_layers.append(
                    nn.Linear(
                        qtd_neuron_per_hidden_layer,
                        qtd_neuron_per_hidden_layer,
                        bias=True,
                    )
                )
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        self.output_layer = nn.Linear(
            self.qtd_neuron_per_hidden_layer, self.n_classes, bias=True
        )

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
            if i == self.aux_layer - 1:
                # Input to the aux layer will be the output from its previous layer and the incoming auxiliary inputs.
                inp = F.relu(self.hidden_layers[i](torch.cat((aux_feat, inp), dim=1)))
                # Based on the incoming aux data, the aux inputs which do not come gets included in the dropout probability.
                # Based on that we calculate the probability to drop the left over neurons in auxiliary layer.

                aux_p = (
                    self.p * self.n_neuron_aux_layer
                    - (aux_mask.size()[1] - torch.sum(aux_mask))
                ) / (self.n_neuron_aux_layer - aux_mask.size()[1])
                binomial = torch.distributions.binomial.Binomial(probs=1 - aux_p)
                non_aux_mask = binomial.sample(
                    [1, self.n_neuron_aux_layer - aux_mask.size()[1]]
                )
                mask = torch.cat((aux_mask, non_aux_mask), dim=1)
                inp = inp * mask * (1.0 / (1 - self.p))
            else:
                inp = F.relu(self.hidden_layers[i](inp))

        out = F.softmax(self.output_layer(inp), dim=1)

        return out

    def validate_input_X(self, data):
        if len(data.shape) != 2:
            raise Exception(
                "Wrong dimension for this X data. It should have only two dimensions."
            )

    def validate_input_Y(self, data):
        if len(data.shape) != 1:
            raise Exception(
                "Wrong dimension for this Y data. It should have only one dimensions."
            )

    def partial_fit(self, X_data, aux_data, aux_mask, Y_data, show_loss=False):
        self.validate_input_X(X_data)
        self.validate_input_X(aux_data)
        self.validate_input_Y(Y_data)

        optimizer = optim.SGD(self.parameters(), lr=self.n)
        optimizer.zero_grad()
        y_pred = self.forward(X_data, aux_data, aux_mask)
        # self.prediction.append(y_pred)
        self.prediction = [y_pred]
        loss = self.loss_fn(y_pred, torch.tensor(Y_data, dtype=torch.long))
        self.loss_array.append(loss.item())
        loss.backward()
        optimizer.step()

        if show_loss:
            print("Loss is: ", loss)



class SingleStageResidualNetODL(t.nn.Module):
    """
    Fully-connected residual architechture with ODL online learning backbone
    """

    def __init__(self,
                 num_blocks_enc: int, num_layers_enc: int, layer_width_enc,
                 num_blocks_stage: int, num_layers_stage: int, layer_width_stage: int,
                 dropout: float, size_in: int, size_out: int, embedding_dim: int, embedding_size: int, embedding_num: int, batch_size: int = 1, 
                 layer_norm: bool = True, eps: float=1e-6, lr: float=1e-3,
                 b: float=0.99, s: float = 0.2, use_cuda: bool = False):
        super().__init__()

        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA :]")

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
        )

        self.layer_width_enc = layer_width_enc
        self.batch_size = batch_size
        self.n_classes = size_out

        self.max_num_hidden_layers = num_blocks_enc + num_blocks_stage
        
        self.embeddings = [Embedding(num_embeddings=embedding_size, embedding_dim=embedding_dim) for _ in
                           range(embedding_num)]
        
        self.n=lr
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(lr), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.p = Parameter(torch.tensor(dropout), requires_grad=False).to(self.device)

        # Stores hidden and output layers
        self.output_layers = []

        for i in range(self.max_num_hidden_layers):
            self.output_layers.append(nn.Linear(layer_width_enc, size_out))

        self.output_layers = nn.ModuleList(self.output_layers).to(self.device)

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

        self.model = t.nn.ModuleList(self.encoder_blocks + self.stage_blocks )

        # The alpha values sum to 1 and are equal at the beginning of the training.
        self.alpha = Parameter(
            torch.Tensor(self.max_num_hidden_layers).fill_(1 / (self.max_num_hidden_layers)), requires_grad=False
        ).to(self.device)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.prediction = []
        self.loss_array = []
        self.alpha_array = []
        self.layerwise_loss_array = []
        self.hidden_preds = []

        # Initialize the gradients of all the parameters with 0.
    def zero_grad(self):
        for i in range(self.max_num_hidden_layers):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
            self.model[i][-1].weight.grad.data.fill_(0)
            self.model[i][-1].weight.grad.data.fill_(0)


    def encode(self, x: t.Tensor, *args) -> t.Tensor:
        """
        x the continuous input : BxF
        e the categorical inputs BxC
        """

        backcast = x
        encoding = 0.0

        # print(len(self.prediction), len(self.loss_array), len(self.alpha_array), len(self.layerwise_loss_array), len(self.hidden_preds))
        self.hidden_preds = []

        # Forward pass of the first hidden layer. Apply the linear transformation and then relu. The output from the relu is the input
        # passed to the next layer.

        for i, block in enumerate(self.encoder_blocks):
            backcast, e = block(backcast)
            encoding = encoding + e

            # weighted average
            prototype = (encoding).sum(dim=1, keepdim=True)

            backcast = backcast - prototype / (i + 1.0)
            backcast = t.relu(backcast)
            self.hidden_preds.append(F.softmax(self.output_layers[i](backcast), dim=1))

        full_embedding = encoding
        return full_embedding

    def decode(self, full_embedding: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast = full_embedding
        stage_forecast = 0.0
        for i, block in enumerate(self.stage_blocks):
            backcast, f = block(backcast)
            stage_forecast = stage_forecast + f
            # print(stage_forecast.shape)
            # print(backcast.shape)
            # exit()
            # hid_out = self.hidden_layers[i+len(self.encoder_blocks)](backcast)
            # self.hidden_preds.append(F.softmax(self.output_layers[i+len(self.encoder_blocks)](hid_out), dim=1))
            self.hidden_preds.append(F.softmax(self.output_layers[i+len(self.encoder_blocks)](backcast), dim=1))
        pred_per_layer = torch.stack(self.hidden_preds)
        return pred_per_layer

    def forward(self, x: t.Tensor, *args) -> Tuple[t.Tensor, t.Tensor]:
        """
        x the continuous input : BxF
        e the categorical inputs BxC
        """

        X = t.from_numpy(x).float()
        aux_feat = t.from_numpy(args[0]).float()
        aux_mask = t.from_numpy(args[1]).float()
        x = t.cat([X, aux_feat * aux_mask], axis=1)
        full_embedding = self.encode(x)
        predictions_per_layer = self.decode(full_embedding)
        return torch.sum(torch.mul(self.alpha.view(self.max_num_hidden_layers, 1).repeat(1, self.batch_size).view(self.max_num_hidden_layers, self.batch_size, 1),
                predictions_per_layer), 0), predictions_per_layer

    def validate_input_X(self, data):
        if len(data.shape) != 2:
            raise Exception(
                "Wrong dimension for this X data. It should have only two dimensions."
            )

    def validate_input_Y(self, data):
        if len(data.shape) != 1:
            raise Exception(
                "Wrong dimension for this Y data. It should have only one dimensions."
            )

    def update_weights(self, X_data, aux_data, aux_mask, Y_data, show_loss=False):
        optimizer = optim.SGD(self.parameters(), lr=self.n)
        optimizer.zero_grad()
        Y = torch.from_numpy(Y_data).to(self.device)

        out, predictions_per_layer = self.forward(X_data, aux_data, aux_mask)
        self.prediction = [out]

        criterion = nn.CrossEntropyLoss().to(self.device)
        loss = criterion(
            out.view(self.batch_size, self.n_classes),
            Y.long(),
        )
        self.loss_array = [loss.item()]

        if show_loss:
            if (len(self.loss_array) % 1000) == 0:
                print(
                    "WARNING: Set 'show_loss' to 'False' when not debugging. "
                    "It will deteriorate the fitting performance."
                )
                loss = np.mean(self.loss_array[-1000:])
                print("Alpha:" + str(self.alpha.data.cpu().numpy()))
                print("Training Loss: " + str(loss))

        losses_per_layer = []

        for o in predictions_per_layer:
            criterion = nn.CrossEntropyLoss().to(self.device)
            loss = criterion(
                o.view(self.batch_size, self.n_classes),
                Y.long(),
            )
            losses_per_layer.append(loss)
        total_loss = torch.dot(self.alpha, torch.stack(losses_per_layer))
        total_loss.backward()
        optimizer.step()
        
        # w = [None] * (len(losses_per_layer))
        # b = [None] * (len(losses_per_layer))

        # with torch.no_grad():

        #     for i in range(len(losses_per_layer)):
        #         losses_per_layer[i].backward(retain_graph=True)

        #         self.output_layers[i].weight.data -= (
        #             self.n * self.alpha[i] * self.output_layers[i].weight.grad.data
        #         )
        #         self.output_layers[i].bias.data -= (
        #             self.n * self.alpha[i] * self.output_layers[i].bias.grad.data
        #         )

        #         if i < self.max_num_hidden_layers:
        #             for j in range(i):
        #                 if w[j] is None:
        #                     w[j] = (
        #                         self.alpha[i] * self.model[j][-1].weight.grad.data
        #                     )
        #                     b[j] = self.alpha[i] * self.model[j][-1].weight.grad.data
        #                 else:
        #                     w[j] += (
        #                         self.alpha[i] * self.model[j][-1].weight.grad.data
        #                     )
        #                     b[j] += self.alpha[i] * self.model[j][-1].weight.grad.data

        #         self.zero_grad()

            # for i in range(self.max_num_hidden_layers):
            #     self.model[i][-1].weight.grad.data -= self.n * w[i]
            #     self.model[i][-1].weight.grad.data -= self.n * b[i]
        with torch.no_grad():
            for i in range(len(losses_per_layer)):
                self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
                self.alpha[i] = torch.max(
                    self.alpha[i], self.s / (len(losses_per_layer))
                )
    
        z_t = torch.sum(self.alpha)
        self.alpha = Parameter(self.alpha / z_t, requires_grad=False).to(self.device)

        # # To save the loss
        # detached_loss = []
        # for i in range(len(losses_per_layer)):
        #     detached_loss.append(losses_per_layer[i].detach().numpy())
        # self.layerwise_loss_array.append(np.asarray(detached_loss))
        self.alpha_array = [self.alpha.detach().numpy()]
        self.hidden_preds = []

        # optimizer = optim.SGD(self.parameters(), lr=self.n)
        # optimizer.zero_grad()
        # y_pred = self.forward(X_data, aux_data, aux_mask)
        # self.prediction.append(y_pred)
        # loss = self.loss_fn(y_pred, torch.tensor(Y_data, dtype=torch.long))
        # self.loss_array.append(loss.item())
        # loss.backward()
        # optimizer.step()

        # if show_loss:
        #     print("Loss is: ", loss)
        # print(len(self.prediction), len(self.loss_array), len(self.alpha_array), len(self.layerwise_loss_array), len(self.hidden_preds))

    def partial_fit(self, X_data, aux_data, aux_mask, Y_data, show_loss=False):
        self.validate_input_X(X_data)
        self.validate_input_X(aux_data)
        self.validate_input_Y(Y_data)
        self.update_weights(X_data, aux_data, aux_mask, Y_data)


# code for Aux-Drop(ODL)
class Fast_AuxDrop_ODL(nn.Module):
    def __init__(
        self,
        features_size,
        max_num_hidden_layers,
        qtd_neuron_per_hidden_layer,
        n_classes,
        aux_layer,
        n_neuron_aux_layer,
        batch_size=1,
        b=0.99,
        n=0.01,
        s=0.2,
        dropout_p=0.5,
        n_aux_feat=3,
        use_cuda=False,
    ):
        super(Fast_AuxDrop_ODL, self).__init__()

        # features_size - Number of base features
        # max_num_hidden_layers - Number of hidden layers
        # qtd_neuron_per_hidden_layer - Number of nodes in each hidden layer except the AuxLayer
        # n_classes - The total number of classes (output labels)
        # aux_layer - The position of auxiliary layer. This code does not work if the AuxLayer position is 1.
        # n_neuron_aux_layer - The total numebr of neurons in the AuxLayer
        # batch_size - The batch size is always 1 since it is based on stochastic gradient descent
        # b - discount rate
        # n - learning rate
        # s - smoothing rate
        # dropout_p - The dropout rate in the AuxLayer
        # n_aux_feat - Number of auxiliary features

        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA :]")

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
        )

        self.features_size = features_size
        self.max_num_hidden_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.n_classes = n_classes
        self.aux_layer = aux_layer
        self.n_neuron_aux_layer = n_neuron_aux_layer
        self.batch_size = batch_size
        self.n_aux_feat = n_aux_feat

        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.p = Parameter(torch.tensor(dropout_p), requires_grad=False).to(self.device)

        # Stores hidden and output layers
        self.hidden_layers = []
        self.output_layers = []

        # The number of hidden layers would be "max_num_hidden_layers". The input to the first layer is the base features i.e. "features_size".
        self.hidden_layers.append(nn.Linear(features_size, qtd_neuron_per_hidden_layer))

        for i in range(max_num_hidden_layers - 1):
            # The input to the aux_layer is the outpout coming from its previous layer, i.e., "qtd_neuron_per_hidden_layer" and the
            # number of auxiliary features, "n_aux_feat".
            if i + 2 == aux_layer:
                self.hidden_layers.append(
                    nn.Linear(
                        n_aux_feat + qtd_neuron_per_hidden_layer, n_neuron_aux_layer
                    )
                )
            elif i + 1 == aux_layer:
                self.hidden_layers.append(
                    nn.Linear(n_neuron_aux_layer, qtd_neuron_per_hidden_layer)
                )
            else:
                self.hidden_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer)
                )

        # The number of output layer would be total number of layer ("max_num_hidden_layers") - 2 (no output from the aux layer and the first layer)
        # No output is taken from the first layer following the ODL paper (see ODL code and the baseline paragraph in the experiemnts section in ODL paper)
        for i in range(max_num_hidden_layers - 2):
            self.output_layers.append(nn.Linear(qtd_neuron_per_hidden_layer, n_classes))

        self.hidden_layers = nn.ModuleList(self.hidden_layers).to(self.device)
        self.output_layers = nn.ModuleList(self.output_layers).to(self.device)

        # Alphas are related to the output layers. So, the nummber of alphas would be equal to the number of hidden layers (max_num_hidden_layers) - 2.
        # The alpha value sums to 1 and is equal at the beginning of the training.
        self.alpha = Parameter(
            torch.Tensor(self.max_num_hidden_layers - 2).fill_(
                1 / (self.max_num_hidden_layers - 2)
            ),
            requires_grad=False,
        ).to(self.device)

        # We store loss, alpha and prediction parameter.
        self.loss_array = []
        self.alpha_array = []
        self.layerwise_loss_array = []
        self.prediction = []

        self.loss_fn = nn.CrossEntropyLoss()


    # Initialize the gradients of all the parameters with 0.
    def zero_grad(self):
        for i in range(self.max_num_hidden_layers - 2):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
        for i in range(self.max_num_hidden_layers):
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    # This function predicts the output from each layer,calculates the loss and do gradient descent and alpha update.
    def update_weights(self, X, aux_feat, aux_mask, Y, show_loss):
        optimizer = optim.SGD(self.parameters(), lr=self.n)
        optimizer.zero_grad()
        Y = torch.from_numpy(Y).to(self.device)

        # Predictions from each layer using the implemented forward function.
        predictions_per_layer = self.forward(X, aux_feat, aux_mask)

        real_output = torch.sum(
            torch.mul(
                self.alpha.view(self.max_num_hidden_layers - 2, 1)
                .repeat(1, self.batch_size)
                .view(self.max_num_hidden_layers - 2, self.batch_size, 1),
                predictions_per_layer,
            ),
            0,
        )
        self.prediction = [real_output]
        # self.prediction.append(real_output)

        criterion = nn.CrossEntropyLoss().to(self.device)
        loss = criterion(
            real_output.view(self.batch_size, self.n_classes),
            Y.view(self.batch_size).long(),
        )
        self.loss_array = [loss.detach().numpy()]

        # if show_loss:
        #     if (len(self.loss_array) % 1000) == 0:
        #         print(
        #             "WARNING: Set 'show_loss' to 'False' when not debugging. "
        #             "It will deteriorate the fitting performance."
        #         )
        #         loss = np.mean(self.loss_array[-1000:])
        #         print("Alpha:" + str(self.alpha.data.cpu().numpy()))
        #         print("Training Loss: " + str(loss))

        losses_per_layer = []

        for out in predictions_per_layer:
            criterion = nn.CrossEntropyLoss().to(self.device)
            loss = criterion(
                out.view(self.batch_size, self.n_classes),
                Y.view(self.batch_size).long(),
            )
            losses_per_layer.append(loss)

        total_loss = torch.dot(self.alpha, torch.stack(losses_per_layer))
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            for i in range(len(losses_per_layer)):
                self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
                self.alpha[i] = torch.max(
                    self.alpha[i], self.s / (len(losses_per_layer))
                )

        z_t = torch.sum(self.alpha)
        self.alpha = Parameter(self.alpha / z_t, requires_grad=False).to(self.device)

        # To save the loss
        # detached_loss = []
        # for i in range(len(losses_per_layer)):
        #     detached_loss.append(losses_per_layer[i].detach().numpy())
        # self.layerwise_loss_array.append(np.asarray(detached_loss))
        # self.alpha_array.append(self.alpha.detach().numpy())
        self.alpha_array = [self.alpha]

    # Forward pass. Get the output from each layer.
    def forward(self, X, aux_feat, aux_mask):
        hidden_connections = []
        linear_x = []
        relu_x = []

        X = torch.from_numpy(X).float().to(self.device)
        aux_feat = torch.from_numpy(aux_feat).float().to(self.device)
        aux_mask = torch.from_numpy(aux_mask).float().to(self.device)

        # Forward pass of the first hidden layer. Apply the linear transformation and then relu. The output from the relu is the input
        # passed to the next layer.
        linear_x.append(self.hidden_layers[0](X))
        relu_x.append(F.relu(linear_x[0]))
        hidden_connections.append(relu_x[0])

        for i in range(1, self.max_num_hidden_layers):
            # Forward pass to the Aux layer.
            if i == self.aux_layer - 1:
                # Input to the aux layer will be the output from its previous layer and the incoming auxiliary inputs.
                linear_x.append(
                    self.hidden_layers[i](
                        torch.cat((aux_feat, hidden_connections[i - 1]), dim=1)
                    )
                )
                relu_x.append(F.relu(linear_x[i]))
                # We apply dropout in the aux layer.
                # Based on the incoming aux data, the aux inputs which do not come gets included in the dropout probability.
                # Based on that we calculate the probability to drop the left over neurons in auxiliary layer.
                aux_p = (
                    self.p * self.n_neuron_aux_layer
                    - (aux_mask.size()[1] - torch.sum(aux_mask))
                ) / (self.n_neuron_aux_layer - aux_mask.size()[1])
                binomial = torch.distributions.binomial.Binomial(probs=1 - aux_p)
                non_aux_mask = binomial.sample(
                    [1, self.n_neuron_aux_layer - aux_mask.size()[1]]
                )
                mask = torch.cat((aux_mask, non_aux_mask), dim=1)
                hidden_connections.append(relu_x[i] * mask * (1.0 / (1 - self.p)))
            else:
                linear_x.append(self.hidden_layers[i](hidden_connections[i - 1]))
                relu_x.append(F.relu(linear_x[i]))
                hidden_connections.append(relu_x[i])

        output_class = []

        for i in range(self.max_num_hidden_layers - 1):
            if i < self.aux_layer - 2:
                output_class.append(
                    F.softmax(self.output_layers[i](hidden_connections[i + 1]), dim=1)
                )
            if i > self.aux_layer - 2:
                output_class.append(
                    F.softmax(
                        self.output_layers[i - 1](hidden_connections[i + 1]), dim=1
                    )
                )

        pred_per_layer = torch.stack(output_class)

        return pred_per_layer

    def validate_input_X(self, data):
        if len(data.shape) != 2:
            raise Exception(
                "Wrong dimension for this X data. It should have only two dimensions."
            )

    def validate_input_Y(self, data):
        if len(data.shape) != 1:
            raise Exception(
                "Wrong dimension for this Y data. It should have only one dimensions."
            )

    def partial_fit(self, X_data, aux_data, aux_mask, Y_data, show_loss=False):
        self.validate_input_X(X_data)
        self.validate_input_X(aux_data)
        self.validate_input_Y(Y_data)
        self.update_weights(X_data, aux_data, aux_mask, Y_data, show_loss)
