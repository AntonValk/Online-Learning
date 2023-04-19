# Code for German Data

# All libraries
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pandas as pd
from tqdm import tqdm
from AuxDrop import AuxDrop_ODL
from dataset import dataset


# Data description
# "german", "svmguide3", "magic04", "a8a"
data_name = "a8a"

# Please change the value of hyperparameters in the dataset.py file corresponding to the chose data name
n_base_feat, max_num_hidden_layers, qtd_neuron_per_hidden_layer, n_classes, aux_layer, n_neuron_aux_layer, batch_size, b,  n, s, dropout_p, n_aux_feat,  use_cuda, X_base, X_aux_new, aux_mask, Y, label = dataset(data_name)

# Creating the Aux-Drop Model
model = AuxDrop_ODL(features_size = n_base_feat, max_num_hidden_layers = max_num_hidden_layers, qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer, 
        n_classes = n_classes, aux_layer = aux_layer, n_neuron_aux_layer = n_neuron_aux_layer, batch_size = batch_size, b = b, n = n, s = s,
        dropout_p = dropout_p, n_aux_feat = n_aux_feat,  use_cuda = use_cuda)
# # For Aux-Drop(OGD) use this
# model = AuxDrop_OGD()

# # For Aux-Drop(ODL) with 1st hidden layer as AuxLayer use this
# model = AuxDrop_ODL_1stlayer()

# Run the model
N = X_base.shape[0]
for i in tqdm(range(N)):
        model.partial_fit(X_base[i].reshape(1,n_base_feat), X_aux_new[i].reshape(1, n_aux_feat), aux_mask[i].reshape(1,n_aux_feat), Y[i].reshape(1))

prediction = []
for i in model.prediction:
        prediction.append(torch.argmax(i).item())

error = len(prediction) - sum(prediction == label)

print("The error in the ", data_name, " dataset is ", error)