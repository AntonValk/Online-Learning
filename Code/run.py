# All libraries
import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pandas as pd
from tqdm import tqdm
from AuxDrop import AuxDrop_ODL, AuxDrop_OGD, AuxDrop_ODL_AuxLayer1stlayer
from AuxDrop import AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer, AuxDrop_ODL_RandomAllLayer 
from AuxDrop import AuxDrop_ODL_RandomInAuxLayer, AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst
from dataset import dataset
from joblib import Parallel, delayed

# Data description
# "german", "svmguide3", "magic04", "a8a", "ItalyPowerDemand", "SUSY", "HIGGS"
data_name = "german"

# Choose the type of data unavailability
# type can be - "variable_p", "trapezoidal", "obsolete_sudden"
type = "variable_p"

# Choose a model to run
# "AuxDrop_ODL" - Aux-Drop applied on ODL framework
#  "AuxDrop_OGD" - Aux-Drop applied on OGD framework
# "AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer" -  On ODL framework, Aux-Dropout in AuxLayer and Random dropout in all the other layers
# "AuxDrop_ODL_RandomAllLayer" - On ODL framework, Random Dropout applied in all the layers
#  "AuxDrop_ODL_RandomInAuxLayer" - On ODL framework, Random Dropout applied in the AuxLayer
# "AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst" - On ODL framework, Random Dropout applied in the first layer and all the features (base + auxiliary) are passed to the first layer
model_to_run = "AuxDrop_ODL"

# Values to change
n = 0.1
aux_feat_prob = 0.2
dropout_p = 0.3
max_num_hidden_layers = 6
qtd_neuron_per_hidden_layer = 50
n_classes = 2
aux_layer = 3
n_neuron_aux_layer = 100
batch_size = 1
b = 0.99
s = 0.2
use_cuda = False
number_of_experiments = 5

n_base_feat, n_aux_feat, X_base, X_aux, X_aux_new, aux_mask, Y, label = dataset(data_name, type = type, aux_feat_prob = aux_feat_prob, use_cuda = use_cuda, seed = 10)

print(X_base.shape)
print(X_aux.shape)
print(Y.shape)
print(aux_mask.shape)

print(X_base[0])
print(X_aux[0])
print(Y[0])
print(aux_mask[0])