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
from AuxDrop import (
    AuxDrop_ODL_DirectedInAuxLayer_RandomOtherLayer,
    AuxDrop_ODL_RandomAllLayer,
)
from AuxDrop import (
    AuxDrop_ODL_RandomInAuxLayer,
    AuxDrop_ODL_RandomInFirstLayer_AllFeatToFirst,
)
from dataset import dataset
from joblib import Parallel, delayed
from modules.residual import SingleStageResidualNet

from torch.utils.tensorboard import SummaryWriter

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

# model_to_run = "AuxDrop_ODL"
# model_to_run = "AuxDrop_OGD"
model_to_run = "ResidualSingleStage"

# Values to change
n = 0.05
aux_feat_prob = 0.72
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

error_list = []
loss_list = []
# for ex in range(number_of_experiments):
def run_trial(ex):
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(log_dir=f"runs/{data_name}/MODEL-{model_to_run}-SEED-{ex}-LR-{str(n)}")
    trial_stats = 0
    print("Experiment number ", ex + 1)
    seed = ex

    # Please change the value of hyperparameters in the dataset.py file corresponding to the chose data name
    n_base_feat, n_aux_feat, X_base, X_aux, X_aux_new, aux_mask, Y, label = dataset(
        data_name, type=type, aux_feat_prob=aux_feat_prob, use_cuda=use_cuda, seed=seed
    )
    # Note: X_aux_new contains the auxiliary data with some data unavailable.
    # X_aux contains the auxiliary features with all the data (even the unavailable ones)

    model = None
    if model_to_run == "AuxDrop_ODL":
        if aux_layer == 1:
            model = AuxDrop_ODL_AuxLayer1stlayer(
                features_size=n_base_feat,
                max_num_hidden_layers=max_num_hidden_layers,
                qtd_neuron_per_hidden_layer=qtd_neuron_per_hidden_layer,
                n_classes=n_classes,
                aux_layer=aux_layer,
                n_neuron_aux_layer=n_neuron_aux_layer,
                batch_size=batch_size,
                b=b,
                n=n,
                s=s,
                dropout_p=dropout_p,
                n_aux_feat=n_aux_feat,
                use_cuda=use_cuda,
            )
        else:
            # Creating the Aux-Drop(ODL) Model
            model = AuxDrop_ODL(
                features_size=n_base_feat,
                max_num_hidden_layers=max_num_hidden_layers,
                qtd_neuron_per_hidden_layer=qtd_neuron_per_hidden_layer,
                n_classes=n_classes,
                aux_layer=aux_layer,
                n_neuron_aux_layer=n_neuron_aux_layer,
                batch_size=batch_size,
                b=b,
                n=n,
                s=s,
                dropout_p=dropout_p,
                n_aux_feat=n_aux_feat,
                use_cuda=use_cuda,
            )
    elif model_to_run == "AuxDrop_OGD":
        if data_name in ["ItalyPowerDemand", "SUSY", "HIGGS"]:
            print(
                "You need to make some changes in the code to support AuxDrop_OGD with ",
                data_name,
                " dataset",
            )
            exit()
        # Creating the Aux-Drop(OGD) use this - The position of AuxLayer cannot be 1 here
        if aux_layer == 1:
            print("Error: Please choose the aux layer position greater than 1")
            exit()
        else:
            model = AuxDrop_OGD(
                features_size=n_base_feat,
                max_num_hidden_layers=max_num_hidden_layers,
                qtd_neuron_per_hidden_layer=qtd_neuron_per_hidden_layer,
                n_classes=n_classes,
                aux_layer=aux_layer,
                n_neuron_aux_layer=n_neuron_aux_layer,
                batch_size=batch_size,
                n_aux_feat=n_aux_feat,
                n=n,
                dropout_p=dropout_p,
            )
    elif model_to_run == "ResidualSingleStage":
        model = SingleStageResidualNet(
                num_blocks_enc=2,
                num_layers_enc=2,
                layer_width_enc=100,
                num_blocks_stage=2, 
                num_layers_stage=2, 
                layer_width_stage=100,
                embedding_dim=0,
                embedding_num=0,
                embedding_size=0,
                size_in=n_base_feat + n_aux_feat,
                size_out=n_classes,
                dropout=dropout_p,
                lr=n,
            )
        

    # Run the model
    N = X_base.shape[0]
    cumulative_error_train = np.array([0])
    cumulative_error_test = np.array([0])
    exp_smoothing = 0.05
    prev_train = prev_test = 0
    for i in tqdm(range(N)):
        model.partial_fit(X_base[i].reshape(1, n_base_feat), X_aux_new[i].reshape(1, n_aux_feat), aux_mask[i].reshape(1, n_aux_feat), Y[i].reshape(1))
        pred = model.prediction[-1]
        cumulative_error_test += torch.argmax(pred).item() != Y[i]
        writer.add_scalar('test/cumulative_error', cumulative_error_test, i)
        # writer.add_scalar('test/exp_smooth_error', exp_smoothing * (torch.argmax(pred).item() != Y[i]) + (1 - exp_smoothing) * prev_test, i)
        # prev_test = exp_smoothing * (torch.argmax(pred).item() != Y[i]) + (1 - exp_smoothing) * prev_test
        writer.add_scalar('test/norm_error', cumulative_error_test/i, i)
        test_loss = model.loss_fn(pred, torch.tensor(Y[i], dtype=torch.long))
        writer.add_scalar('test/test loss', test_loss, i)
        
        with torch.no_grad():
            if hasattr(model, 'alpha_array'):
                pred = model.forward(X_base[i].reshape(1, n_base_feat), X_aux[i].reshape(1, n_aux_feat), aux_mask[i].reshape(1, n_aux_feat))
                pred = torch.sum(torch.mul(model.alpha.view(model.max_num_hidden_layers - 2, 1).repeat(1, model.batch_size).view(model.max_num_hidden_layers - 2, 
                                                                        model.batch_size, 1), pred), 0)
            else:
                pred = model.forward(X_base[i].reshape(1, n_base_feat), X_aux[i].reshape(1, n_aux_feat), aux_mask[i].reshape(1, n_aux_feat))

        cumulative_error_train += torch.argmax(pred).item() != Y[i]
        # cumulative_error_train += torch.argmax(model.prediction[i]).item() != Y[i]
        writer.add_scalar('train/cumulative_error', cumulative_error_train, i)
        # writer.add_scalar('train/exp_smooth_error', exp_smoothing * (torch.argmax(pred).item() != Y[i]) + (1 - exp_smoothing) * prev_train, i)
        # prev_train = exp_smoothing * (torch.argmax(model.prediction[-1]).item() != Y[i]) + (1 - exp_smoothing) * prev_train
        writer.add_scalar('train/norm_error', cumulative_error_train/i, i)
        writer.add_scalar('train/training_loss', model.loss_array[-1], i)
        if hasattr(model, 'alpha_array'):
            for j in range(len(model.alpha_array[-1])):
                writer.add_scalar(f'alphas/{str(j)}', model.alpha_array[-1][j], i)

    # Calculate error or loss
    if data_name == "ItalyPowerDemand":
        loss = np.mean(model.loss_array)
        # print("The loss in the ", data_name, " dataset is ", loss)
        # loss_list.append(loss)
        trial_stats = loss
    else:
        # prediction = []
        # print('len', len(model.prediction))
        # for i in model.prediction:
        #     prediction.append(torch.argmax(i).item())
        # error = len(prediction) - sum(prediction == label)
        # print("The error in the ", data_name, " dataset is ", error)
        # print("Cumulative error is", cumulative_error_test)
        # trial_stats = error
        trial_stats = cumulative_error_test
        # # logging
        # print(error)
        # print(cumulative_error_test)
    return trial_stats


result = Parallel(n_jobs=min(number_of_experiments, os.cpu_count()))(
    delayed(run_trial)(i) for i in range(number_of_experiments)
)

# result = run_trial(1)

if data_name == "ItalyPowerDemand":
    print(
        "The mean loss in the ",
        data_name,
        " dataset for ",
        number_of_experiments,
        " number of experiments is ",
        np.mean(result),
        " and the standard deviation is ",
        np.std(result),
    )
else:
    print(
        "The mean error in the ",
        data_name,
        " dataset for ",
        number_of_experiments,
        " number of experiments is ",
        np.mean(result),
        " and the standard deviation is ",
        np.std(result),
    )
