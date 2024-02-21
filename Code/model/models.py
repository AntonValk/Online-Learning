import pandas as pd
import numpy as np
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
import torch

from utils.model_factory import instantiate
from metrics import CumulativeError, NormalizedCumulativeError, SmoothedCumulativeError, MovingWindowAccuracy
from modules import ODLSetSingleStageResidualNet, SetDecoder
    
    
class OnlineLearner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.backbone = instantiate(cfg.model.nn.backbone)
        self.init_metrics()
        self.loss = instantiate(cfg.model.loss)
        self.automatic_optimization = False
        
    def init_metrics(self):
        self.train_norm_err = NormalizedCumulativeError()
        self.train_exp_err = SmoothedCumulativeError()
        self.train_err = CumulativeError()

    def shared_forward(self, x): 
        prediction = self.backbone(x)   
        # self.backbone.update_alpha(prediction[1], x['Y'])
        return {'prediction': prediction}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['prediction']

    def training_step(self, batch, batch_idx):
        
        opt = self.optimizers()
        opt.zero_grad()
        batch_size=1
        net_output = self.shared_forward(batch)
        # print(batch)
        # print(net_output['prediction'][0])
        # print(net_output['prediction'][1])
        # print(net_output['prediction'][2])
        # exit()
        
        y_hat = net_output['prediction'][0]
        # loss = self.loss(y_hat, batch['Y']) 
        losses_per_layer = []
        for out in net_output['prediction'][1]:
            criterion = self.loss
            loss = criterion(
                out.view(batch_size, 2),
                batch['Y'].view(batch_size).long(),
            )
            losses_per_layer.append(loss)

        total_loss = torch.dot(self.backbone.alpha, torch.stack(losses_per_layer))
        self.manual_backward(total_loss)
        opt.step()
        
        self.backbone.update_alpha(losses_per_layer, batch['Y'])
        # self.manual_backward(loss)
        # opt.step()
        
        # self.log("train/loss", loss, on_step=False, on_epoch=True, 
        #          prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_err(y_hat, batch['Y'])
        self.log("train/cumulative_error", self.train_err.compute(), on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_norm_err(y_hat, batch['Y'])
        self.log("train/normalized_error", self.train_norm_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)

        self.train_exp_err(y_hat, batch['Y'])
        self.log("train/smoothed_error", self.train_exp_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        for i, a in enumerate(self.backbone.alpha_array[0]):
            self.log(f"train/alpha_{i}", a, on_step=True, on_epoch=False, prog_bar=True,
                    logger=True, batch_size=batch_size)
        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     pass
        
    # def test_step(self, batch, batch_idx):
    #     pass

    # def on_after_optimizer_step(self, batch, batch_idx):
    #     print("here")
    #     prediction = self.shared_forward(batch)
    #     self.backbone.update_alpha(prediction[1], batch['Y'])

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.optimizer, self.parameters())
        # scheduler = instantiate(self.cfg.model.scheduler, optimizer)
        # if scheduler is not None:
        #     optimizer = {"optimizer": optimizer, 
        #                  "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer


class AlphaExperiment(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.backbone = instantiate(cfg.model.nn.backbone)
        self.init_metrics()
        self.loss = instantiate(cfg.model.loss)
        self.automatic_optimization = False
        
    def init_metrics(self):
        self.train_norm_err = NormalizedCumulativeError()
        self.train_exp_err = SmoothedCumulativeError()
        self.train_err = CumulativeError()
        self.acc = [MovingWindowAccuracy() for _ in range(len(self.backbone.alpha))]

    def shared_forward(self, x): 
        prediction = self.backbone(x)   
        # self.backbone.update_alpha(prediction[1], x['Y'])
        return {'prediction': prediction}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['prediction']

    def training_step(self, batch, batch_idx):
        
        opt = self.optimizers()
        opt.zero_grad()
        batch_size=1
        net_output = self.shared_forward(batch)
        # print(batch)
        # print(net_output)
        # exit()
        
        y_hat = net_output['prediction'][0]
        # loss = self.loss(y_hat, batch['Y']) 
        losses_per_layer = []
        for out in net_output['prediction'][1]:
            criterion = self.loss
            loss = criterion(
                out.view(batch_size, 2),
                batch['Y'].view(batch_size).long(),
            )
            losses_per_layer.append(loss)

        total_loss = torch.dot(self.backbone.alpha, torch.stack(losses_per_layer))
        self.manual_backward(total_loss)
        opt.step()
        
        # self.backbone.update_alpha(losses_per_layer, batch['Y'])
        # self.manual_backward(loss)
        # opt.step()
        
        # self.log("train/loss", loss, on_step=False, on_epoch=True, 
        #          prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_err(y_hat, batch['Y'])
        self.log("train/cumulative_error", self.train_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_norm_err(y_hat, batch['Y'])
        self.log("train/normalized_error", self.train_norm_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)

        # self.train_exp_err(y_hat, batch['Y'])
        # self.log("train/smoothed_error", self.train_exp_err.compute(), on_step=True, on_epoch=False, 
        #          prog_bar=True, logger=True, batch_size=batch_size)

        for i, a in enumerate(self.backbone.alpha):
            self.log(f"train/alpha_{i}", a, on_step=True, on_epoch=False, prog_bar=False,
                    logger=True, batch_size=batch_size)
            self.acc[i](net_output['prediction'][1][i], batch['Y'])
            self.log(f"train/clf-{i}-accuracy", self.acc[i].compute(), on_step=True, on_epoch=False, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     pass
        
    # def test_step(self, batch, batch_idx):
    #     pass

    # def on_after_optimizer_step(self, batch, batch_idx):
    #     print("here")
    #     prediction = self.shared_forward(batch)
    #     self.backbone.update_alpha(prediction[1], batch['Y'])

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.optimizer, self.parameters())
        # scheduler = instantiate(self.cfg.model.scheduler, optimizer)
        # if scheduler is not None:
        #     optimizer = {"optimizer": optimizer, 
        #                  "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer


class VariableAlphaExperiment(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.backbone = instantiate(cfg.model.nn.backbone)
        self.init_metrics()
        self.loss = instantiate(cfg.model.loss)
        self.automatic_optimization = False
        
    def init_metrics(self):
        self.train_norm_err = NormalizedCumulativeError()
        self.train_exp_err = SmoothedCumulativeError()
        self.train_err = CumulativeError()
        self.acc = [MovingWindowAccuracy() for _ in range(len(self.backbone.alpha))]

    def shared_forward(self, x): 
        prediction = self.backbone(x)   
        # self.backbone.update_alpha(prediction[1], x['Y'])
        return {'prediction': prediction}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['prediction']

    def training_step(self, batch, batch_idx):
        
        opt = self.optimizers()
        opt.zero_grad()
        batch_size=1
        net_output = self.shared_forward(batch)
        # print(batch)
        # print(net_output)
        # exit()
        
        y_hat = net_output['prediction'][0]
        # loss = self.loss(y_hat, batch['Y']) 
        losses_per_layer = []
        for out in net_output['prediction'][1]:
            criterion = self.loss
            loss = criterion(
                out.view(batch_size, 2),
                batch['Y'].view(batch_size).long(),
            )
            losses_per_layer.append(loss)

        total_loss = torch.dot(self.backbone.alpha, torch.stack(losses_per_layer))
        self.manual_backward(total_loss)
        opt.step()
        
        self.backbone.update_alpha(losses_per_layer, batch['Y'])
        # self.manual_backward(loss)
        # opt.step()
        
        # self.log("train/loss", loss, on_step=False, on_epoch=True, 
        #          prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_err(y_hat, batch['Y'])
        self.log("train/cumulative_error", self.train_err.compute(), on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_norm_err(y_hat, batch['Y'])
        self.log("train/normalized_error", self.train_norm_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)

        self.train_exp_err(y_hat, batch['Y'])
        self.log("train/smoothed_error", self.train_exp_err.compute(), on_step=True, on_epoch=False, 
                 prog_bar=True, logger=True, batch_size=batch_size)

        for i, a in enumerate(self.backbone.alpha):
            self.log(f"train/alpha_{i}", a, on_step=True, on_epoch=False, prog_bar=True,
                    logger=True, batch_size=batch_size)
            self.acc[i](net_output['prediction'][1][i], batch['Y'])
            self.log(f"train/clf-{i}-accuracy", self.acc[i].compute(), on_step=True, on_epoch=False, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     pass
        
    # def test_step(self, batch, batch_idx):
    #     pass

    # def on_after_optimizer_step(self, batch, batch_idx):
    #     print("here")
    #     prediction = self.shared_forward(batch)
    #     self.backbone.update_alpha(prediction[1], batch['Y'])

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.optimizer, self.parameters())
        # scheduler = instantiate(self.cfg.model.scheduler, optimizer)
        # if scheduler is not None:
        #     optimizer = {"optimizer": optimizer, 
        #                  "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer