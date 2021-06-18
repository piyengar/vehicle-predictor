import os
import sys
import xml.etree.ElementTree as et
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from IPython.display import Image as ipImage
from ipywidgets import interact
# from pl_bolts.datasets import DummyDataset
from PIL import Image
from pytorch_lightning.callbacks import (BasePredictionWriter,
                                         LearningRateMonitor)
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchmetrics import Accuracy, ConfusionMatrix, Precision, Recall, F1
from torchmetrics.functional import accuracy, confusion_matrix, f1, precision
from torchvision import transforms
from torchvision.models import resnet18, squeezenet1_1, resnet50, resnet152

from .dataset import ColorDataset

class ColorModel(pl.LightningModule):

    def __init__(self, num_colors=11, model_arch:str='resnet18', learning_rate=1e-3, lr_step=1, lr_step_factor=0.9):
        super().__init__()
        self.save_hyperparameters()
        self.num_colors = num_colors
        self.learning_rate = learning_rate
        self.lr_step = lr_step
        self.lr_step_factor = lr_step_factor
        self.model = self._get_model(model_arch, True, num_colors)
        
        # Metrics
        avg_method = 'weighted'
        self.train_acc = Accuracy(num_classes=num_colors, average=avg_method)

        self.val_confusion = ConfusionMatrix(num_classes=num_colors, normalize='true', compute_on_step=False)
        self.val_acc = Accuracy(num_classes=num_colors, average=avg_method)
        self.val_prec = Precision(num_classes=num_colors, average=avg_method)
        self.val_rec = Recall(num_classes=num_colors, average=avg_method)
        self.val_f1 = F1(num_classes=num_colors, average=avg_method)

        self.test_confusion = ConfusionMatrix(num_classes=num_colors, normalize='true', compute_on_step=False)
        self.test_acc = Accuracy(num_classes=num_colors, average=avg_method)

        
    def _get_model(self, model_arch:str='resnet18', pretrained:bool = True, num_classes:int = 10):
        model = None
        if model_arch == 'resnet18':
          model = resnet18(pretrained=pretrained)
          # turn off backprop update for all the weights in the model
          for param in model.parameters():
              param.requires_grad = False
          model.fc = nn.Linear(in_features = model.fc.in_features, out_features=num_classes)
        elif model_arch == 'resnet50':
          model = resnet50(pretrained=pretrained)
          # turn off backprop update for all the weights in the model
          for param in model.parameters():
              param.requires_grad = False
          model.fc = nn.Linear(in_features = model.fc.in_features, out_features=num_classes)
        elif model_arch == 'resnet152':
          model = resnet152(pretrained=pretrained)
          # turn off backprop update for all the weights in the model
          for param in model.parameters():
              param.requires_grad = False
          model.fc = nn.Linear(in_features = model.fc.in_features, out_features=num_classes)
        elif model_arch == 'squeezenet':
          model = squeezenet1_1(pretrained=pretrained)
          # # turn off backprop update for all the weights in the model
          # for param in model.parameters():
          #     param.requires_grad = False
          # change the last Conv2D layer in case of squeezenet. there is no fc layer in the end.
          num_ftrs = 512
          model.classifier._modules["1"] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))

          # because in forward pass, there is a view function call which depends on the final output class size.
          model.num_classes = num_classes
        else:
          options = ','.join(['resnet18', 'resnet50', 'resnet152', 'squeezenet'])
          raise ValueError(f'Unsupported model_arch - {model_arch}, Supported values are {options}') 
        return model

    def _log_confusion(self, metric: ConfusionMatrix):
        cm = ConfusionMatrixDisplay(metric.compute().cpu().numpy(), display_labels=ColorDataset.color_master)
        fig = cm.plot(cmap=plt.cm.Blues, values_format='.2f').figure_
        self.logger.experiment.add_figure('val_confusion', fig, self.current_epoch)

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, *_ = batch
        y_hat = self(x)
        labels_hat = torch.argmax(y_hat, dim=1)
        return labels_hat

    def training_step(self, batch, batch_idx):
        x,_, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        
        # calculate metrics
        self.train_acc(y_hat.argmax(dim=1), y)
        self.log('train_acc', self.train_acc, prog_bar=True)

        return loss

    def training_epoch_end(self, training_step_outputs):
        pass

    def validation_step(self, batch, batch_idx):
        x,_, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # calculate metrics
        preds = F.softmax(y_hat, dim=1)
        self.val_acc(preds, y)
        self.val_prec(preds, y)
        self.val_rec(preds, y)
        self.val_f1(preds, y)
        self.val_confusion(preds, y)
        # log stats
        self.log_dict({'val_loss': loss, 'val_acc': self.val_acc}, prog_bar=True)

    def validation_epoch_end(self, validation_step_outputs):
        self._log_confusion(self.val_confusion)

    def test_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        # calculate metrics
        preds = F.softmax(y_hat, dim=1)
        self.test_acc(preds, y)
        self.test_confusion(preds, y)

        # log stats
        self.log_dict({'test_loss': loss, 'test_acc': self.test_acc}, prog_bar=True)

    def test_epoch_end(self, outputs):
        self.test_confusion.compute()
        self._log_confusion(self.test_confusion)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step, gamma=self.lr_step_factor)
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.lr_scheduler,
                'monitor': 'val_loss',
            }
        }