import os
import time
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchmetrics.functional import (accuracy, confusion_matrix, f1, precision,
                                     recall)
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

from framework.datamodule import BaseDataModule

from .datasets import Datasets
from .prediction_writer import PredictionWriter

class BaseExperiment(ABC):
    prediction_root= 'predictions'
    checkpoints_root= 'checkpoints'
    
    @abstractmethod
    def get_eval_data_module(self) -> BaseDataModule:
        pass
    
    @abstractmethod
    def get_train_data_module(self) -> BaseDataModule:
        pass
    
    @abstractmethod
    def train(self) -> str:
        pass
    
    @abstractmethod
    def get_eval_trainer(self, predict_callback: PredictionWriter) -> pl.Trainer:
        # return pl.Trainer(
        #     gpus=gpus, progress_bar_refresh_rate=20, callbacks=[predict_callback]
        # )
        pass
    
    @abstractmethod
    def get_model(self):
        pass
    
    @abstractmethod
    def get_model_from_checkpoint(self, model_checkpoint_file: str) -> pl.LightningModule:
        pass
    
    @abstractmethod
    def get_target_names(self) -> List[str]:
        pass
    
    def get_prediction_root(self) -> str:
        return "predictions"
    
    def get_checkpoint_root(self) -> str:
        return "checkpoints"
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Returns a name for the experiment.
        Sub classes should ensure that the name does not have any spaces in it as it will be used in path-like scenarios. 
        """
        pass
    
    def get_prediction_path(self, test_dataset: Datasets, model_checkpoint_path: str) -> str:
        _, model_checkpoint_file = os.path.split(model_checkpoint_path)
        prediction_file_name = os.path.splitext(model_checkpoint_file)[0]+f'_{time.time()}.txt'
        return os.path.join(self.get_prediction_root(), self.get_name(), f'{test_dataset.name}_{prediction_file_name}')
    
    def get_model_checkpoint_path(self, model_checkpoint_file: str) -> str:
        """
        We assume that the model exists within the checkpoints/<experiment_name> folder and return the full path to the checkpoint

        Args:
            model_checkpoint_file (str): The file name of the model checkpoint

        Returns:
            str: The path to the model checkpoint as per convention
        """
        return os.path.join(self.checkpoints_root, self.get_name(), model_checkpoint_file)
    
    def predict_and_persist(self,
        model_checkpoint_path: str,
        test_dataset: Datasets,
        batch_size: int,
    ):
        predictions_file_path = self.get_prediction_path(test_dataset, model_checkpoint_path)
        prediction_writer = PredictionWriter(
            write_interval="batch",
            out_file_name=predictions_file_path,
        )
        # datamodule
        dm = self.get_eval_data_module()
        # init model from checkpoint
        model = self.get_model_from_checkpoint(model_checkpoint_path)
        # init trainer
        trainer = self.get_eval_trainer(prediction_writer)
        dm.setup("test")
        dl = DataLoader(dm.test_dataset, batch_size=batch_size)
        trainer.predict(model, dataloaders=dl)
        return predictions_file_path
    
    def evaluate_predictions(
        self,
        predictions_file_path: str,
        test_dataset: Datasets,
    ):

        # load dataset with ground truth
        dm_gt = self.get_eval_data_module()
        dm_gt.setup("test")
        # gt will be indexed based on the dataset
        print(f"Evaluating {test_dataset.name} using {predictions_file_path}")
        
        # TODO need to delegate access to the datamodule
        targets = dm_gt.get_test_targets()
        
        gt = np.array(targets)
        gt = torch.from_numpy(gt)
        predictions = np.loadtxt(predictions_file_path, dtype=int)
        predictions = torch.from_numpy(predictions)
        target_names = self.get_target_names()
        avg_met = "weighted"
        num_classes = len(target_names)
        accuracy_val = accuracy(predictions, gt, average=avg_met, num_classes=num_classes)
        precision_val = precision(predictions, gt, average=avg_met, num_classes=num_classes)
        f1_val = f1(predictions, gt, average=avg_met, num_classes=num_classes)
        recall_val = recall(predictions, gt, average=avg_met, num_classes=num_classes)
        print(f"The accuracy is {accuracy_val}")
        print(f"The precision is {precision_val}")
        print(f"The f1 is {f1_val}")
        print(f"The recall is {recall_val}")

        # confusion matrix
        cm = confusion_matrix(predictions, gt, num_classes, "true")
        
        pred_dir, pred_file = os.path.split(predictions_file_path)
        eval_results_file_name = os.path.splitext(pred_file)[0]+f'_eval.txt'
        cm_results_file_name = os.path.splitext(pred_file)[0]+f'_cm.npy'
        eval_results_file_path = os.path.join(pred_dir, eval_results_file_name)
        cm_results_file_path = os.path.join(pred_dir, cm_results_file_name)
        
        with open(eval_results_file_path, mode="w") as writer:
            writer.write(target_names)
            writer.write(accuracy_val)
            writer.write(precision_val)
            writer.write(f1_val)
            writer.write(recall_val)
        with open(cm_results_file_path, mode="wb") as writer:
            np.save(writer, cm.numpy())
        # plt.savefig())

        return accuracy_val, precision_val, f1_val, recall_val, cm

    
    