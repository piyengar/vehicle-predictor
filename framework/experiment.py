import os
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

from .datasets import Datasets
from .prediction_writer import PredictionWriter

class BaseExperiment(ABC):
    prediction_root= 'predictions'
    checkpoints_root= 'checkpoints'
    
    @classmethod
    @abstractmethod
    def get_eval_data_module(self) -> pl.LightningDataModule:
        pass
    
    @classmethod
    @abstractmethod
    def get_train_data_module(self) -> pl.LightningDataModule:
        pass
    
    @classmethod
    @abstractmethod
    def train(self):
        pass
    
    @classmethod
    @abstractmethod
    def get_eval_trainer(self, predict_callback: PredictionWriter):
        # return pl.Trainer(
        #     gpus=gpus, progress_bar_refresh_rate=20, callbacks=[predict_callback]
        # )
        pass
    
    @classmethod
    @abstractmethod
    def get_model(self):
        pass
    
    @classmethod
    @abstractmethod
    def get_model_from_checkpoint(self, model_checkpoint_file: str) -> pl.LightningModule:
        pass
    
    @classmethod
    @abstractmethod
    def get_target_names(self) -> pl.LightningModule:
        pass
    
    @classmethod
    @abstractmethod
    def get_name(self) -> str:
        """
        Returns a name for the experiment.
        Sub classes should ensure that the name does not have and spaces in it as it will be used in path-like scenarios. 
        """
        pass
    
    def get_prediction_path(self, test_dataset: Datasets, model_checkpoint_file: str) -> str:
        prediction_file_name = os.path.splitext(model_checkpoint_file)[0]+'.txt'
        return os.path.join(self.prediction_root, self.get_name(), f'{test_dataset.name}_{prediction_file_name}')
    
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
        model_checkpoint_file: str,
        test_dataset: Datasets,
        batch_size: int,
    ):
        prediction_writer = PredictionWriter(
            write_interval="batch",
            out_file_name=self.get_prediction_path(test_dataset, model_checkpoint_file),
        )
        # datamodule
        dm = self.get_eval_data_module()
        # init model from checkpoint
        model = self.get_model_from_checkpoint(model_checkpoint_file)
        # init trainer
        trainer = self.get_eval_trainer(prediction_writer)
        dm.setup("test")
        dl = DataLoader(dm.test_dataset, batch_size=batch_size)
        trainer.predict(model, dataloaders=dl)
    
    def evaluate_predictions(
        self,
        model_checkpoint_file: str,
        test_dataset: Datasets,
    ):

        prediction_out_file_path = self.get_prediction_path(test_dataset, model_checkpoint_file)
        # load dataset with ground truth
        dm_gt = self.get_eval_data_module()
        dm_gt.setup("test")
        # gt will be indexed based on the dataset
        print(f"{test_dataset.name} using {model_checkpoint_file}")
        targets = dm_gt.test_dataset.targets
        gt = np.array(targets)
        gt = torch.from_numpy(gt)
        predictions = np.loadtxt(prediction_out_file_path, dtype=int)
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

        # print confusion matrix
        cm = confusion_matrix(predictions, gt, num_classes, "true")
        cm = ConfusionMatrixDisplay(cm.numpy(), display_labels=target_names)
        cm.plot(
            cmap=plt.cm.Blues,
            values_format=".2f",
        )
        plt.show()

        return accuracy_val, precision_val, f1_val, recall_val

    
    