import enum
import os
import time
from abc import ABC, abstractmethod
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import (accuracy, confusion_matrix, f1, precision,
                                     recall)

from framework.datamodule import BaseDataModule
from sklearn.metrics import ConfusionMatrixDisplay

from .datasets import Datasets
from .prediction_writer import PredictionWriter


class Command(Enum):
    train_stats = auto(),
    test_stats = auto(),
    tune = auto(),
    train = auto(),
    predict = auto(),
    evaluate = auto(),
    
    def __str__(self):
        return self.name
    
    @staticmethod
    def from_string(s):
        try:
            return Command[s]
        except KeyError:
            raise ValueError()
        
        
class BaseExperiment(ABC):
    
    def __init__(
        self,
        class_names: List[str],
        model_arch: str = "resnet18",
        learning_rate: float = 0.001,
        lr_step=1,
        lr_step_factor=0.9,
        data_dir: str = "dataset",
        batch_size: int = 32,
        img_size: int = 224,
        train_split=0.7,
        num_workers=1,
        train_dataset: Datasets = Datasets.VEHICLE_ID,
        test_dataset: Datasets = Datasets.VEHICLE_ID,
        early_stop_patience: int = 3,
        early_stop_delta: int = 0.001,
        gpus: Optional[int] = None,
        is_dev_run: bool = False,
        max_epochs: int = 10,
        model_checkpoint_path: str = None,
        prediction_file_path: str = None,
        prediction_root: str = 'predictions',
        checkpoints_root: str = 'checkpoints',
        **kwargs,
    ) -> None:
        """ The base abstract experiment class that can be extended in experiments.

        Args:
            class_names (List[str]): The list of classes that should be used in the experiments, can be None
            model_arch (str, optional): The model architechture to use. Refer to `BaseModel.validArchs` for valid options. If the `BaseModel` is also extended in the experiment to add new architectures then that can be used as well. Defaults to "resnet18".
            learning_rate (float, optional): The learning rate. Defaults to 0.001.
            lr_step (int, optional): The number of epochs after which the learning rate is reduced. Defaults to 1.
            lr_step_factor (float, optional): The factor to reduce the learning rate by. Defaults to 0.9.
            data_dir (str, optional): The root directory where the datasets exist. Defaults to "dataset".
            batch_size (int, optional): The batch size for training the model. Defaults to 32.
            img_size (int, optional): The height and width to transform the images to before passing it to the model. Defaults to 224.
            train_split (float, optional): The fraction of sample to use for training the model. the rest will be used for validation. Defaults to 0.7.
            num_workers (int, optional): The number of workers for the dataloader. Defaults to 1.
            train_dataset (Datasets, optional): The dataset to train the model on. Defaults to Datasets.VEHICLE_ID.
            test_dataset (Datasets, optional): The dataset to test/evaluate the model on. Defaults to Datasets.VEHICLE_ID.
            early_stop_patience (int, optional): The number of epochs to wait when validation accuracy stops increasing. Defaults to 3.
            early_stop_delta (int, optional): The minimum val accuracy difference to track for early stopping. Defaults to 0.001.
            gpus (Optional[int], optional): The number of gpus to use, use -1 to use all available gpus, None for cpu. Defaults to None.
            is_dev_run (bool, optional): Run training in dev mode. This only runs a minimum required epochs to test code executes correctly. Defaults to False.
            max_epochs (int, optional): The max number of epochs to train the model. Defaults to 10.
            model_checkpoint_path (str, optional): The location of the model checkpoint file to run predictions on. Defaults to None.
            prediction_file_path (str, optional): The location to store the predictions or read prediction from to calculate metrics. Defaults to None.
            prediction_root (str, optional): The default location where the predictions should be stored. Defaults to 'predictions'.
            checkpoints_root (str, optional): The root folder where the training logs and checkpoints are stored. Defaults to 'checkpoints'.
        """
        self.class_names = class_names
        self.model_arch = model_arch
        self.learning_rate = learning_rate
        self.lr_step = lr_step
        self.lr_step_factor = lr_step_factor
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_split = train_split
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        self.gpus = gpus if torch.cuda.is_available() else None
        self.is_dev_run = is_dev_run
        self.max_epochs = max_epochs
        self.model_checkpoint_path = model_checkpoint_path
        self.prediction_file_path = prediction_file_path
        self.prediction_root = prediction_root
        self.checkpoints_root = checkpoints_root
        
    
    @abstractmethod
    def get_eval_data_module(self) -> BaseDataModule:
        """Returns a datamodule that loads the train dataset

        Returns:
            BaseDataModule: An instance of the BaseDataModule class. 
        """
        pass
    
    @abstractmethod
    def get_train_data_module(self) -> BaseDataModule:
        """Returns a datamodule that loads the test dataset

        Returns:
            BaseDataModule: An instance of the BaseDataModule class.
        """
        pass
    
    def train(self) -> str:
        """ Runs the training script based on the configuration params provided to the experiment class

        Returns:
            str: Returns the best model checkpoint path
        """
        # init model
        model = self.get_model()
        # init datamodule
        dm = self.get_train_data_module()
        # callbacks
        model_checkpoint_cb = ModelCheckpoint(monitor="val_acc", mode="max")
        callbacks = [
            EarlyStopping(
                "val_acc",
                mode="max",
                patience=self.early_stop_patience,
                verbose=True,
                min_delta=self.early_stop_delta,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            model_checkpoint_cb,
        ]
        # Initialize a trainer
        trainer = pl.Trainer(
            gpus=self.gpus,
            fast_dev_run=self.is_dev_run,
            max_epochs=self.max_epochs,
            progress_bar_refresh_rate=20,
            default_root_dir=os.path.join(self.checkpoints_root, self.get_name()),
            callbacks=callbacks,
        )
        trainer.fit(model, datamodule=dm)
        print(
            "Training completed. Best model is :", model_checkpoint_cb.best_model_path
        )
        self.model_checkpoint_path = model_checkpoint_cb.best_model_path
        return model_checkpoint_cb.best_model_path

    
    @abstractmethod
    def get_eval_trainer(self, predict_callback: PredictionWriter) -> pl.Trainer:
        """Get the trainer that is used for evaluating on the test dataset

        Args:
            predict_callback (PredictionWriter): The prediction writer class configured for the prediction location

        Returns:
            pl.Trainer: The trainer instance for running the evaluations
        """
        # return pl.Trainer(
        #     gpus=gpus, progress_bar_refresh_rate=20, callbacks=[predict_callback]
        # )
        pass
    
    @abstractmethod
    def get_model(self) -> pl.LightningModule:
        """The model configuration happens here in this method. Has to be implemented in the extended experiment class

        Returns:
            pl.LightningModule: The module holding the mdel
        """
        pass
    
    def get_model_from_checkpoint(self, model_checkpoint_file: str) -> pl.LightningModule:
        return self.get_model().load_from_checkpoint(model_checkpoint_file)
    
    @abstractmethod
    def get_target_names(self) -> List[str]:
        """Returns a list of class labels

        Returns:
            List[str]: List of class label strings
        """
        pass
    
    def get_prediction_root(self) -> str:
        return self.prediction_root
    
    def get_checkpoint_root(self) -> str:
        return self.checkpoints_root
    
    def tune_learning_rate(self) -> float:
        """Run the learning rate finder algorithm

        Returns:
            float: The suggested learning rate
        """
        trainer = pl.Trainer(
            gpus=self.gpus, 
            max_epochs=20, progress_bar_refresh_rate=20, auto_lr_find=True
            )
        lr_finder = trainer.tune(self.get_model(),datamodule=self.get_train_data_module())['lr_find']
        fig = lr_finder.plot(suggest=True)
        fig.show()
        return lr_finder.suggestion()[0]
    
    def get_train_stats(self) -> Dict[str, int]:
        """Get the dataset statistics of classes in the train set

        Returns:
            Dict[str, int]: A dict with class names as keys and number of samples as the value
        """
        return self.get_train_data_module().get_train_stats()
        
    def get_test_stats(self) -> Dict[str, int]:
        """Get the dataset statistics of classes in the test set


        Returns:
            Dict[str, int]: A dict with class names as keys and number of samples as the value
        """
        return self.get_eval_data_module().get_test_stats()
        
    
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
    
    def predict_and_persist(self,
    ) -> str: 
        """Run predictions on the specified test_dataset and store the results

        Returns:
            str: The location where the predictions are stored
        """
        model_checkpoint_path = self.model_checkpoint_path
        test_dataset = self.test_dataset
        batch_size = self.batch_size
        self.prediction_file_path = self.prediction_file_path or self.get_prediction_path(test_dataset, model_checkpoint_path)
        prediction_writer = PredictionWriter(
            write_interval="batch",
            out_file_name=self.prediction_file_path,
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
        print('Predictions stored at : ', self.prediction_file_path)
        return self.prediction_file_path
    
    def evaluate_predictions(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the accuracy, precision, recall, f1 and confusion matrix for the provided prediction files

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The accuracy, precision, F1, and confusion matrix
        """
        prediction_file_path = self.prediction_file_path
        test_dataset = self.test_dataset

        # load dataset with ground truth
        dm_gt = self.get_eval_data_module()
        dm_gt.setup("test")
        # gt will be indexed based on the dataset
        print(f"Evaluating {test_dataset.name} using {prediction_file_path}")
        
        targets = dm_gt.get_test_targets()
        
        gt = np.array(targets)
        gt = torch.from_numpy(gt)
        predictions = np.loadtxt(prediction_file_path, dtype=int)
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
        
        pred_dir, pred_file = os.path.split(prediction_file_path)
        eval_results_file_name = os.path.splitext(pred_file)[0]+f'_eval.txt'
        cm_results_file_name = os.path.splitext(pred_file)[0]+f'_cm.png'
        eval_results_file_path = os.path.join(pred_dir, eval_results_file_name)
        cm_results_file_path = os.path.join(pred_dir, cm_results_file_name)
        
        with open(eval_results_file_path, mode="w") as writer:
            writer.write(",".join(target_names))
            writer.write(f'\nAccuracy: {accuracy_val.item():.4f}')
            writer.write(f'\nPrecision: {precision_val.item():.4f}')
            writer.write(f'\nRecall: {recall_val.item():.4f}')
            writer.write(f'\nF1: {f1_val.item():.4f}')
            print('Saved metrics to ', eval_results_file_path)
        with open(cm_results_file_path, mode="wb") as writer:
            cmd = ConfusionMatrixDisplay(
                cm, display_labels=target_names
            )
            fig = cmd.plot(cmap=plt.cm.Blues, values_format=".2f", xticks_rotation='vertical').figure_
            fig.savefig(writer)
            # np.save(writer, cm.numpy())
            print("Saved confusion matrix to ", cm_results_file_path)
        return accuracy_val, precision_val, f1_val, recall_val, cm

    @staticmethod
    def add_parser_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds parser arg configuration for the experiment

        Args:
            parent_parser (ArgumentParser): The parent parser to which experiment args will be added

        Returns:
            ArgumentParser: The child arg parser with added arguments
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False,
                                formatter_class=ArgumentDefaultsHelpFormatter,
                                )
        parser.add_argument('commands', nargs='+', choices=list(Command), type=Command.from_string, default=Command.train, 
                        help="The commands that should be run. Multiple options can be provided separated by spaces. Run priority is :" + " > ".join(list(map(str, Command))))
        parser.add_argument("--model_arch", type=str, help='The architecture to be used for the model. Depends ont he experiment', default="resnet18"),
        parser.add_argument("--learning_rate", type=float, default=0.001, help='The learning rate for the model'),
        parser.add_argument("--lr_step", type=int, default=1, help='The number of epochs between each learning rate step'),
        parser.add_argument("--lr_step_factor", type=float, default=0.9, help='The factor with which the learning rate should be multiplied with at each step'),
        parser.add_argument("--data_dir", type=str, default="dataset", help='The root folder where the extracted datasets are stored'),
        parser.add_argument("--batch_size", type=int, default=32, help='Training batch size'),
        parser.add_argument("--img_size", type=int, default=224, help='The size of the image input into the model. Here we consider H=W'),
        parser.add_argument("--train_split", type=float, default=0.7, help='The proportion of data to be used for training, rest is used for validation. Between 0-1'),
        parser.add_argument("--num_workers", type=int, default=1, help='The number of worker threads to be used for loading data'),
        parser.add_argument(
            "--train_dataset",
            type=Datasets.from_string,
            choices=list(Datasets),
            default=Datasets.VEHICLE_ID,
            help='The dataset to be used for training'
        ),
        parser.add_argument(
            "--test_dataset",
            type=Datasets.from_string,
            choices=list(Datasets),
            default=Datasets.VEHICLE_ID,
            help='The dataset to be used for testing'
        ),
        parser.add_argument("--early_stop_patience", type=int, default=3, help='The number of epochs to wait for the observed metric to stop reducing/increasing before stopping training'),
        parser.add_argument("--early_stop_delta", type=int, default=0.001, help='The minimum observable difference to track'),
        parser.add_argument("--gpus", type=int, default=-1, help='The number of gpus to use. set to -1 to use all available'),
        parser.add_argument("--is_dev_run", type=bool, default=False, help='Run only one batch of data during training to test code if True'),
        parser.add_argument("--max_epochs", type=int, default=10, help="The max number of epochs to train"),
        parser.add_argument("--model_checkpoint_path", type=str, default=None, help='The model checkpoint file (.ckpt) to use during evaluation'),
        parser.add_argument("--prediction_file_path", type=str, default=None, help='The path were the predictions will be stored in. Should be a full file path.'),
        parser.add_argument("--prediction_root", type=str, default='predictions', help='The root folder where the predictions will be stored. This will only be used to generate a predictions file path if the --prediction_file_path arg is not specified'),
        parser.add_argument("--checkpoint_root", type=str, default='checkpoints', help='The root folder where the training logs and model checkpoints will be stored under'),
        return parser
    