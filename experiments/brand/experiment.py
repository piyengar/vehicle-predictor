import os
from typing import List, Optional
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pytorch_lightning as pl
from experiments.brand.datamodule import BrandDataModule
from framework import Datasets
from framework.datamodule import BaseDataModule
from framework.experiment import BaseExperiment
from framework.model import BaseModel, valid_archs
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    model_checkpoint,
)

from framework.prediction_writer import PredictionWriter


class BrandExperiment(BaseExperiment):
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
        model_checkpoint_file: str = None,
        **kwargs,
    ) -> None:
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
        self.gpus = gpus
        self.is_dev_run = is_dev_run
        self.max_epochs = max_epochs
        self.model_checkpoint_file = model_checkpoint_file

    def get_name(self):
        return "brand"

    def get_model(self):
        return BaseModel(
            self.class_names,
            self.model_arch,
            learning_rate=self.learning_rate,
            lr_step=self.lr_step,
            lr_step_factor=self.lr_step_factor,
        )

    def get_train_data_module(self):
        return BrandDataModule(
            dataset_type=self.train_dataset,
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            img_size=self.img_size,
            train_split=self.train_split,
            allowed_target_names=self.class_names,
            num_workers=self.num_workers,
        )

    def get_eval_data_module(self):
        return BrandDataModule(
            dataset_type=self.test_dataset,
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            img_size=self.img_size,
            train_split=self.train_split,
            allowed_target_names=self.class_names,  
            num_workers=self.num_workers,
        )
        
    def get_target_names(self) -> List[str]:
        return self.class_names
    
    def get_eval_trainer(self, predict_callback: PredictionWriter) -> pl.Trainer:
        return pl.Trainer(
            gpus=self.gpus, progress_bar_refresh_rate=20, callbacks=[predict_callback]
        )
        
    def get_model_from_checkpoint(self, model_checkpoint_file: str) -> pl.LightningModule:
        return self.get_model().load_from_checkpoint(model_checkpoint_file)

    def train(self) -> str:
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
        self.model_checkpoint_file = model_checkpoint_cb.best_model_path
        return model_checkpoint_cb.best_model_path

    @staticmethod
    def add_parser_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False,
                                formatter_class=ArgumentDefaultsHelpFormatter,
                                )
        parser.add_argument("--model_arch", type=str),
        parser.add_argument("--learning_rate", type=float, default=0.001),
        parser.add_argument("--lr_step", type=int, default=1),
        parser.add_argument("--lr_step_factor", type=float, default=0.9),
        parser.add_argument("--data_dir", type=str, default="dataset"),
        parser.add_argument("--batch_size", type=int, default=32),
        parser.add_argument("--img_size", type=int, default=224),
        parser.add_argument("--train_split", type=float, default=0.7),
        parser.add_argument("--num_workers", type=int, default=1),
        parser.add_argument(
            "--train_dataset",
            type=Datasets.from_string,
            choices=list(Datasets),
            default=Datasets.VEHICLE_ID,
        ),
        parser.add_argument(
            "--test_dataset",
            type=Datasets.from_string,
            choices=list(Datasets),
            default=Datasets.VEHICLE_ID,
        ),
        parser.add_argument("--early_stop_patience", type=int, default=3),
        parser.add_argument("--early_stop_delta", type=int, default=0.001),
        parser.add_argument("--gpus", type=int, default=0),
        parser.add_argument("--is_dev_run", type=bool, default=False),
        parser.add_argument("--max_epochs", type=int, default=10),
        parser.add_argument("--model_checkpoint_file", type=str, default=None),
        return parser
    
    def predict_and_persist(self):
        return super().predict_and_persist(self.model_checkpoint_file, self.test_dataset, self.batch_size)
    
    def evaluate_predictions(self, predictions_file_path: str):
        return super().evaluate_predictions(predictions_file_path, self.test_dataset)
