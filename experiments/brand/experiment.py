import os
from typing import Dict, List, Optional

import pytorch_lightning as pl
from experiments.brand.datamodule import BrandDataModule
from experiments.brand.dataset.brand import Brand
from framework.experiment import BaseExperiment
from framework.model import BaseModel, valid_archs

from framework.prediction_writer import PredictionWriter


class BrandExperiment(BaseExperiment):
    
    def __init__(self, class_names: List[str], model_arch: str = "resnet18", learning_rate: float = 0.001, lr_step=1, lr_step_factor=0.9, data_dir: str = "dataset", batch_size: int = 32, img_size: int = 224, train_split=0.7, num_workers=1, train_dataset: Datasets = ..., test_dataset: Datasets = ..., early_stop_patience: int = 3, early_stop_delta: int = 0.001, gpus: Optional[int] = None, is_dev_run: bool = False, max_epochs: int = 10, model_checkpoint_path: str = None, prediction_file_path: str = None, prediction_root: str = 'predictions', checkpoints_root: str = 'checkpoints', **kwargs) -> None:
        if not class_names:
            class_names = list(map(lambda x: x.name, Brand))
        super().__init__(class_names, model_arch=model_arch, learning_rate=learning_rate, lr_step=lr_step, lr_step_factor=lr_step_factor, data_dir=data_dir, batch_size=batch_size, img_size=img_size, train_split=train_split, num_workers=num_workers, train_dataset=train_dataset, test_dataset=test_dataset, early_stop_patience=early_stop_patience, early_stop_delta=early_stop_delta, gpus=gpus, is_dev_run=is_dev_run, max_epochs=max_epochs, model_checkpoint_path=model_checkpoint_path, prediction_file_path=prediction_file_path, prediction_root=prediction_root, checkpoints_root=checkpoints_root, **kwargs)

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
        dm = BrandDataModule(
            dataset_type=self.train_dataset,
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            img_size=self.img_size,
            train_split=self.train_split,
            allowed_target_names=self.class_names,
            num_workers=self.num_workers,
        )
        return dm

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
        return self.class_names or list(map(str, list(Brand)))
    
    def get_eval_trainer(self, predict_callback: PredictionWriter) -> pl.Trainer:
        return pl.Trainer(
            gpus=self.gpus, progress_bar_refresh_rate=20, callbacks=[predict_callback]
        )
        
