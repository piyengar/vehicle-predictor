import os
from typing import Dict, List, Optional

import pytorch_lightning as pl
from framework.experiment import BaseExperiment
from framework.model import BaseModel, valid_archs

from framework.prediction_writer import PredictionWriter

from .datamodule import ColorDataModule
from .dataset import ColorDataset

class ColorExperiment(BaseExperiment):

    def get_name(self):
        return "color"

    def get_model(self):
        return BaseModel(
            self.class_names,
            self.model_arch,
            learning_rate=self.learning_rate,
            lr_step=self.lr_step,
            lr_step_factor=self.lr_step_factor,
        )

    def get_train_data_module(self):
        return ColorDataModule(
            dataset_type=self.train_dataset,
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            img_size=self.img_size,
            train_split=self.train_split,
            allowed_target_names=self.class_names,
            num_workers=self.num_workers,
        )

    def get_eval_data_module(self):
        return ColorDataModule(
            dataset_type=self.test_dataset,
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            img_size=self.img_size,
            train_split=self.train_split,
            allowed_target_names=self.class_names,  
            num_workers=self.num_workers,
        )
        
    def get_target_names(self) -> List[str]:
        return self.class_names or ColorDataset.color_master
    
    def get_eval_trainer(self, predict_callback: PredictionWriter) -> pl.Trainer:
        return pl.Trainer(
            gpus=self.gpus, 
            progress_bar_refresh_rate=20, 
            callbacks=[predict_callback],
            default_root_dir=os.path.join(self.checkpoints_root, self.get_name()),
        )
        
