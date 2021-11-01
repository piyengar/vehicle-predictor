import os
from argparse import ArgumentParser
from typing import List, Optional

import pytorch_lightning as pl

# from pl_bolts.datasets import DummyDataset
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import transforms

from .datasets import Datasets


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_type: Datasets = Datasets.VEHICLE_ID,
        data_dir: str = "dataset",
        batch_size: int = 32,
        img_size: int = 224,
        train_split=0.7,
        allowed_target_names: List[str] = None,
        num_workers=1,
    ):
        super().__init__()
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.train_split = train_split
        # we normalize according to ImageNet stats
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize]
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.allowed_target_names = allowed_target_names

    def _get_dataset(
        self, dataset_name: Datasets = Datasets.VEHICLE_ID, stage: Optional[str] = None
    ):
        raise NotImplementedError("Please implement this method")

    def setup(self, stage: Optional[str] = None):
        
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_val_dataset = self._get_dataset(
                self.dataset_type, "train"
            )
            ds_len = len(self.train_val_dataset)
            train_len = int(ds_len * self.train_split)
            val_len = ds_len - train_len
            self.train_dataset, self.val_dataset = random_split(
                self.train_val_dataset, [train_len, val_len]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = self._get_dataset(self.dataset_type, "test")
        if stage == "predict" or stage is None:
            self.predict_dataset = self._get_dataset(
                self.dataset_type, "predict"
            )

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size,num_workers=self.num_workers,)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,)
    