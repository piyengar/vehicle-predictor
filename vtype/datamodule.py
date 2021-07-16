import os
from typing import List, Optional

import pytorch_lightning as pl

# from pl_bolts.datasets import DummyDataset
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import transforms

from .dataset import (
    BoxCars116kDataset,
    Cars196Dataset,
    CompCarsDataset,
    VehicleIDDataset,
    VeriDataset,
    VRICDataset,
)


class TypeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "VRIC",
        data_dir: str = "dataset",
        batch_size: int = 32,
        img_size: int = 224,
        with_predictions=False,
        prediction_file=None,
        allowed_color_list: List[str] = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        # we normalize according to ImageNet stats
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        # if not with_predictions:
        self.transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize]
        )
        # else:
        #     self.transform = None
        self.batch_size = batch_size
        self.with_predictions = with_predictions
        self.prediction_file = prediction_file

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.allowed_color_list = allowed_color_list

    def _get_dataset_from_str(
        self, dataset_name: str = "VRIC", stage: Optional[str] = None
    ):
        if dataset_name == "VRIC":
            return VRICDataset(
                os.path.join(self.data_dir, "VRIC"),
                data_transform=self.transform,
                stage=stage,
                prediction_file=self.prediction_file,
            )
        elif dataset_name == "Cars196":
            return Cars196Dataset(
                os.path.join(self.data_dir, "Cars196"),
                data_transform=self.transform,
                stage=stage,
                prediction_file=self.prediction_file,
            )
        elif dataset_name == "BoxCars116k":
            return BoxCars116kDataset(
                os.path.join(self.data_dir, "BoxCars116k"),
                data_transform=self.transform,
                stage=stage,
                prediction_file=self.prediction_file,
            )
        elif dataset_name == "VehicleID":
            return VehicleIDDataset(
                os.path.join(self.data_dir, "VehicleID"),
                data_transform=self.transform,
                stage=stage,
                prediction_file=self.prediction_file,
                allowed_color_list=self.allowed_color_list,
            )
        elif dataset_name == "CompCars":
            return CompCarsDataset(
                os.path.join(self.data_dir, "CompCars", "sv_data"),
                data_transform=self.transform,
                stage=stage,
                prediction_file=self.prediction_file,
                allowed_type_list=self.allowed_color_list,
            )
        elif dataset_name == "Veri":
            return VeriDataset(
                os.path.join(self.data_dir, "VeRi_with_plate"),
                data_transform=self.transform,
                stage=stage,
                prediction_file=self.prediction_file,
                allowed_type_list=self.allowed_color_list,
            )
        elif dataset_name == "Combined":
            # import types
            # from collections import Counter
            # from operator import add
            # from functools import reduce

            ds = ConcatDataset(
                [
                    self._get_dataset_from_str("Veri", stage),
                    self._get_dataset_from_str("VehicleID", stage),
                    self._get_dataset_from_str("CompCars", stage),
                ]
            )
            # def get_color_counts(self):
            #     cc = map(lambda d: Counter(d.get_color_counts()), ds.datasets)
            #     cc = reduce(add, cc)
            #     return cc
            # ds.get_color_counts = types.MethodType(get_color_counts, ds)

            return ds
        else:
            raise ValueError("Dataset not supported")

    def setup(self, stage: Optional[str] = None):
        train_split = 0.7
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_val_dataset = self._get_dataset_from_str(
                self.dataset_name, "train"
            )
            ds_len = len(self.train_val_dataset)
            train_len = int(ds_len * train_split)
            val_len = ds_len - train_len
            self.train_dataset, self.val_dataset = random_split(
                self.train_val_dataset, [train_len, val_len]
            )

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = self._get_dataset_from_str(self.dataset_name, "test")
        if stage == "predict" or stage is None:
            self.predict_dataset = self._get_dataset_from_str(
                self.dataset_name, "predict"
            )

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
