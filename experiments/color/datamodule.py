import os
from typing import Dict, List, Optional

import pytorch_lightning as pl

# from pl_bolts.datasets import DummyDataset
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import transforms

from framework.datamodule import BaseDataModule
from framework.datasets import Datasets

from .dataset import (
    ColorDataset,
    # BoxCars116kDataset,
    # Cars196Dataset,
    CompCarsDataset,
    VehicleIDDataset,
    VeriDataset,
    # VRICDataset,
)


class ColorDataModule(BaseDataModule):
    def _get_dataset(
        self,
        dataset: Datasets = Datasets.VEHICLE_ID,
        stage: Optional[str] = None,
    ):
        if dataset == Datasets.VEHICLE_ID:
            return VehicleIDDataset(
                os.path.join(self.data_dir, "VehicleID"),
                data_transform=self.transform,
                stage=stage,
                allowed_color_list=self.allowed_target_names,
            )
        elif dataset == Datasets.COMP_CARS:
            return CompCarsDataset(
                os.path.join(self.data_dir, "CompCars", "sv_data"),
                data_transform=self.transform,
                stage=stage,
                allowed_color_list=self.allowed_target_names,
            )
        elif dataset == Datasets.VERI:
            return VeriDataset(
                os.path.join(self.data_dir, "VeRi_with_plate"),
                data_transform=self.transform,
                stage=stage,
                allowed_color_list=self.allowed_target_names,
            )
        elif dataset == Datasets.COMBINED:
            # import types
            # from collections import Counter
            # from operator import add
            # from functools import reduce

            ds = ConcatDataset(
                [
                    self._get_dataset(Datasets.VERI, stage),
                    self._get_dataset(Datasets.VEHICLE_ID, stage),
                    self._get_dataset(Datasets.COMP_CARS, stage),
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

    def get_test_targets(self):
        self.setup("test")
        if self.dataset_type in [
            Datasets.VEHICLE_ID,
            Datasets.VERI,
            Datasets.COMP_CARS,
            Datasets.COMBINED,
        ]:
            return self.test_dataset.colors
        else:
            raise ValueError("Dataset not supported")

    def get_train_stats(self) -> Dict[str, int]:
        self.setup("fit")
        return self._get_dataset_stats(self.train_val_dataset)

    def get_test_stats(self) -> Dict[str, int]:
        self.setup("test")
        return self._get_dataset_stats(self.test_dataset)

    def _get_dataset_stats(self, dataset: ColorDataset):
        if self.dataset_type not in [
            Datasets.VEHICLE_ID,
            Datasets.VERI,
            Datasets.COMP_CARS,
            Datasets.COMBINED,
        ]:
            raise ValueError("Dataset not supported")

        counts = dataset.get_color_counts()
        counts = sorted(counts, key=lambda ct: ct[1], reverse=True)
        return {ct[1]: int(ct[2]) for ct in counts}
