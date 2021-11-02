import os
from typing import List, Optional

import pytorch_lightning as pl

# from pl_bolts.datasets import DummyDataset
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import transforms
from experiments.brand.dataset.brand import Brand

from framework.datamodule import BaseDataModule
from framework.datasets import Datasets

from .dataset import (
    # BoxCars116kDataset,
    Cars196Dataset,
    # CompCarsDataset,
    VehicleIDDataset,
    # VeriDataset,
    # VRICDataset,
)


class BrandDataModule(BaseDataModule):
    def _get_dataset(
        self,
        dataset_name: Datasets = Datasets.VEHICLE_ID,
        stage: Optional[str] = None,
    ):
        allowed_brand_list = map(lambda x: Brand[x], self.allowed_target_names)
        if dataset_name == Datasets.VEHICLE_ID:
            return VehicleIDDataset(
                os.path.join(self.data_dir, "VehicleID"),
                data_transform=self.transform,
                stage=stage,
                allowed_brand_list=allowed_brand_list,
            )
        # elif dataset_name == Datasets.VRIC:
        #     return VRICDataset(
        #         os.path.join(self.data_dir, "VRIC"),
        #         data_transform=self.transform,
        #         with_predictions=self.with_predictions,
        #     )
        elif dataset_name == Datasets.CARS196:
            return Cars196Dataset(
                os.path.join(self.data_dir, "Cars196"),
                data_transform=self.transform,
                stage=stage,
                allowed_brand_list=allowed_brand_list,
            )
        # elif dataset_name == Datasets.BOXCARS116K:
        #     return BoxCars116kDataset(
        #         os.path.join(self.data_dir, "BoxCars116k"),
        #         data_transform=self.transform,
        #         with_predictions=self.with_predictions,
        #     )
        # elif dataset_name == Datasets.COMP_CARS:
        #     return CompCarsDataset(
        #         os.path.join(self.data_dir, "CompCars", "sv_data"),
        #         data_transform=self.transform,
        #         stage=stage,
        #         prediction_file=self.prediction_file,
        #         allowed_brand_list=allowed_brand_list,
        #     )
        # elif dataset_name == Datasets.VERI:
        #     return VeriDataset(
        #         os.path.join(self.data_dir, "VeRi_with_plate"),
        #         data_transform=self.transform,
        #         stage=stage,
        #         prediction_file=self.prediction_file,
        #         allowed_brand_list=allowed_brand_list,
        #     )
        # elif dataset_name == Datasets.COMBINED:
        #     # import types
        #     # from collections import Counter
        #     # from operator import add
        #     # from functools import reduce

        #     ds = ConcatDataset(
        #         [
        #             self._get_dataset(Datasets.VERI, stage),
        #             self._get_dataset(Datasets.VEHICLE_ID, stage),
        #             self._get_dataset(Datasets.COMP_CARS, stage),
        #         ]
        #     )
        #     # def get_brand_counts(self):
        #     #     cc = map(lambda d: Counter(d.get_brand_counts()), ds.datasets)
        #     #     cc = reduce(add, cc)
        #     #     return cc
        #     # ds.get_brand_counts = types.MethodType(get_brand_counts, ds)

        #     return ds
        else:
            raise ValueError("Dataset not supported")
        
    def get_test_targets(self):
        self.setup('test')
        return self.test_dataset.targets
