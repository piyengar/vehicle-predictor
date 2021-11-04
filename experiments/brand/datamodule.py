import os
from typing import Dict, List, Optional

import pytorch_lightning as pl

# from pl_bolts.datasets import DummyDataset
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import transforms
from experiments.brand.dataset.brand import Brand, BrandDataset

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
        dataset: Datasets = Datasets.VEHICLE_ID,
        stage: Optional[str] = None,
    ):
        allowed_brand_list = list(map(lambda x: Brand[x], self.allowed_target_names))
        if dataset == Datasets.VEHICLE_ID:
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
        elif dataset == Datasets.CARS196:
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
        if self.dataset_type == Datasets.VEHICLE_ID:
            return self.test_dataset.brands
        elif self.dataset_type == Datasets.CARS196:
            return self.test_dataset.brands
        else:
            raise ValueError("Dataset not supported")
            
            
    def get_train_stats(self) -> Dict[str, int]:
        self.setup('fit')
        return self._get_dataset_stats(self.train_dataset)
    
    def get_test_stats(self) -> Dict[str, int]:
        self.setup('test')
        return self._get_dataset_stats(self.test_dataset)
    
    def _get_dataset_stats(self, dataset: BrandDataset):
        if self.dataset_type in [Datasets.VEHICLE_ID]:
            counts = dataset.get_brand_counts()
            counts = sorted(counts, key= lambda ct: ct[1], reverse=True)
            data = {}
            for ct in counts:
                data[ct[1]] = int(ct[2])
            return data
        else:
            raise ValueError("Dataset not supported")
            
