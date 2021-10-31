import os
from typing import List
import xml.etree.ElementTree as et

from . import ColorDataset


class VRICDataset(ColorDataset):
    dataset_name = "VRIC"

    def __init__(
        self,
        data_dir,
        img_dir="train_images",
        img_list="vric_train.txt",
        data_transform=None,
        with_predictions=False,
        prediction_root: str = 'PREDICTION_ROOT',
    ):
        super().__init__(
            data_dir, with_predictions=with_predictions, prediction_root=prediction_root
        )
        self.img_dir = img_dir
        self.img_list = img_list
        self.data_transform = data_transform
        self.names = []
        with open(os.path.join(self.data_dir, self.img_list)) as reader:
            lines = reader.readlines()
            for line in lines:
                name = line.rstrip().split()[0]
                self.names.append(name)
