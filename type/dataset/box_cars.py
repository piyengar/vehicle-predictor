import os
from typing import List
import xml.etree.ElementTree as et

from . import ColorDataset


from scipy.io import loadmat
import pickle


class BoxCars116kDataset(ColorDataset):
    dataset_name = "BoxCars116k"

    def __init__(
        self,
        data_dir,
        img_dir="images",
        data_transform=None,
        with_predictions=False,
        prediction_root: str = 'PREDICTION_ROOT',
    ):
        super().__init__(
            data_dir, with_predictions=with_predictions, prediction_root=prediction_root
        )
        img_list = "dataset.pkl"
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.img_list = img_list
        self.data_transform = data_transform
        self.with_predictions = with_predictions
        self.names = []
        encoding = "latin-1"
        with open(os.path.join(data_dir, self.img_list), "rb") as f:
            dataset = pickle.load(f, encoding=encoding, fix_imports=True)
            for sample in dataset["samples"]:
                for instance in sample["instances"]:
                    self.names.append((instance["path"]))
