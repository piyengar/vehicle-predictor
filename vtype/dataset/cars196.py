import os
from typing import List
import xml.etree.ElementTree as et

from . import TypeDataset


from scipy.io import loadmat


class Cars196Dataset(TypeDataset):
    dataset_name = "Cars196"

    def __init__(
        self,
        data_dir,
        img_dir="cars_train",
        img_list="cars_train_annos.mat",
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
        names_mat = loadmat(os.path.join(data_dir, self.img_list))
        self.names = [name[0] for name in names_mat["annotations"]["fname"][0]]
