import os
from typing import Any, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# Superclass for Datasets
class ColorDataset(Dataset):
    # These are all the possible colors, dataset specific colors should be defined in the respective sub class
    color_master = [
        "yellow",  # 0
        "orange",  # 1
        "green",  # 2
        "gray",  # 3
        "red",  # 4
        "blue",  # 5
        "white",  # 6
        "golden",  # 7
        "brown",  # 8
        "black",  # 9
        "purple",  # 10
    ]
    STAGE_TRAIN = "train"
    STAGE_TEST = "test"
    STAGE_PREDICT = "predict"

    def __init__(
        self,
        data_dir,
        prediction_file=None,
        stage: Optional[str] = None,
        allowed_color_list: List[str] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.stage = stage
        self.prediction_file = prediction_file
        self.allowed_color_list = allowed_color_list

        # names will be loaded in the implementation
        self.names = []
        self.colors = None
        if self.prediction_file != None:
            self._load_predictions()

    def _load_predictions(self):
        self.colors = []
        with open(self.prediction_file) as reader:
            lines = reader.readlines()
            for line in lines:
                color = line.rstrip().split()[0]
                self.colors.append(int(color))

    @staticmethod
    def to_common_color(dataset_color):
        """
        Convert dataset specific color code to common color code.
        Return the same code by default
        """
        return dataset_color

    def get_color_counts(self):
        if self.colors != None:
            color_list = (
                self.allowed_color_list
                if self.allowed_color_list != None
                else self.color_master
            )
            c, counts = np.unique(np.array(self.colors), return_counts=True)
            return [
                (color, color_list[color], count) for color, count in zip(c, counts)
            ]

    def filter_by_colors(self, color_list: List[str] = None):
        """
        Returns a filtered list of names, colors
        """
        if color_list == None or color_list == [] or self.colors == None:
            return self.names, self.colors

        def fil_fun(entry):
            name, color = entry
            return self.color_master[color] in color_list

        # return a filtered list of names, colors
        filtered_tuples = filter(fil_fun, zip(self.names, self.colors))
        names = []
        colors = []
        for tup in filtered_tuples:
            names.append(tup[0])
            colors.append(tup[1])
        return names, colors

    def _remap_colors(self, custom_color_list):
        if not custom_color_list:
            self.colors = list(map(lambda b: b.value, self.colors))
            return
        mapping = []
        for color in self.color_master:
            if color in custom_color_list:
                mapping.append(custom_color_list.index(color))
            else:
                mapping.append(-1)

        def map_fun(color):
            return mapping[color]

        # print('old colors', self.colors)
        if self.colors != None:
            self.colors = list(map(map_fun, self.colors))
        # print('new colors', self.colors)

    def __getitem__(self, index):
        # For normalize
        name = self.names[index]
        img = Image.open(os.path.join(self.data_dir, self.img_dir, name))
        # .convert('RGB') # convert gray to rgb

        if self.data_transform != None:
            img = self.data_transform(img)
        if self.stage == self.STAGE_PREDICT:
            # color are not available
            return img, os.path.join(self.data_dir, self.img_dir, name)
        color = self.colors[index]
        return img, os.path.join(self.data_dir, self.img_dir, name), color

    def __len__(self):
        return len(self.names)
