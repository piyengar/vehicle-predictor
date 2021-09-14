import os
from typing import List

from scipy.io import loadmat

from . import ColorDataset


# CompCars
class CompCarsDataset(ColorDataset):
    dataset_name = "CompCars"

    # this color list is only for reference,
    # convention is to convert all colors to common color scheme
    # color_master = [
    #         'black',
    #         'white',
    #         'red',
    #         'yellow',
    #         'blue',
    #         'green',
    #         'purple',
    #         'brown',
    #         'golden',
    #         'silver'
    # ]
    @staticmethod
    def to_common_color(dataset_color):
        """
        Convert dataset specific color code to common color code.
        Return the same code by default
        """
        _common_color_mapping = [9, 6, 4, 0, 5, 2, 10, 8, 7, 3]
        return _common_color_mapping[dataset_color]

    def __init__(
        self,
        data_dir,
        data_transform=None,
        stage: str = None,
        prediction_file=None,
        allowed_color_list: List[str] = None,
    ):
        self.img_dir = "image"
        if stage == self.STAGE_TRAIN:
            name_file = "test_surveillance.txt"
            prediction_file = "color_list.mat"
        elif stage == self.STAGE_TEST:
            name_file = "train_surveillance.txt"
            prediction_file = "color_list.mat"
        else:  # predict
            name_file = "color_list.mat"
        self.name_file = name_file
        super().__init__(data_dir, stage=stage, prediction_file=prediction_file, allowed_color_list=allowed_color_list)
        self.img_dir = "image"
        self.data_transform = data_transform
        self.allowed_color_list = allowed_color_list
        self.names = []
        if self.stage in [self.STAGE_TRAIN, self.STAGE_TEST]:
            # Only small subset of data has color labels
            # print('loading color attributes')
            self.colors = []
            self.color_attr = {}
            colors_mat = loadmat(os.path.join(data_dir, self.prediction_file))
            for row in colors_mat["color_list"]:
                name, color = row[0][0], row[1][0][0]
                if color != -1:
                    self.color_attr[name] = color

        if self.stage in [self.STAGE_TRAIN, self.STAGE_TEST]:
            with open(os.path.join(self.data_dir, self.name_file)) as reader:
                lines = reader.readlines()
                for line in lines:
                    name = line.rstrip()
                    if name in self.color_attr:
                        self.names.append(name)
                        self.colors.append(
                            self.to_common_color(int(self.color_attr[name]))
                        )
                    # else:
                    #     print(f'could not find name - {name} in color _attr')
        else:
            colors_mat = loadmat(os.path.join(data_dir, self.name_file))
            for row in colors_mat["color_list"]:
                name, color = row[0][0], row[1][0][0]
                self.names.append(name)
        self.names, self.colors = self.filter_by_colors(allowed_color_list)
        if allowed_color_list != None:
            self._remap_colors(allowed_color_list)

    def _load_predictions(self):
        # only for predict
        if self.stage == self.STAGE_PREDICT:
            super()._load_predictions()
