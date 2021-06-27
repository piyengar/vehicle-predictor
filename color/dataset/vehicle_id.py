import os
from typing import List

from . import ColorDataset


class VehicleIDDataset(ColorDataset):
    dataset_name='VehicleID'
    
    # this color list is only for reference, 
    # convention is to convert all colors to common color scheme
    # color_master = [
    #     'black',
    #     'blue',
    #     'gray',
    #     'red',
    #     'sliver',
    #     'white',
    #     'yellow',
    # ]

    @staticmethod
    def to_common_color(dataset_color):
        '''
        Convert dataset specific color code to common color code.
        Return the same code by default
        '''
        _common_color_mapping = [
            9,
            5,
            3,
            4,
            3,
            6,
            0
        ]
        return _common_color_mapping[dataset_color]

    def __init__(self, data_dir, 
                 data_transform = None, stage:str = None, prediction_file=None,
                 allowed_color_list:List[str] = None
                 ):
        self.img_dir = 'image'
        if stage == self.STAGE_TRAIN:
            name_file = 'train_test_split/train_list.txt'
            prediction_file = 'attribute/color_attr.txt'
        elif stage == self.STAGE_TEST:
            name_file = 'train_test_split/train_list.txt'
            prediction_file = 'attribute/color_attr.txt'
        else: # predict
            name_file = 'train_test_split/train_list.txt'
        self.name_file = name_file
        super().__init__(data_dir, stage=stage, prediction_file=prediction_file)
        self.data_transform = data_transform
        self.allowed_color_list = allowed_color_list
        self.names = []
        if self.stage in [self.STAGE_TRAIN, self.STAGE_TEST]:
            # Only small subset of data has color labels
            # print('loading color attributes')
            self.color_attr = {} # vid -> color_id
            self.colors = []
            with open(os.path.join(self.data_dir,self.prediction_file)) as reader:
                lines = reader.readlines()
                for line in lines:
                    vid, color = line.rstrip().split()
                    # print(f'Found name, color = {vid, color}')
                    self.color_attr[vid] = color

        with open(os.path.join(self.data_dir,self.name_file)) as reader:
            lines = reader.readlines()
            if self.stage in [self.STAGE_TRAIN, self.STAGE_TEST]:
                for line in lines:
                    name, vid = line.rstrip().split()
                    if vid in self.color_attr:
                        self.names.append(f'{name}.jpg')
                        self.colors.append(self.to_common_color(int(self.color_attr[vid])))
                    # else:
                    #     print(f'could not find name - {name} in color _attr')
            else:
                for line in lines:
                    name = line.rstrip().split()[0]
                    self.names.append(f'{name}.jpg')
        self.names, self.colors = self.filter_by_colors(allowed_color_list)
        if allowed_color_list != None:
            # print('remapping color list to allowed list')
            self._remap_colors(allowed_color_list)

    def _load_predictions(self):
        # only for predict
        if self.stage == self.STAGE_PREDICT:
            super()._load_predictions()
