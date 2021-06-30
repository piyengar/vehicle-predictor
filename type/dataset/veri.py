import os
from typing import List
import xml.etree.ElementTree as et 

from . import ColorDataset


class VeriDataset(ColorDataset):
    dataset_name='Veri'
    # this color list is only for reference, 
    # convention is to convert all colors to common color scheme
    # color_master = [
    #     'yellow',       #0
    #     'orange',       #1
    #     'green',        #2
    #     'gray',         #3
    #     'red',          #4
    #     'blue',         #5
    #     'white',        #6
    #     'golden',       #7
    #     'brown',        #8
    #     'black',        #9
    # ]

    @staticmethod
    def to_common_color(dataset_color):
        '''
        Convert dataset specific color code to common color code.
        Return the same code by default
        '''
        return dataset_color

    def __init__(self, data_dir, data_transform = None, 
                 stage:str = None, prediction_file=None, 
                 allowed_color_list: List[str] = None
                ):
        if stage == self.STAGE_TRAIN:
            name_file = 'name_train.txt'
            prediction_file = 'train_label.xml'
            self.img_dir = 'image_train'
        elif stage == self.STAGE_TEST:
            name_file = 'name_test.txt'
            prediction_file = 'test_label.xml'
            self.img_dir = 'image_test'
        else: # predict
            name_file = 'name_train.txt'
            self.img_dir = 'image_train'

        super().__init__(data_dir, stage=stage, prediction_file=prediction_file)

        self.data_transform = data_transform
        self.allowed_color_list = allowed_color_list
        
        self.names = []

        reader = open(os.path.join(self.data_dir,name_file))
        lines = reader.readlines()
        for line in lines:
            self.names.append(line.rstrip())
        
        self.names, self.colors = self.filter_by_colors(allowed_color_list)
        if allowed_color_list != None:
            self._remap_colors(allowed_color_list)
            
    def _load_predictions(self):
        if self.stage == self.STAGE_PREDICT and self.prediction_file.endswith('.txt'):
            super()._load_predictions()
        else:
            color_col = "colorID" 
            self.colors = []
            label_file = os.path.join(self.data_dir,self.prediction_file)
            xtree = et.parse(label_file, parser=et.XMLParser(encoding="utf-8"))
            xroot = xtree.getroot()[0]
            for label in xroot:
                self.colors.append(self.to_common_color(int(label.attrib.get(color_col))-1))