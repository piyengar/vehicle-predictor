import os
from typing import List
import xml.etree.ElementTree as et 

from . import TypeDataset, Type


class VeriDataset(TypeDataset):
    dataset_name='Veri'

    # dataset types
    _dataset_type_mapping = [
        Type.SEDAN,         # sedan
        Type.SUV,           # suv
        Type.HATCHBACK,     # van
        Type.HATCHBACK,     # hatchback
        Type.SUV,           # mpv
        Type.PICKUP,        # pickup
        Type.BUS,           # bus
        Type.TRUCK,         # truck
        Type.SUV,           # estate
    ]

    @staticmethod
    def to_common_type(dataset_type):
        '''
        Convert dataset specific color code to common color code.
        Return the same code by default
        '''
        return VeriDataset._dataset_type_mapping[dataset_type]

    def __init__(self, data_dir, data_transform = None, 
                 stage:str = None, prediction_file=None, 
                 allowed_type_list: List[Type] = None
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
            name_file = 'name_test.txt'
            self.img_dir = 'image_test'

        super().__init__(data_dir, 
                        stage=stage, 
                        prediction_file=prediction_file,
                        data_transform = data_transform,
                        allowed_type_list = allowed_type_list,
        )
        
        self.names = []

        reader = open(os.path.join(self.data_dir,name_file))
        lines = reader.readlines()
        for line in lines:
            self.names.append(line.rstrip())
        if self.stage in [self.STAGE_TEST, self.STAGE_TRAIN]:
            target_col = "typeID" 
            self.types = []
            label_file = os.path.join(self.data_dir,self.prediction_file)
            xtree = et.parse(label_file, parser=et.XMLParser(encoding="utf-8"))
            xroot = xtree.getroot()[0]
            for label in xroot:
                self.types.append(self.to_common_type(int(label.attrib.get(target_col))-1))
        
        self.names, self.types = self.filter_by_types(allowed_type_list)
        if allowed_type_list != None:
            self._remap_indexes(allowed_type_list)
            
    def _load_predictions(self):
        if self.stage == self.STAGE_PREDICT:
            super()._load_predictions()