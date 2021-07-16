from enum import Enum, IntEnum
import os
from typing import Any, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class Type(IntEnum):
    SEDAN = 0
    HATCHBACK = 1
    COUPE = 2
    SUV = 3
    CONVERTIBLE = 4
    PICKUP = 5
    MINIVAN = 6
    WAGON = 7
    BUS = 8
    TRUCK = 9
    
    @staticmethod
    def list():
        return list(map(lambda c: c.value, Type))
    
# Superclass for Datasets
class TypeDataset(Dataset):
    # These are all the possible types, dataset specific types should be defined in the respective sub class
    
    STAGE_TRAIN = "train"
    STAGE_TEST = "test"
    STAGE_PREDICT = "predict"

    def __init__(
        self,
        data_dir,
        prediction_file=None,
        stage: Optional[str] = None,
        data_transform=None,
        allowed_type_list: List[Type] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.stage = stage
        self.prediction_file = prediction_file
        self.data_transform = data_transform
        self.allowed_color_list = allowed_type_list
        self.type_master = Type.list()

        # names will be loaded in the implementation
        self.names = []
        self.types = []
        if self.prediction_file != None:
            self._load_predictions()

    def _load_predictions(self):
        self.types = []
        with open(self.prediction_file) as reader:
            lines = reader.readlines()
            for line in lines:
                typ = line.rstrip().split()[0]
                self.types.append(int(typ))

    @staticmethod
    def to_common_type(dataset_typ):
        """
        Convert dataset specific typ code to common typ code.
        Return the same code by default
        """
        return dataset_typ

    def get_type_counts(self, allowed_type_list= None):
        if self.types != None:
            get_type = lambda i : (
                allowed_type_list[i]
                if allowed_type_list != None
                else Type(i).name
            )
            c, counts = np.unique(np.array(self.types), return_counts=True)
            return [
                (typ, get_type(typ), count) for typ, count in zip(c, counts)
            ]

    def filter_by_types(self, type_list: List[Type] = None):
        """
        Returns a filtered list of names, types
        """
        if type_list == None or type_list == [] or self.types == None:
            return self.names, self.types

        def fil_fun(entry):
            name, typ = entry
            return self.type_master[typ] in type_list

        # return a filtered list of names, types
        filtered_tuples = filter(fil_fun, zip(self.names, self.types))
        names = []
        types = []
        for tup in filtered_tuples:
            names.append(tup[0])
            types.append(tup[1])
        return names, types

    def _remap_indexes(self, custom_type_list: List[Type]):
        """Remaps indexes from those defined in the Type enum to the order 
        specified in the custom_type_list var. mutates the "types" instance variable

        Args:
            custom_type_list (List[Type]): The new order that the types field should refer to.
        
        """
        mapping = []
        for typ in self.type_master:
            if typ in custom_type_list:
                mapping.append(custom_type_list.index(typ))
            else:
                mapping.append(-1)

        def map_fun(typ):
            return mapping[typ]

        # print('old types', self.types)
        if self.types != None:
            self.types = list(map(map_fun, self.types))
        # print('new types', self.types)

    def __getitem__(self, index):
        # For normalize
        name = self.names[index]
        img = Image.open(os.path.join(self.data_dir, self.img_dir, name))
        # .convert('RGB') # convert gray to rgb

        if self.data_transform != None:
            img = self.data_transform(img)
        if self.stage == self.STAGE_PREDICT:
            # typ are not available
            return img, os.path.join(self.data_dir, self.img_dir, name)
        typ = self.types[index]
        return img, os.path.join(self.data_dir, self.img_dir, name), typ

    def __len__(self):
        return len(self.names)
