import os
from typing import Any, List, Optional
from enum import IntEnum, auto

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Brand(IntEnum):
    AUDI= auto()
    BMW= auto()
    BUICK= auto()
    BYD= auto()
    CADILLAC= auto()
    CHANGAN= auto()
    CHANGHE= auto()
    CHERY= auto()
    CHEVROLET= auto()
    CHINA= auto()
    CITROEN= auto()
    DONGFENG= auto()
    EAST= auto()
    FORD= auto()
    FOTON= auto()
    GEELY= auto()
    GREAT= auto()
    HAFEI= auto()
    HAVAL= auto()
    HIPPOCAMPUS= auto()
    HONDA= auto()
    HYUNDAI= auto()
    IVECO= auto()
    JAC= auto()
    JEEP= auto()
    JIEFANG= auto()
    JINBEI= auto()
    JMC= auto()
    KIA= auto()
    LAND= auto()
    LEXUS= auto()
    LIBERATION= auto()
    MAZDA= auto()
    MERCEDES= auto()
    MG3= auto()
    MG5= auto()
    MINI= auto()
    MITSUBISHI= auto()
    NISSAN= auto()
    PEUGEOT= auto()
    RENAULT= auto()
    ROEWE= auto()
    SEAHORSE= auto()
    SHENWO= auto()
    SKODA= auto()
    SOUTHEAST= auto()
    SUBARU= auto()
    SUZUKI= auto()
    TOYOTA= auto()
    VOLKSWAGEN= auto()
    VOLVO= auto()
    WULING= auto()
    ZHONGHUA= auto()
    
    @staticmethod
    def list():
        return list(map(lambda c: c.value, Brand))

# Superclass for Datasets
class BrandDataset(Dataset):
    STAGE_TRAIN = "train"
    STAGE_TEST = "test"
    STAGE_PREDICT = "predict"

    def __init__(
        self,
        data_dir,
        prediction_file=None,
        stage: Optional[str] = None,
        allowed_brand_list: List[Brand] = None,
        data_transform = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.stage = stage
        self.prediction_file = prediction_file
        self.allowed_brand_list = allowed_brand_list
        self.data_transform = data_transform
        self.brand_master = Brand.list()

        # names will be loaded in the implementation
        self.names = []
        self.brands = None
        if self.prediction_file != None:
            self._load_predictions()

    def _load_predictions(self):
        self.brands = []
        with open(self.prediction_file) as reader:
            lines = reader.readlines()
            for line in lines:
                brand = line.rstrip().split()[0]
                self.brands.append(int(brand))

    @staticmethod
    def to_common_brand(dataset_brand):
        """
        Convert dataset specific brand code to common brand code.
        Return the same code by default
        """
        return dataset_brand

    def get_brand_counts(self):
        if self.brands != None:
            get_brand = lambda i : (
                self.allowed_brand_list[i].name
                if self.allowed_brand_list != None
                else Brand(i).name
            )
            c, counts = np.unique(np.array(self.brands), return_counts=True)
            return [
                (brand, get_brand(brand), count) for brand, count in zip(c, counts)
            ]

    def filter_by_brands(self, brand_list: List[Brand] = None) -> List[Brand]:
        """
        Returns a filtered list of names, brands
        """
        if brand_list == None or brand_list == [] or self.brands == None:
            return self.names, self.brands

        def fil_fun(entry):
            name, brand = entry
            return self.brand_master[brand] in brand_list

        # return a filtered list of names, brands
        filtered_tuples = filter(fil_fun, zip(self.names, self.brands))
        names = []
        brands = []
        for tup in filtered_tuples:
            names.append(tup[0])
            brands.append(tup[1])
        return names, brands

    def _remap_brands(self, custom_brand_list: List[Brand]):
        mapping = []
        if not custom_brand_list:
            self.brands = list(map(lambda b: b.value, self.brands))
            return
            
        for brand in self.brand_master:
            if brand in custom_brand_list:
                mapping.append(custom_brand_list.index(brand))
            else:
                mapping.append(-1)

        def map_fun(brand):
            return mapping[brand]

        # print('old brands', self.brands)
        if self.brands != None:
            self.brands = list(map(map_fun, self.brands))
        # print('new brands', self.brands)

    def __getitem__(self, index):
        # For normalize
        name = self.names[index]
        img = Image.open(os.path.join(self.data_dir, self.img_dir, name))
        # .convert('RGB') # convert gray to rgb

        if self.data_transform != None:
            img = self.data_transform(img)
        if self.stage == self.STAGE_PREDICT:
            # brand are not available
            return img, os.path.join(self.data_dir, self.img_dir, name)
        brand = self.brands[index]
        return img, os.path.join(self.data_dir, self.img_dir, name), brand

    def __len__(self):
        return len(self.names)
