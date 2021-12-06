import os
from typing import Any, List, Union
import xml.etree.ElementTree as et

from . import BrandDataset, Brand


from scipy.io import loadmat
import pickle
import json

class BoxCars116kDataset(BrandDataset):
    dataset_name = "BoxCars116k"
    
    # dataset brands
    _dataset_brand_mapping = {
        'rover': Brand.LAND,
        'mitsubishi': Brand.MITSUBISHI,
        'infinity': Brand.INFINITI,
        'subaru': Brand.SUBARU,
        'lancia': Brand.LANCIA,
        'citroen': Brand.CITROEN,
        'mercedes-benz': Brand.MERCEDES,
        'saab': Brand.SAAB,
        'mazda': Brand.MAZDA,
        'skoda': Brand.SKODA,
        'chrysler': Brand.CHRYSLER,
        'dacia': Brand.DACIA,
        'seat': Brand.SEAT,
        'honda': Brand.HONDA,
        'hyundai': Brand.HYUNDAI,
        'land-rover': Brand.LAND,
        'nissan': Brand.NISSAN,
        'range-rover': Brand.LAND,
        'chevrolet': Brand.CHEVROLET,
        'kia': Brand.KIA,
        'volkswagen': Brand.VOLKSWAGEN,
        'porsche': Brand.PORSCHE,
        'lada': Brand.LADA,
        'bmw': Brand.BMW,
        'suzuki': Brand.SUZUKI,
        'daewoo': Brand.DAEWOO,
        'alfaromeo': Brand.ALFAROMEO,
        'iveco': Brand.IVECO,
        'peugeot': Brand.PEUGEOT,
        'ssangyong': Brand.SSANGYONG,
        'toyota': Brand.TOYOTA,
        'jeep': Brand.JEEP,
        'jaguar': Brand.JAGUAR,
        'renault': Brand.RENAULT,
        'smart': Brand.SMART,
        'tesla': Brand.TESLA,
        'mini': Brand.MINI,
        'volvo': Brand.VOLVO,
        'opel': Brand.OPEL,
        'fiat': Brand.FIAT,
        'toyoto': Brand.TOYOTA,
        'audi': Brand.AUDI,
        'ford': Brand.FORD,
        'lexus': Brand.LEXUS,
    }

    def __init__(
        self,
        data_dir,
        data_transform=None,
        prediction_file: str = None,
        stage: str= None,
        allowed_brand_list: List[Brand] = None
    ):
        super().__init__(
            data_dir, prediction_file=prediction_file, stage=stage
        )
        self.data_dir = data_dir
        self.data_transform = data_transform
        dataset_file = "dataset.pkl"
        split_file = "classification_splits.pkl"
        self.img_dir="images"
        self.names = []
        self.brands: List[int] = []
        self.allowed_brand_list = allowed_brand_list
        # name_brand_map = {}
        encoding = "latin-1"
        with open(os.path.join(data_dir, dataset_file), "rb") as f:
            self.dataset = pickle.load(f, encoding=encoding, fix_imports=True)
        with open(os.path.join(data_dir, split_file), "rb") as f:
            self.split = pickle.load(f)
            self.split = self.split['body']
            if self.stage in [self.STAGE_TEST, self.STAGE_PREDICT]:
                self.split = self.split['test']
            elif self.stage == self.STAGE_TRAIN:
                self.split = self.split['train'] + self.split['validation']
        # with open(os.path.join(os.path.dirname(__file__), model_brand_file), "rb") as f:
        #     for line in f:
        #         name, typ = line.split(',')
        #         name_brand_map[name] = typ
        # print(self.dataset['samples'][0])
        for entry in self.split:
            sample_id, _ = entry
            sample = self.dataset['samples'][sample_id]
            anno = sample['annotation']
            if not anno:
                continue
            brand = anno.split(' ')[0]
            brand = self.to_common_brand(brand)
            self.names.extend([inst['path'] for inst in sample['instances']])
            if self.stage in [self.STAGE_TEST, self.STAGE_TRAIN]:
                dataset_brand = brand
                self.brands.extend([dataset_brand] * len(sample['instances']))
        # print(set(self.brands))
        self.names, self.brands = self.filter_by_brands(allowed_brand_list)
        if allowed_brand_list != None:
            self._remap_brands(allowed_brand_list)
        
            
    def _load_predictions(self):
        # only for predict
        if self.stage == self.STAGE_PREDICT:
            super()._load_predictions()

    @staticmethod
    def to_common_brand(dataset_typ: Union[str,int]) -> Brand:
        """Convert dataset specific brand code to common brand defined in the Brand enum.
        Return the same code by default

        Args:
            dataset_typ (str): [description]

        Returns:
            Brand: [description]
        """
        return BoxCars116kDataset._dataset_brand_mapping[dataset_typ]
