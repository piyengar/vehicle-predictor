import os
from typing import List

from . import BrandDataset, Brand


class VehicleIDDataset(BrandDataset):
    dataset_name='VehicleID'
    
    # this brand list is only for reference, 
    # convention is to convert all brands to common brand scheme
    # brand_master = [
    #     'black',
    #     'blue',
    #     'gray',
    #     'red',
    #     'sliver',
    #     'white',
    #     'yellow',
    # ]

    @staticmethod
    def to_common_brand(dataset_brand):
        '''
        Convert dataset specific brand code to common brand code.
        Return the same code by default
        '''
        _common_brand_mapping = [
            Brand.WULING,
            Brand.VOLKSWAGEN,
            Brand.AUDI,
            Brand.TOYOTA,
            Brand.MERCEDES,
            Brand.GREAT,
            Brand.CITROEN,
            Brand.KIA,
            Brand.NISSAN,
            Brand.CITROEN,
            Brand.HYUNDAI,
            Brand.HONDA,
            Brand.NISSAN,
            Brand.FOTON,
            Brand.MITSUBISHI,
            Brand.SKODA,
            Brand.KIA,
            Brand.BUICK,
            Brand.JINBEI,
            Brand.HYUNDAI,
            Brand.TOYOTA,
            Brand.TOYOTA,
            Brand.CITROEN,
            Brand.HONDA,
            Brand.HYUNDAI,
            Brand.MERCEDES,
            Brand.VOLVO,
            Brand.HIPPOCAMPUS,
            Brand.HYUNDAI,
            Brand.MG5,
            Brand.JMC,
            Brand.BYD,
            Brand.HAVAL,
            Brand.BYD,
            Brand.MG3,
            Brand.FORD,
            Brand.SEAHORSE,
            Brand.PEUGEOT,
            Brand.NISSAN,
            Brand.HYUNDAI,
            Brand.IVECO,
            Brand.FORD,
            Brand.CHANGAN,
            Brand.HYUNDAI,
            Brand.CHANGAN,
            Brand.CHERY,
            Brand.AUDI,
            Brand.MERCEDES,
            Brand.VOLKSWAGEN,
            Brand.HYUNDAI,
            Brand.MAZDA,
            Brand.HAVAL,
            Brand.HYUNDAI,
            Brand.VOLKSWAGEN,
            Brand.TOYOTA,
            Brand.CHEVROLET,
            Brand.VOLKSWAGEN,
            Brand.CHEVROLET,
            Brand.BYD,
            Brand.VOLVO,
            Brand.SUZUKI,
            Brand.LIBERATION,
            Brand.KIA,
            Brand.CITROEN,
            Brand.MAZDA,
            Brand.AUDI,
            Brand.MERCEDES,
            Brand.WULING,
            Brand.BUICK,
            Brand.BUICK,
            Brand.BUICK,
            Brand.BMW,
            Brand.BUICK,
            Brand.MITSUBISHI,
            Brand.BMW,
            Brand.CHERY,
            Brand.TOYOTA,
            Brand.CHERY,
            Brand.NISSAN,
            Brand.MAZDA,
            Brand.NISSAN,
            Brand.HIPPOCAMPUS,
            Brand.CHANGAN,
            Brand.CHANGHE,
            Brand.HYUNDAI,
            Brand.PEUGEOT,
            Brand.HYUNDAI,
            Brand.SUZUKI,
            Brand.SKODA,
            Brand.CHERY,
            Brand.TOYOTA,
            Brand.GREAT,
            Brand.HYUNDAI,
            Brand.TOYOTA,
            Brand.CHERY,
            Brand.JINBEI,
            Brand.MAZDA,
            Brand.CHEVROLET,
            Brand.PEUGEOT,
            Brand.MAZDA,
            Brand.SUBARU,
            Brand.CHINA,
            Brand.MERCEDES,
            Brand.VOLKSWAGEN,
            Brand.EAST,
            Brand.BUICK,
            Brand.HYUNDAI,
            Brand.AUDI,
            Brand.BMW,
            Brand.TOYOTA,
            Brand.MITSUBISHI,
            Brand.LEXUS,
            Brand.ROEWE,
            Brand.AUDI,
            Brand.TOYOTA,
            Brand.SUBARU,
            Brand.WULING,
            Brand.HONDA,
            Brand.HONDA,
            Brand.HYUNDAI,
            Brand.CHERY,
            Brand.TOYOTA,
            Brand.ZHONGHUA,
            Brand.HYUNDAI,
            Brand.PEUGEOT,
            Brand.AUDI,
            Brand.BUICK,
            Brand.DONGFENG,
            Brand.SKODA,
            Brand.BMW,
            Brand.KIA,
            Brand.HONDA,
            Brand.GEELY,
            Brand.TOYOTA,
            Brand.JIEFANG,
            Brand.AUDI,
            Brand.BMW,
            Brand.LEXUS,
            Brand.SOUTHEAST,
            Brand.SHENWO,
            Brand.BMW,
            Brand.HAVAL,
            Brand.CITROEN,
            Brand.FORD,
            Brand.NISSAN,
            Brand.LEXUS,
            Brand.WULING,
            Brand.MINI,
            Brand.BYD,
            Brand.TOYOTA,
            Brand.ROEWE,
            Brand.HYUNDAI,
            Brand.FORD,
            Brand.IVECO,
            Brand.BYD,
            Brand.NISSAN,
            Brand.CADILLAC,
            Brand.KIA,
            Brand.BMW,
            Brand.TOYOTA,
            Brand.SUBARU,
            Brand.CHERY,
            Brand.JEEP,
            Brand.CHINA,
            Brand.NISSAN,
            Brand.NISSAN,
            Brand.CITROEN,
            Brand.BMW,
            Brand.JAC,
            Brand.VOLKSWAGEN,
            Brand.KIA,
            Brand.BMW,
            Brand.CHEVROLET,
            Brand.TOYOTA,
            Brand.DONGFENG,
            Brand.FORD,
            Brand.JAC,
            Brand.HIPPOCAMPUS,
            Brand.DONGFENG,
            Brand.SHENWO,
            Brand.SHENWO,
            Brand.KIA,
            Brand.DONGFENG,
            Brand.VOLKSWAGEN,
            Brand.HONDA,
            Brand.PEUGEOT,
            Brand.RENAULT,
            Brand.JINBEI,
            Brand.BYD,
            Brand.LIBERATION,
            Brand.DONGFENG,
            Brand.VOLKSWAGEN,
            Brand.TOYOTA,
            Brand.VOLKSWAGEN,
            Brand.ROEWE,
            Brand.IVECO,
            Brand.ROEWE,
            Brand.SUZUKI,
            Brand.BUICK,
            Brand.SUZUKI,
            Brand.JAC,
            Brand.CITROEN,
            Brand.HAFEI,
            Brand.FOTON,
            Brand.TOYOTA,
            Brand.FORD,
            Brand.SOUTHEAST,
            Brand.RENAULT,
            Brand.SEAHORSE,
            Brand.VOLKSWAGEN,
            Brand.HONDA,
            Brand.NISSAN,
            Brand.HONDA,
            Brand.CHEVROLET,
            Brand.KIA,
            Brand.FORD,
            Brand.DONGFENG,
            Brand.FORD,
            Brand.TOYOTA,
            Brand.LAND,
            Brand.NISSAN,
            Brand.CITROEN,
            Brand.HYUNDAI,
            Brand.RENAULT,
            Brand.CHEVROLET,
            Brand.KIA,
            Brand.VOLKSWAGEN,
            Brand.VOLKSWAGEN,
            Brand.VOLKSWAGEN,
            Brand.WULING,
            Brand.KIA,
            Brand.CITROEN,
            Brand.HONDA,
            Brand.CHANGAN,
            Brand.KIA,
            Brand.VOLKSWAGEN,
            Brand.CHEVROLET,
            Brand.CHERY,
            Brand.DONGFENG,
            Brand.FOTON,
            Brand.VOLKSWAGEN,
            Brand.NISSAN,
            Brand.CHANGAN,
            Brand.SUZUKI,
            Brand.VOLKSWAGEN,
            Brand.GREAT,
            Brand.TOYOTA,
            Brand.JAC,
            Brand.TOYOTA,
            Brand.MERCEDES,
        ]
        return _common_brand_mapping[dataset_brand]

    def __init__(self, data_dir, 
                 data_transform = None, stage:str = None, prediction_file=None,
                 allowed_brand_list:List[Brand] = None
                 ):
        self.img_dir = 'image'
        if stage == self.STAGE_TRAIN:
            name_file = 'train_test_split/train_list.txt'
            prediction_file = 'attribute/model_attr.txt'
        elif stage == self.STAGE_TEST:
            name_file = 'train_test_split/test_list.txt'
            prediction_file = 'attribute/model_attr.txt'
        else: # predict
            name_file = 'train_test_split/train_list.txt'
        self.name_file = name_file
        super().__init__(data_dir, stage=stage, prediction_file=prediction_file, data_transform = data_transform)
        # self.data_transform = data_transform
        self.allowed_brand_list = allowed_brand_list
        self.names = []
        if self.stage in [self.STAGE_TRAIN, self.STAGE_TEST]:
            # Only small subset of data has brand labels
            # print('loading brand attributes')
            self.brand_attr = {} # vid -> brand_id
            self.brands = []
            with open(os.path.join(self.data_dir,self.prediction_file)) as reader:
                lines = reader.readlines()
                for line in lines:
                    vid, brand = line.rstrip().split()
                    # print(f'Found name, brand = {vid, brand}')
                    self.brand_attr[vid] = brand

        with open(os.path.join(self.data_dir,self.name_file)) as reader:
            lines = reader.readlines()
            if self.stage in [self.STAGE_TRAIN, self.STAGE_TEST]:
                for line in lines:
                    name, vid = line.rstrip().split()
                    if vid in self.brand_attr:
                        self.names.append(f'{name}.jpg')
                        self.brands.append(self.to_common_brand(int(self.brand_attr[vid])))
                    # else:
                    #     print(f'could not find name - {name} in brand _attr')
            else:
                for line in lines:
                    name = line.rstrip().split()[0]
                    self.names.append(f'{name}.jpg')
        self.names, self.brands = self.filter_by_brands(allowed_brand_list)
        if allowed_brand_list != None:
            # print('remapping brand list to allowed list')
            self._remap_brands(allowed_brand_list)

    def _load_predictions(self):
        # only for predict
        if self.stage == self.STAGE_PREDICT:
            super()._load_predictions()
