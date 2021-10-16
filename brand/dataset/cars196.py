import os
import xml.etree.ElementTree as et
from typing import List

from scipy.io import loadmat

from . import Type, BrandDataset


class Cars196Dataset(BrandDataset):
    dataset_name = "Cars196"
     # dataset brands
    _dataset_brand_mapping = {
        "suv" : Type.SUV,
        "sedan" : Type.SEDAN,
        "coupe" : Type.SEDAN,
        "hatchback" : Type.HATCHBACK,
        "convertible" : Type.SEDAN,
        "wagon" : Type.SUV,
        "pickup" : Type.PICKUP,
        "van" : Type.MINIVAN,
        "cab" : Type.PICKUP,
        "minivan" : Type.MINIVAN,
        "supercab" : Type.PICKUP,
    }
    
    def __init__(
        self,
        data_dir,
        data_transform=None,
        prediction_file: str = None,
        stage: str = None,
        allowed_brand_list: List[Type] = None,
    ):
        model_brand_file = 'cars196_model_brands.txt'
        super().__init__(
            data_dir, prediction_file=prediction_file, stage=stage
        )
        self.allowed_brand_list = allowed_brand_list
        self.stage = stage
        self.src_classes = []
        with open(os.path.join(os.path.dirname(__file__), model_brand_file), "r") as f:
            for line in f:
                # store the class name, brand
                self.src_classes.append(line.strip().split(','))
        if self.stage in [self.STAGE_TEST, self.STAGE_PREDICT]:
            self.img_dir="cars_test"
            self.img_list="cars_test_annos_withlabels.mat"
        elif self.stage == self.STAGE_TRAIN:
            self.img_dir="cars_train"
            self.img_list="cars_train_annos.mat"
        
        self.data_transform = data_transform
        self.names = []
        mat = loadmat(os.path.join(data_dir, self.img_list))
        self.names = [name[0] for name in mat["annotations"]["fname"][0]]
        if self.stage in [self.STAGE_TEST, self.STAGE_TRAIN]:
            self.brands = []
            # load brand info from class labels
            for class_idx in mat["annotations"]["class"][0]:
                typ = self.src_classes[class_idx[0][0] - 1][1]
                self.brands.append(self.to_common_brand(typ))
        self.names, self.brands = self.filter_by_brands(allowed_brand_list)
        if allowed_brand_list != None:
            self._remap_brands(allowed_brand_list)

    def _load_predictions(self):
        # only for predict
        if self.stage == self.STAGE_PREDICT:
            super()._load_predictions()
            
    @staticmethod
    def to_common_brand(dataset_typ: str) -> Type:
        """Convert dataset specific brand code to common brand defined in the Type enum.
        Return the same code by default

        Args:
            dataset_typ (str): [description]

        Returns:
            Type: [description]
        """
        return Cars196Dataset._dataset_brand_mapping[dataset_typ.lower()]