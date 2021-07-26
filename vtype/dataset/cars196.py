import os
import xml.etree.ElementTree as et
from typing import List

from scipy.io import loadmat

from . import Type, TypeDataset


class Cars196Dataset(TypeDataset):
    dataset_name = "Cars196"
     # dataset types
    _dataset_type_mapping = {
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
        allowed_type_list: List[Type] = None,
    ):
        model_type_file = 'cars196_model_types.txt'
        super().__init__(
            data_dir, prediction_file=prediction_file, stage=stage
        )
        self.allowed_type_list = allowed_type_list
        self.stage = stage
        self.src_classes = []
        with open(os.path.join(os.path.dirname(__file__), model_type_file), "r") as f:
            for line in f:
                # store the class name, type
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
            self.types = []
            # load type info from class labels
            for class_idx in mat["annotations"]["class"][0]:
                typ = self.src_classes[class_idx[0][0] - 1][1]
                self.types.append(self.to_common_type(typ))
        self.names, self.types = self.filter_by_types(allowed_type_list)
        if allowed_type_list != None:
            self._remap_indexes(allowed_type_list)

    def _load_predictions(self):
        # only for predict
        if self.stage == self.STAGE_PREDICT:
            super()._load_predictions()
            
    @staticmethod
    def to_common_type(dataset_typ: str) -> Type:
        """Convert dataset specific type code to common type defined in the Type enum.
        Return the same code by default

        Args:
            dataset_typ (str): [description]

        Returns:
            Type: [description]
        """
        return Cars196Dataset._dataset_type_mapping[dataset_typ.lower()]