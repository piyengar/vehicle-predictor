import os
from typing import Any, List
import xml.etree.ElementTree as et

from . import TypeDataset, Type


from scipy.io import loadmat
import pickle
import json

class BoxCars116kDataset(TypeDataset):
    dataset_name = "BoxCars116k"
    
    # dataset types
    _dataset_type_mapping = {
        "combi" : Type.WAGON,
        "fastback": Type.SEDAN,
        "coupe" : Type.COUPE,
        "pickup" : Type.PICKUP,
        "cabriolet" : Type.CONVERTIBLE,
        "mpv" : Type.SUV,
        "suv" : Type.SUV,
        "offroad" : Type.SUV,
        "van" : Type.MINIVAN,
        "sedan" : Type.SEDAN,
        "hatchback" : Type.HATCHBACK,
    }
    # combi -> wagon
    # fastback
    # coupe
    # pickup
    # cabriolet -> convertible
    # mpv -> suv
    # suv
    # offroad -> suv
    # van
    # sedan
    # hatchback

    def __init__(
        self,
        data_dir,
        data_transform=None,
        prediction_file: str = None,
        stage: str= None,
        allowed_type_list: List[str] = None
    ):
        super().__init__(
            data_dir, prediction_file=prediction_file, stage=stage
        )
        self.data_dir = data_dir
        self.data_transform = data_transform
        dataset_file = "dataset.pkl"
        split_file = "classification_splits.pkl"
        self.img_dir="images"
        self.paths = []
        self.types: List[int] = []
        # name_type_map = {}
        encoding = "latin-1"
        with open(os.path.join(data_dir, dataset_file), "rb") as f:
            self.dataset = pickle.load(f, encoding=encoding, fix_imports=True)
        with open(os.path.join(data_dir, split_file), "rb") as f:
            self.split = pickle.load(f)['body']
            if self.stage in [self.STAGE_TEST, self.STAGE_PREDICT]:
                self.split = self.split['test']
            elif self.stage == self.STAGE_TRAIN:
                self.split = self.split['train']
        # with open(os.path.join(os.path.dirname(__file__), model_type_file), "rb") as f:
        #     for line in f:
        #         name, typ = line.split(',')
        #         name_type_map[name] = typ

        for entry in self.split:
            sample_id, instance_id = entry
            sample = self.dataset['samples'][sample_id]
            self.paths.append(sample['instances'][instance_id]['path'])
            if self.stage in [self.STAGE_TEST, self.STAGE_TRAIN]:
                dataset_type = sample['annotation'].split(' ')[-2]
                self.types.append(self.to_common_type(dataset_type).value)

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
        return BoxCars116kDataset._dataset_type_mapping[dataset_typ]
