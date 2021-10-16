import os
from typing import List, Union

from scipy.io import loadmat

from . import BrandDataset, Brand


# CompCars
class CompCarsDataset(BrandDataset):
    dataset_name = "CompCars"

    _dataset_brand_mapping = [
        # Type.SUV,           # MPV
        # Type.SUV,           # SUV
        # Type.SEDAN,         # sedan
        # Type.HATCHBACK,     # hatchback
        # Type.MINIVAN,       # minibus
        # Type.SEDAN,         # fastback
        # Type.SUV,           # estate
        # Type.PICKUP,        # pickup
        # Type.SEDAN,   # hardtop convertible
        # Type.SEDAN,         # sports
        # Type.SUV,           # crossover
        # Type.SEDAN,   # convertible
    ]
    
    def __init__(
        self,
        data_dir,
        data_transform=None,
        stage: str = None,
        prediction_file=None,
        allowed_brand_list: List[str] = None,
    ):
        self.img_dir = "image"
        if stage == self.STAGE_TRAIN:
            name_file = "train_surveillance.txt"
        elif stage in [self.STAGE_TEST, self.STAGE_PREDICT]:
            name_file = "test_surveillance.txt"
        self.name_file = name_file
        super().__init__(data_dir, stage=stage, prediction_file=prediction_file, data_transform = data_transform, allowed_brand_list = allowed_brand_list)
        self.img_dir = "image"
        # vehicle_id -> attribute
        attributes = {}
        with open(os.path.join(os.path.dirname(__file__), 'comp_cars_files', 'attributes.txt'), 'r') as reader:
            first = True
            for entry in reader:
                if first:
                    first = False
                    continue
                id, _,_,doors,seats,typ = entry.split(' ')
                attributes[int(id)] = int(doors),int(typ.strip())
        model_contents = loadmat(os.path.join(self.data_dir, 'sv_make_model_name.mat'))
        sv_web_model_map = {
            i + 1: cont[2][0][0]
            for i, cont in enumerate(model_contents['sv_make_model_name'])
        }

        # if self.stage in [self.STAGE_TRAIN, self.STAGE_TEST]:
        #     # Only small subset of data has color labels
        #     # print('loading color attributes')
        #     self.colors = []
        #     self.color_attr = {}
        #     colors_mat = loadmat(os.path.join(data_dir, self.prediction_file))
        #     for row in colors_mat["color_list"]:
        #         name, color = row[0][0], row[1][0][0]
        #         if color != -1:
        #             self.color_attr[name] = color

        with open(os.path.join(self.data_dir, self.name_file)) as reader:
            lines = reader.readlines()
            for line in lines:
                name = line.rstrip()
                self.names.append(name)
                if self.stage in [self.STAGE_TRAIN, self.STAGE_TEST]:
                    # load brands for train and test stages
                    sv_model_id = int(os.path.split(name)[0])
                    typ = attributes[sv_web_model_map[sv_model_id]][1]
                    typ = self.to_common_brand(typ)
                    self.brands.append(typ)
        self.names, self.brands = self.filter_by_brands(allowed_brand_list)
        if allowed_brand_list != None:
            self._remap_brands(allowed_brand_list)

    def _load_predictions(self):
        # only for predict
        if self.stage == self.STAGE_PREDICT:
            super()._load_predictions()

    @staticmethod
    def to_common_brand(dataset_typ: Union[str,int]) -> Brand:
        """Convert dataset specific brand code to common brand defined in the Type enum.
        Return the same code by default

        Args:
            dataset_typ (str): [description]

        Returns:
            Type: [description]
        """
        return CompCarsDataset._dataset_brand_mapping[dataset_typ]