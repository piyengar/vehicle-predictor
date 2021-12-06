import os
from typing import List, Union

from scipy.io import loadmat

from . import BrandDataset, Brand


# CompCars
class CompCarsDataset(BrandDataset):
    dataset_name = "CompCars"

    _dataset_brand_mapping = {
        'Acura': Brand.ACURA,
        'Audi': Brand.AUDI,
        'BWM': Brand.BMW,
        'BYD': Brand.BYD,
        'Baojun': Brand.BAOJUN,
        'Benz': Brand.MERCEDES,
        'Besturn': Brand.BESTURN,
        'Buck': Brand.BUICK,
        'Cadillac': Brand.CADILLAC,
        'Changan': Brand.CHANGAN,
        'Changhe': Brand.CHANGHE,
        'Chevy': Brand.CHEVROLET,
        'Chrey': Brand.CHEVROLET,
        'Chrysler': Brand.CHRYSLER,
        'Citroen': Brand.CITROEN,
        'Dodge': Brand.DODGE,
        'Dongfengfengdu': Brand.DONGFENG,
        'Dongfengfengshen': Brand.DONGFENG,
        'Everus': Brand.EVERUS,
        'FIAT': Brand.FIAT,
        'Ford': Brand.FORD,
        'GAC': Brand.GAC,
        'Geely': Brand.GEELY,
        'GreatWall': Brand.GREAT,
        'Guangqichuanqi': Brand.GUANGQICHUANQI,
        'Hafei': Brand.HAFEI,
        'Haima': Brand.HAIMA,
        'Haima(Zhengzhou)': Brand.HAIMA,
        'Haval': Brand.HAVAL,
        'Honda': Brand.HONDA,
        'Huanghai': Brand.HUANGHAI,
        'Huatai': Brand.HUATAI,
        'Hyundai ': Brand.HYUNDAI,
        'Infiniti': Brand.INFINITI,
        'Jaguar': Brand.JAGUAR,
        'Jeep': Brand.JEEP,
        'Jianghuai': Brand.JIANGHUAI,
        'Jinbei': Brand.JINBEI,
        'KIA': Brand.KIA,
        'Karry': Brand.KARRY,
        'LAND-ROVER': Brand.LAND,
        'Lexus': Brand.LEXUS,
        'Lifan': Brand.LIFAN,
        'Lotus': Brand.LOTUS,
        'Lufeng': Brand.LUFENG,
        'MAZDA': Brand.MAZDA,
        'Maserati': Brand.MASERATI,
        'Mitsubishi': Brand.MITSUBISHI,
        'Nissan': Brand.NISSAN,
        'Opel': Brand.OPEL,
        'Peugeot': Brand.PEUGEOT,
        'Porsche': Brand.PORSCHE,
        'Renault': Brand.RENAULT,
        'Roewe': Brand.ROEWE,
        'Shangqidatong': Brand.SHANGQIDATONG,
        'Shuanghuan': Brand.SHUANGHUAN,
        'Shuanglong': Brand.SHUANGLONG,
        'Skoda': Brand.SKODA,
        'Subaru': Brand.SUBARU,
        'Suzuki': Brand.SUZUKI,
        'Toyota': Brand.TOYOTA,
        'Venucia': Brand.VENUCIA,
        'Volkswagen': Brand.VOLKSWAGEN,
        'Volvo': Brand.VOLVO,
        'Wuling': Brand.WULING,
        'Yiqi': Brand.YIQI,
        'Zhonghua': Brand.ZHONGHUA,
        'Zoyte': Brand.ZOYTE,
    }
    
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
        # attributes = {}
        # with open(os.path.join(os.path.dirname(__file__), 'comp_cars_files', 'attributes.txt'), 'r') as reader:
        #     first = True
        #     for entry in reader:
        #         if first:
        #             first = False
        #             continue
        #         id, _,_,doors,seats,typ = entry.split(' ')
        #         attributes[int(id)] = int(doors),int(typ.strip())
        model_contents = loadmat(os.path.join(self.data_dir, 'sv_make_model_name.mat'))
        # print(model_contents['sv_make_model_name'])
        sv_web_model_map = {
            i + 1: cont[0][0]
            for i, cont in enumerate(model_contents['sv_make_model_name'])
        }
        # print(sv_web_model_map)

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
        self.names = []
        self.brands = []
        with open(os.path.join(self.data_dir, self.name_file)) as reader:
            lines = reader.readlines()
            for line in lines:
                name = line.rstrip()
                self.names.append(name)
                if self.stage in [self.STAGE_TRAIN, self.STAGE_TEST]:
                    # load brands for train and test stages
                    sv_model_id = int(os.path.split(name)[0])
                    brand = self.to_common_brand(sv_web_model_map[sv_model_id])
                    self.brands.append(brand)
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