import os

from brand import BrandDataModule, BrandDatasets
from brand.dataset import VehicleIDDataset, Brand, BrandDataset


class TestVehicleID:
    
    dataset_root = 'dataset'
    data_dir = os.path.join(dataset_root, 'VehicleID')
    allowed_brand_list = [
        Brand.HYUNDAI,
        Brand.VOLKSWAGEN,
        Brand.BUICK,
        Brand.WULING,
        Brand.CHEVROLET,
        Brand.NISSAN,
        Brand.KIA,
        Brand.TOYOTA,
        Brand.AUDI,
        Brand.HONDA,
    ]
    
    def test_dataset_setup(self):
        stage = BrandDataset.STAGE_TEST
        ds = VehicleIDDataset(self.data_dir, None, stage=stage)
        assert len(ds) == 8144
        brand_dist = ds.get_brand_counts()
        assert brand_dist[0][2] == 2075
        
        
        ds = VehicleIDDataset(self.data_dir, None, stage=stage, allowed_brand_list=self.allowed_brand_list)
        brand_dist = ds.get_brand_counts()
        assert len(brand_dist) == 4
        
    def test_datamodule_setup(self):
        stage = BrandDataset.STAGE_TRAIN
        dm = BrandDataModule(BrandDatasets.VEHICLE_ID, self.dataset_root)
        dm.setup('fit')
        ds = dm.train_val_dataset
        assert len(ds) == 8144
        brand_dist = ds.get_brand_counts()
        assert brand_dist[0][2] == 2075
        
        dm = BrandDataModule(BrandDatasets.VEHICLE_ID, self.dataset_root, allowed_brand_list=self.allowed_brand_list)
        dm.setup('fit')
        ds = dm.train_val_dataset
        brand_dist = ds.get_brand_counts()
        assert len(brand_dist) == 4
        
        