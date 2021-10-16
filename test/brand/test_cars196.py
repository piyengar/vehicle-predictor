import os

from brand import BrandDataModule, BrandDatasets
from brand.dataset import Cars196Dataset, Brand, BrandDataset


class TestCars196:
    
    dataset_root = 'dataset'
    data_dir = os.path.join(dataset_root, 'Cars196')
    allowed_brand_list = [
        Brand.MAYBACH,
        Brand.MCLAREN,
        Brand.PLYMOUTH,
        Brand.PORSCHE,
        Brand.RAM,
        Brand.ROLLS_ROYCE,
        Brand.SCION,
        Brand.SPYKER,
        Brand.TESLA,
        Brand.SMART,
    ]
    
    def test_dataset_setup(self):
        stage = BrandDataset.STAGE_TRAIN
        ds = Cars196Dataset(self.data_dir, None, stage=stage)
        assert len(ds) == 8144
        brand_dist = ds.get_brand_counts()
        assert brand_dist[0][2] == 589
        
        
        ds = Cars196Dataset(self.data_dir, None, stage=stage, allowed_brand_list=self.allowed_brand_list)
        brand_dist = ds.get_brand_counts()
        assert len(brand_dist) == 10
        
    def test_datamodule_setup(self):
        stage = BrandDataset.STAGE_TRAIN
        dm = BrandDataModule(BrandDatasets.CARS196, self.dataset_root)
        dm.setup('fit')
        ds = dm.train_val_dataset
        assert len(ds) == 8144
        brand_dist = ds.get_brand_counts()
        assert brand_dist[0][2] == 589
        
        dm = BrandDataModule(BrandDatasets.CARS196, self.dataset_root, allowed_brand_list=self.allowed_brand_list)
        dm.setup('fit')
        ds = dm.train_val_dataset
        brand_dist = ds.get_brand_counts()
        assert len(brand_dist) == 10
        
        