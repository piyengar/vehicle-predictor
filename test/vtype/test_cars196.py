import os

from vtype import TypeDataModule, TypeDatasets
from vtype.dataset import Cars196Dataset, Type, TypeDataset


class TestCars196:
    
    dataset_root = 'dataset'
    data_dir = os.path.join(dataset_root, 'Cars196')
    allowed_type_list = [
        Type.SEDAN,
        Type.HATCHBACK,
        Type.SUV,
        Type.PICKUP,
    ]
    
    def test_dataset_setup(self):
        stage = TypeDataset.STAGE_TRAIN
        ds = Cars196Dataset(self.data_dir, None, stage=stage)
        assert len(ds) == 8144
        type_dist = ds.get_type_counts()
        assert type_dist[0][2] == 4493
        
        
        ds = Cars196Dataset(self.data_dir, None, stage=stage, allowed_type_list=self.allowed_type_list)
        type_dist = ds.get_type_counts()
        assert len(type_dist) == 4
        
    def test_datamodule_setup(self):
        stage = TypeDataset.STAGE_TRAIN
        dm = TypeDataModule(TypeDatasets.CARS196, self.dataset_root)
        dm.setup('fit')
        ds = dm.train_val_dataset
        assert len(ds) == 8144
        type_dist = ds.get_type_counts()
        assert type_dist[0][2] == 4493
        
        dm = TypeDataModule(TypeDatasets.CARS196, self.dataset_root, allowed_type_list=self.allowed_type_list)
        dm.setup('fit')
        ds = dm.train_val_dataset
        type_dist = ds.get_type_counts()
        assert len(type_dist) == 4
        
        