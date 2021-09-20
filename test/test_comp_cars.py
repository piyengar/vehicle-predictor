import os
from vtype.dataset import CompCarsDataset

class TestCompCars:
    
    data_dir = os.path.join('dataset', 'CompCars', 'sv_data')
    
    def test_type_setup(self):
        stage = CompCarsDataset.STAGE_TRAIN
        ds = CompCarsDataset(self.data_dir, None, stage=stage)
        assert len(ds) == 31148
        assert len(ds.names) == len(ds.types)
        type_dist = ds.get_type_counts()
        assert type_dist[0][2] == 9736