import os
from vtype.dataset import Cars196Dataset

class TestCars196:
    
    data_dir = os.path.join('dataset', 'Cars196')
    
    def test_type_setup(self):
        stage = Cars196Dataset.STAGE_TRAIN
        ds = Cars196Dataset(self.data_dir, None, stage=stage)
        assert len(ds) == 8144
        type_dist = ds.get_type_counts()
        assert type_dist[0][2] == 2075