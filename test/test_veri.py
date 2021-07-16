import os
from vtype.dataset import VeriDataset

class TestVeri:
    
    data_dir = os.path.join('dataset', 'VeRi_with_plate',)
    
    def test_type_setup(self):
        stage = VeriDataset.STAGE_TRAIN
        ds = VeriDataset(self.data_dir, None, stage=stage)
        assert len(ds) == 37746
        assert len(ds.names) == len(ds.types)
        type_dist = ds.get_type_counts()
        assert type_dist[0][2] == 19911