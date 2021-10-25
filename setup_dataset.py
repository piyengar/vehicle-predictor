import os
import subprocess
import argparse
from zipfile import ZipFile
from enum import Enum, auto
import gdown

class Datasets(Enum):
    VEHICLE_ID = auto()
    VRIC = auto()
    CARS196 = auto()
    BOXCARS116K = auto()
    COMP_CARS = auto()
    VERI = auto()
    COMBINED = auto()
    
    def __str__(self):
        return self.name
    
    @staticmethod
    def from_string(s):
        try:
            return Datasets[s]
        except KeyError:
            raise ValueError()


def setup_dataset(dataset: Datasets, dataset_src_root:str= 'carzam', dest_dir:str = 'dataset'):
    if dataset == Datasets.COMP_CARS:
        os.makedirs(os.path.join(dest_dir, 'Datasets'), exist_ok=True)
        archive_path = os.path.join(dataset_src_root, 'CompCars', 'sv_data.zip')
        temp_file = os.path.join(dest_dir, 'cc_combined.zip')
        subprocess.run(['zip', '-F', archive_path,  '-b', dest_dir, '--out', temp_file])
        subprocess.run(['unzip', '-P', 'd89551fd190e38', '-d', os.path.join(dest_dir, 'CompCars'), temp_file])
        os.remove(temp_file)
    elif dataset == Datasets.VEHICLE_ID:
        os.makedirs(os.path.join(dest_dir), exist_ok=True)
        archive_path = os.path.join(dataset_src_root, 'Datasets', 'VehicleID_V1.0.zip')
        subprocess.run(['unzip', '-P', 'CVPR16_IDM@PKU', '-d', dest_dir, archive_path])
        os.rename(os.path.join(dest_dir, "VehicleID_V1.0"), os.path.join(dest_dir, "VehicleID"))
    elif dataset == Datasets.VERI:
        os.makedirs(os.path.join(dest_dir), exist_ok=True)
        archive_path = os.path.join(dataset_src_root, 'Datasets', 'VeRi_with_plate.zip')
        subprocess.run(['unzip', '-d', dest_dir, archive_path])
    elif dataset == Datasets.CARS196:
        os.makedirs(os.path.join(dest_dir, 'Cars196'), exist_ok=True)
        archive_path = os.path.join(dataset_src_root, 'Datasets', 'Cars196.zip')
        # print(archive_path)
        # print(" ".join(['unzip', '-d', os.path.join(dest_dir, 'Cars196'), archive_path]))
        subprocess.run(['unzip', '-d', os.path.join(dest_dir, 'Cars196'), archive_path])
    elif dataset == Datasets.BOXCARS116K:
        os.makedirs(os.path.join(dest_dir), exist_ok=True)
        archive_path = os.path.join(dataset_src_root, 'Datasets', 'BoxCars116k.zip')
        subprocess.run(['unzip', '-d', dest_dir, archive_path])
    elif dataset == Datasets.VRIC:
        os.makedirs(os.path.join(dest_dir, 'VRIC'), exist_ok=True)
        archive_path = os.path.join(dataset_src_root, 'Datasets', 'VRIC.zip')
        subprocess.run(['unzip', '-d', os.path.join(dest_dir, 'VRIC'), archive_path])
        
def download_dataset(dataset: Datasets, source: str = 'gdrive', temp_folder:str = 'dataset_source'):
    # TODO need to handle source other than `gdrive`
    if source == 'gdrive':
        files = []
        if dataset == Datasets.COMP_CARS:
            files = [
                'https://drive.google.com/file/d/1KK5802tW2fXJN-q9rj2yrJ8hZASItXe_/view?usp=sharing',
                'https://drive.google.com/file/d/13s3RUeD4xsC3N2My8YxTkqymEAo_mmdt/view?usp=sharing',
                'https://drive.google.com/file/d/1pTaaCSQ7KQxlFyoqwP7DUizbQFuopW_y/view?usp=sharing',
                'https://drive.google.com/file/d/1fXAIAusPmSdnaO6IQgwzIp6BNPgTo0kv/view?usp=sharing'
            ]
            temp_folder = os.path.join(temp_folder, 'CompCars')
        elif dataset == Datasets.VEHICLE_ID:
            files.append('https://drive.google.com/file/d/14nMEQr-YrxVV2VIeAeFh-Fngu927SbJr/view?usp=sharing')
        elif dataset == Datasets.VERI:
            files.append('https://drive.google.com/file/d/1K0W_VKBHP-jCkFpLSLKMOWJfkqqxl4Lq/view?usp=sharing')
        elif dataset == Datasets.VRIC:
            files.append('https://drive.google.com/file/d/18HgS75mGDNeoFE4MtawIobov8g8V1_Iq/view?usp=sharing')
        elif dataset == Datasets.CARS196:
            files.append('https://drive.google.com/file/d/1Pa2EO84Xz6sIla3zh3jSWYS2OYsHFEPL/view?usp=sharing')
        elif dataset == Datasets.BOXCARS116K:
            files.append('https://drive.google.com/file/d/17WODwYLOIwZDK-yA1vzxS6shf2XPBAkL/view?usp=sharing')
        for file in files:
            gdown.download(fuzzy=True, url=file, output=temp_folder)
        
    
if __name__=="__main__":
    parser=argparse.ArgumentParser(
        description="Setup dataset by downloading it from remote storage and extract it in a local subfolder")
    parser.add_argument("--dataset-dir","-dd",help="Root directory for the datasets to be extracted into", default="./dataset")
    parser.add_argument("--source","-s",help="Remote location where dataset archives can be downloaded from", default="gdrive", choices=['gdrive', 'local'])
    parser.add_argument("--dataset-name","-n",help="Name of the dataset to download", default="VERI", \
        choices=list(Datasets), type=Datasets.from_string)
    parser.add_argument("--archive-folder", '-a', help="Folder where the downloaded archive files will be stored", default="dataset_source")
    
    args=parser.parse_args()
    
    # download the archive
    if args.source in ['gdrive']:
        download_dataset(args.datasetName, args.source, args.archiveFolder)
    # extract the archive
    if args.datasetDir:
        setup_dataset(args.datasetName, args.archiveFolder, args.datasetDir)