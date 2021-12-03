import os
import subprocess
import argparse
from zipfile import ZipFile
from enum import Enum, auto
import gdown
from .datasets import Datasets

def ensure_folder(path: str):
    if (os.path.exists(path) and os.path.isdir(path)) or not os.path.exists(path):
        return path if path.endswith(os.path.os.sep) else f"{path}{os.path.sep}"
    else:
        raise NotADirectoryError(path)

def setup_dataset(dataset: Datasets, dataset_src_root:str= 'dataset_source', dest_dir:str = 'dataset'):
    if dataset == Datasets.COMP_CARS:
        os.makedirs(os.path.join(dest_dir), exist_ok=True)
        archive_path = os.path.join(dataset_src_root, 'CompCars', 'sv_data.zip')
        temp_file = os.path.join(dest_dir, 'cc_combined.zip')
        subprocess.run(['zip', '-F', archive_path,  '-b', dest_dir, '--out', temp_file])
        PW = os.environ.get('VP_COMP_CARS_PASS', 'dummy_pass')
        subprocess.run(['unzip', '-P', PW, '-d', os.path.join(dest_dir, 'CompCars'), temp_file])
        os.remove(temp_file)
    elif dataset == Datasets.VEHICLE_ID:
        os.makedirs(os.path.join(dest_dir), exist_ok=True)
        archive_path = os.path.join(dataset_src_root, 'VehicleID_V1.0.zip')
        PW = os.environ.get('VP_VEHICLE_ID_PASS', 'dummy_pass')
        subprocess.run(['unzip', '-P', PW, '-d', dest_dir, archive_path])
        os.rename(os.path.join(dest_dir, "VehicleID_V1.0"), os.path.join(dest_dir, "VehicleID"))
    elif dataset == Datasets.VERI:
        os.makedirs(os.path.join(dest_dir), exist_ok=True)
        archive_path = os.path.join(dataset_src_root, 'VeRi_with_plate.zip')
        subprocess.run(['unzip', '-d', dest_dir, archive_path])
    elif dataset == Datasets.CARS196:
        os.makedirs(os.path.join(dest_dir, 'Cars196'), exist_ok=True)
        archive_path = os.path.join(dataset_src_root, 'Cars196.zip')
        # print(archive_path)
        # print(" ".join(['unzip', '-d', os.path.join(dest_dir, 'Cars196'), archive_path]))
        subprocess.run(['unzip', '-d', os.path.join(dest_dir, 'Cars196'), archive_path])
    elif dataset == Datasets.BOXCARS116K:
        os.makedirs(os.path.join(dest_dir), exist_ok=True)
        archive_path = os.path.join(dataset_src_root, 'BoxCars116k.zip')
        subprocess.run(['unzip', '-d', dest_dir, archive_path])
    elif dataset == Datasets.VRIC:
        os.makedirs(os.path.join(dest_dir, 'VRIC'), exist_ok=True)
        archive_path = os.path.join(dataset_src_root, 'VRIC.zip')
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
            gdown.download(fuzzy=True, url=file, output=ensure_folder(temp_folder))
