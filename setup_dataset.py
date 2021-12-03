import os
import subprocess
import argparse
from zipfile import ZipFile
from enum import Enum, auto
import gdown
from framework.datasets import Datasets
from framework.setup_dataset import setup_dataset, download_dataset

COMMAND_DOWNLOAD = "download"
COMMAND_SETUP = "setup"


def ensure_folder(path: str):
    if (os.path.exists(path) and os.path.isdir(path)) or not os.path.exists(path):
        return path if path.endswith(os.path.os.sep) else f"{path}{os.path.sep}"
    else:
        raise NotADirectoryError(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool to download and setup vehicle datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--dataset-name",
        "-n",
        help="Name of the dataset to download",
        default=Datasets.VERI,
        choices=list(Datasets),
        type=Datasets.from_string,
        dest="dataset_name",
    )

    subparsers = parser.add_subparsers(
        help="Valid sub-commands", dest="command", required=True, metavar="command"
    )

    dl_parser = subparsers.add_parser(
        COMMAND_DOWNLOAD,
        parents=[parent_parser],
        help="Download the specified dataset from a remote location. Currently supports only the default google drive location",
    )
    dl_parser.add_argument(
        "--source",
        "-s",
        help="Remote location where dataset archives can be downloaded from",
        default="gdrive",
        choices=["gdrive"],
    )
    dl_parser.add_argument(
        "--dest",
        "-d",
        help="Folder where the downloaded archive files will be stored",
        default="./dataset_source",
        dest="archive_folder",
        type=ensure_folder
    )

    setup_parser = subparsers.add_parser(
        COMMAND_SETUP,
        parents=[parent_parser],
        help="Extract the dataset archive into the specified directory",
    )
    setup_parser.add_argument(
        "--source",
        "-s",
        help="Folder where the archive files are stored",
        default="./dataset_source",
        dest="archive_folder",
        type=ensure_folder
    )
    setup_parser.add_argument(
        "--dest",
        "-d",
        dest="dataset_dir",
        help="Root directory for the datasets to be extracted into",
        default="./dataset/",
        type=ensure_folder
    )
    
    setup_parser.add_argument(
        "--password",
        "-p",
        help="Password for extracting dataset. Used for COMP_CARS and VEHICLE_ID",
        default="",
        type=str,
    )
    

    args = parser.parse_args()
    # print(args)
    # download the archive
    if args.command == COMMAND_DOWNLOAD:
        download_dataset(args.dataset_name, args.source, args.archive_folder)
    # extract the archive
    elif args.command == args.command == COMMAND_SETUP:
        setup_dataset(args.dataset_name, args.archive_folder, args.dataset_dir, password=args.password)
