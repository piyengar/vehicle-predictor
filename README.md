# vehicle-predictor

## Project structure
The following folders exist in the project
- dataset - code expects datasets will be in subfolders within it, **not tracked by git**
  - The actual sub-folder structure will be specific to each dataset and details can be inferred from the dataset class files
  - Colab - This folder can be generated using code provided in the notebooks
  - Local env - This folder has to be manually created for now. **TODO** abstract out the colab setup script to automate setup for new environments
  - Cloudlab - Should point to a mounted storage location to avoid setup everytime
- predictions - predictions will be stored here, **not tracked by git**
  - Colab - This folder will be symlinked to a mounted google drive folder during setup
  - Cloudlab - Should be symlinked to a mounted storage location 
- checkpoints - training checkpoints will be stored here, **not tracked by git**
  - Colab - This folder will be symlinked to a mounted google drive folder during setup
  - Cloudlab - Should be symlinked to a mounted storage location 
- color - code for training color based models
- vtype - code for training type based models
- test - pytest test cases should be stored here
- Vehicle_color_predictor.ipynb - Notebook to train color models
- Vehicle_type_trainer.ipynb - Notebook to train type models

## Getting started
- The project requires python >= 3.6 and the project dependencies are defined in the setup.py file.
- Working in a virtual environment should be the preferred approach. You can use any method to setup one - conda, venv, etc
- The dependencies can be installed with the command below executed from the project root dir

    `pip install -e .`
- Dataset setup
  - Local Environment - The dataset files need to be extracted into the `dataset` folder
  - Colab - The notebooks contain the steps to download the compressed files from google drive and expand it into the `dataset` folder

- Run the notebook. These broad steps are included in the notebooks
  - Prepare dataset
  - Train model
  - View training logs/charts
  - Run predictions on test sets and save them to files
  - Evaluate predictions from the files saved in the previous step
## Dataset Sizes for reference
The storage requirements (when expanded) for the datasets used for this project are given below:

| Dataset         |  GB |
|-----------------|----:|
| VeRi_with_plate | 1.1 |
| CompCars        | 2.5 |
| Cars196         | 1.9 |
| BoxCars116k     | 9.2 |

## Metadata available on different datasets
Not all datasets have all the attributes. See below table for details
| Dataset         |Color|Type|
|-----------------|----:|----:|
| VeRi_with_plate | Y   |    Y|
| CompCars        | Y   |    Y|
| Cars196         | N   |    Y|
| BoxCars116k     | N   |    Y|
| VehicleID       | Y   |    N|
| VRIC            | N   |    N|

## Cloudlab 
The experiments can be run on cloudlab using the profile `carzam_training_v1`. This profile has been parameterized with node type options that have GPU access and can be set when starting the experiment. This creates a Ubuntu 20.04 node with CUDA and conda/pytorch installed. 

### Notes for setting up the profile
- Used the default Ubuntu-20 image to create a profile. 
- Disable GUI. This turns off gnome
    ```
    systemctl set-default multi-user.target
    ```
- Install the nvidia driver
    ```
    apt install nvidia-driver-470
    ```
- Install conda. Installed at `/opt/miniconda`
    ```
    wget -O "Miniconda-installer.sh" "https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh"
    bash Miniconda-installer.sh -p /opt/miniconda
    ```