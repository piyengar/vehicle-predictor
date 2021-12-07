# vehicle-predictor

## Project structure
The following folders exist in the project
- framework - The framework module 
- experiments - The experiments module
  - color - code for training color based models
  - vtype - code for training type based models
  - brand - code for training brand based models
<!-- - test - pytest test cases should be stored here -->
- Experiment_framework.ipynb - A notebook that shows examples on how to use the experiment framework

## Getting started
- The project requires python >= 3.6 and the project dependencies are defined in the setup.py file.
- Working in a virtual environment should be the preferred approach. You can use any method to setup one - conda, venv, etc
- The dependencies can be installed with the command below executed from the project root dir
    `pip install -e .`
- Dataset setup
  - Local Environment - The dataset files need to be extracted into the `dataset` folder
  - Colab - The notebooks contain the steps to download the compressed files from google drive and expand it into the `dataset` folder
  - The framework provides an API to conveniently download the dataset archive from google drive and setup the downloaded archives
  - The `setup_dataset.py` script provides a cli to invoke the setup API
- Experiment Framework
  - The experiment framework organizes the dataset loading, model training and evaluation logic. 3 experiments for training models to classify vehicle color, type and brand are present inside the experiments folder. They give good examples on how the framework API can be used
  - The framework is built on the `pytorch-lightning` framework 
  - The `BaseExperiment` class provides the structure that can be extended in child classes. 
    - The six main operations are 
      - `train_stats` - Get the dataset statistics of classes in the train set
      - `test_stats` - Get the dataset statistics of classes in the test set
      - `tune` - Run the learning rate finder algorithm
      - `train` - Train the model using the provided train_dataset parameter
      - `predict` - Run predictions on the specified `test_dataset` and store the results
      - `evaluate` - Evaluate the accuracy, precision, recall, f1 and confusion matrix for the provided prediction files
    - The class exposes the default CLI args that can be used, refer to `train_brand.py` for how to use it. 

  - The `BaseModel` class has been configured to train models with the following architectures
    - resnet18
    - resnet50
    - resnet152
    - squeezenet
    - mobilenetv3-small
    - efficientnet-b0

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
| VehicleID       | 7.9 |
| VRIC            | 0.3 |

## Metadata available on different datasets
Not all datasets have all the attributes. See below table for details
| Dataset         |Color|Type|Brand|
|-----------------|----:|----:|----:|
| VeRi_with_plate | Y   |    Y|    N|
| CompCars        | Y   |    Y|    Y|
| Cars196         | N   |    Y|    Y|
| BoxCars116k     | N   |    Y|    Y|
| VehicleID       | Y   |    N|    Y|
| VRIC            | N   |    N|    N|
