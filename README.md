# vehicle-predictor

## Project folder structure
The following folders exist in the project
- dataset - code expects datasets will be in subfolders within it, **not tracked by git**
  - The actual sub-folder structure will be specific to each dataset and details can be found inferred from the dataset class files
- predictions - predictions will be stored here, **not tracked by git**
- checkpoints - training checkpoints will be stored here, **not tracked by git**
- color - code for training color based models
- vtype - code for training type based models
- test - pytest test cases should be stored here

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

