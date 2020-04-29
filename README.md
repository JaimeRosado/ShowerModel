# ToyModel
A Python package for modelling cosmic-ray showers, their light production and their detection

--------
* Code : https://github.com/JaimeRosado/ToyModel
* License: GPL-3.0
--------

## Install

* You will need to install anaconda first.

* Create and activate the conda environment:
```
git clone https://github.com/JaimeRosado/ToyModel.git
cd ToyModel
conda env create -f environment.yml
conda activate toymodel
```

* To update the environment (e.g. when dependencies get updated), use:
```
conda env update -n toymodel -f environment.yml
```

Install ToyModel, run the following command from the ToyModel root directory:
```
conda develop .
```
