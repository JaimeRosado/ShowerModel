![ShowerModel logo](https://github.com/JaimeRosado/ShowerModel/blob/master/docs/source/ShoweModel_logo_small.jpg)

# ShowerModel
A Python package for modelling cosmic-ray showers, their light production and their detection.

See presentation vieo: https://drive.google.com/file/d/14AGV91mQXDwecGy2qxgNEmWeIcxKy_I0/view?usp=sharing


--------
* Code : https://github.com/JaimeRosado/ShowerModel
* License: GPL-3.0
--------

## Install

* You will need to install anaconda first.

* Create and activate the conda environment:
```
git clone https://github.com/JaimeRosado/ShowerModel.git
cd ShowerModel
conda env create -f environment.yml
conda activate showermodel
```

* To update the environment (e.g. when dependencies get updated), use:
```
conda env update -n showermodel -f environment.yml
```

Install ShowerModel, run the following command from the ShowerModel root directory:
```
conda develop .
```
