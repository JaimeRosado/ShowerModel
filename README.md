# ShowerModel

[![Build Status](https://github.com/JaimeRosado/ShowerModel/workflows/CI/badge.svg?branch=master)](https://github.com/JaimeRosado/ShowerModel/actions?query=workflow%3ACI+branch%3Amaster)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JaimeRosado/ShowerModel/master?filepath=notebooks)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4727298.svg)](https://doi.org/10.5281/zenodo.4727298)
[![pypi](https://img.shields.io/pypi/v/ShowerModel.svg)](https://pypi.org/project/ShowerModel)
[![Documentation Status](https://readthedocs.org/projects/showermodel/badge/?version=latest)](https://showermodel.readthedocs.io/en/latest/?badge=latest)

![ShowerModel logo](docs/logo_showermodel.png)

A Python package for modelling cosmic-ray showers, their light production and their detection.

--------
* Code : https://github.com/JaimeRosado/ShowerModel
* Docs: https://showermodel.readthedocs.io/
* License: GPL-3.0
--------

## Install

* Install miniconda or anaconda first.

### As user
It can be installed by doing:
```
pip install ShowerModel
```
If `pip install ShowerModel` fails as it is, you probably need to use `--user` option. 
This may happen in Windows installations.
```
pip install --user ShowerModel
```

Although it is optional, it is recommended to create a dedicated conda virtual environment.

### As developer

* Create and activate the conda environment:
```
git clone https://github.com/JaimeRosado/ShowerModel.git
cd ShowerModel
conda env create -f environment.yml
conda activate showermodel
```

* To update the environment when dependencies get updated use:
```
conda env update -n showermodel -f environment.yml
```

To install `ShowerModel`, run the following command from the ShowerModel root directory
where the `setup.py` file is located:
```
pip install -e .
```

Test your installation by running any of the notebooks in this repository.
Otherwise, open an Issue with your error.


## Further information
See our presentation video and poster from ADASS XXX conference https://adass2020.es/:
* https://drive.google.com/file/d/14AGV91mQXDwecGy2qxgNEmWeIcxKy_I0/view?usp=sharing
* https://adass2020.es/static/ftp/P4-176/P4-176.pdf
