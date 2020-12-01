![ShowerModel logo](docs/logo_showermodel.png)

A Python package for modelling cosmic-ray showers, their light production and their detection.

--------
* Code : https://github.com/JaimeRosado/ShowerModel
* Docs: https://jaimerosado.github.io/ShowerModel
* License: GPL-3.0
--------

## Install

* Install miniconda or anaconda first.

### As user

```
SHOWERMODEL_VER=0.1.2
wget https://raw.githubusercontent.com/JaimeRosado/ShowerModel/v$SHOWERMODEL_VER/environment.yml
conda env create -n showermodel -f environment.yml
conda activate showermodel
pip install ShowerModel==$SHOWERMODEL_VER
rm environment.yml
```

### As developer

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

To install ShowerModel, run the following command from the ShowerModel root directory:
```
pip install -e .
```

Installation, versioning and docs-web deploying methods are base on 
the [*ctapipe* repository](https://github.com/cta-observatory/ctapipe).

## Further information
See our poster and presentation video from ADASS XXX conference https://adass2020.es/:
* https://drive.google.com/file/d/14AGV91mQXDwecGy2qxgNEmWeIcxKy_I0/view?usp=sharing
* https://adass2020.es/static/ftp/P4-176/P4-176.pdf
