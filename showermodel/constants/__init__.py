# coding: utf-8

from scipy import constants as ct
import toml
from pathlib import Path

# Configuration file with default parameters of classes
try:
    config = toml.load("./showermodel_config.toml")  # User defined
except Exception:
    config_file = Path(__file__).parent.joinpath("./showermodel_config.toml")
    config = toml.load(config_file)
    
# Atmospheric model parameters
try:
    atm_models = toml.load("./atm_models.toml")  # User defined
except Exception:
    atm_file = Path(__file__).parent.joinpath('./atm_models.toml')
    atm_models = toml.load(atm_file)

# Fluorescence model parameters
fluo_file = Path(__file__).parent.joinpath("./fluorescence_model.toml")
fluo_model = toml.load(fluo_file)

# Telescope data
try:
    tel_data = toml.load("./tel_data.toml")  # User defined
except Exception:
    tel_file = Path(__file__).parent.joinpath('./tel_data.toml')
    tel_data = toml.load(tel_file)

# Constants
pi = ct.pi  # Pi
M_air = 28.9647  # molar mass of air in g/mol
M_w = 18.01528  # molar mass of water in g/mol
g_cm = 100.*ct.g  # acceleration of gravity in cm/s^2
R_erg = ct.R / ct.erg  # molar gas constant in erg/K/mol
mc2, *foo = ct.physical_constants['electron mass energy equivalent in MeV'] # MeV
alpha = ct.alpha  # fine-structure constant
lambda_r = 36.7  # radiation length in g/cm^2 in air
E_c = 81.0  # critial energy in MeV in air
c_km_us = ct.c / 1.e9  # speed of light in km/us
