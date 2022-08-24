from setuptools import setup, find_packages
import os

docs_require = [
    "sphinx_rtd_theme",
    "sphinx_automodapi",
    "sphinx",
    "nbsphinx",
    "nbsphinx-link",
    "numpydoc",
    "jupyter",
    "notebook",
    "graphviz",
]

tests_require = [
    "pytest",
    "pytest-cov",
]

setup(
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "toml",
        "ipython",
        "setuptools_scm"
    ],
    package_data={
        'showermodel': [
            'extra/atm_models.toml',
            'extra/averaged_profile_5sh_100TeV.dat',
            'extra/Edep_profile_1000GeV_1000sh_0deg.dat',
            'extra/mean_annual_global_reference_atmosphere.xlsx',
            'extra/showermodel_config.toml',
            'extra/tel_data.toml',
            'showermodel/constants/*.toml',
        ],
    },
    extras_require={
        "docs": docs_require,
        "tests": tests_require,
        "all": docs_require + tests_require,
    },
    use_scm_version={"write_to": os.path.join("showermodel", "_version.py")},
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
