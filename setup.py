from setuptools import setup, find_packages
import os

docs_require = [
    "sphinx_rtd_theme",
    "sphinx_automodapi",
    "sphinx",
    "nbsphinx",
    "numpydoc",
    "jupyter",
    "notebook",
    "graphviz",
]

setup(
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "ipython",
        "setuptools_scm>=3.4"
    ],
    package_data={
        'showermodel': [
            'extra/averaged_profile_5sh_100TeV.dat',
            'extra/Edep_profile_1000GeV_1000sh_0deg.dat'
        ],
    },
    # here are optional dependencies (as "tag" : "dependency spec")
    extras_require={"docs": docs_require},
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
