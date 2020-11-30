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
        "setuptools_scm",
    ],
    # here are optional dependencies (as "tag" : "dependency spec")
    extras_require={"docs": docs_require},
    use_scm_version={"write_to": os.path.join("ShowerModel", "_version.py")},
)
