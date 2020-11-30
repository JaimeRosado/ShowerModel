from setuptools import setup, find_packages

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
        "ipython"
    ],
    name='ShowerModel',
    version='0.1.0',
    url='https://github.com/JaimeRosado/ShowerModel',
    license='GPL-3.0',
    author='Jaime Rosado',
    author_email='jrosadov@ucm.es',
    description='Modelling cosmic-ray showers, their light production and its detection.',
    # here are optional dependencies (as "tag" : "dependency spec")
    extras_require={"docs": docs_require},
)
