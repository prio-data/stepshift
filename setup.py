
from pathlib import Path
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
        Extension("*", ["stepshift/*.pyx"],
                include_dirs= [
                    np.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
    ]

here = Path(__file__).parent
long_description = (here / "README.md").read_text()

setup(
    name = "stepshift",
    author="peder2911",
    author_email="pglandsverk@gmail.com",
    description = "Implementation of the Views stepshifting modelling framework",
    version = "2.2.4",
    python_requires=">=3.8",
    license_files = ("LICENSE",),
    install_requires=[
        "pandas>=1.3.2",
        "PyMonad>=2.4.0",
        "toolz>=0.11.1",
        "xarray>=0.19.0",
        ],
    packages = find_packages(),
    ext_modules = cythonize(
            extensions,
            compiler_directives={
                    "language_level": "3str",
                }
            ),
    annotate = True,
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://www.github.com/prio-data/stepshift",
        )
