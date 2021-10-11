
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

setup(
    name = "stepshift",
    version = "1.2.0",
    python_requires=">=3.8,<3.10",
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
            annotate = True
        )
