import pyximport
import numpy as np

pyximport.install(setup_args={
    "include_dirs":np.get_include(), 
    "compiler_directives":{
        "language_level":"3str"
        }
    })

from .cast import time_unit_feature_cube 
from .stepshift import *
from .bootstrap import *
