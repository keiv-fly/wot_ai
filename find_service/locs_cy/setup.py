from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules=[
    Extension("locs_cy",
              sources=["locs_cy.pyx"],
              include_dirs=[np.get_include()]
    )
]

setup(
    name = "locs_cy",
    ext_modules = cythonize(ext_modules,
                            annotate=True)
)