from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules=[
    Extension("white_in_part_cy",
              sources=["white_in_part_cy.pyx"],
              include_dirs=[np.get_include()]
    )
]

setup(
    name = "white_in_part_cy",
    ext_modules = cythonize(ext_modules,
                            annotate=True)
)