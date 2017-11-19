from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules=[
    Extension("gray_conv_cy",
              sources=["gray_conv_cy.pyx"],
              include_dirs=[np.get_include()]
    )
]

setup(
    name = "find_close_cy",
    ext_modules = cythonize(ext_modules,
                            annotate=True)
)