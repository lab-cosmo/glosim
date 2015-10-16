#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

setup(name             = "rndperm",
      version          = "0.1.3",
      description      = "Calculates the permanent of a Numpy matrix",
      author           = "Sandip De",
      author_email     = "pete.shadbolt@gmail.com",
      maintainer       = "pete.shadbolt@gmail.com",
      url              = "https://github.com/peteshadbolt/permanent",
      packages         = ["mypermanent"],
      ext_modules      = [
          Extension(
              'mypermanent.mypermanent', ['./src/mypermanent.cpp'],
              extra_compile_args=["-Ofast", "-march=native","-std=c++11"],
              include_dirs=[numpy.get_include()]),
      ],

)

