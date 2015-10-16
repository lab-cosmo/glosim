#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

setup(name             = "mcpermanent",
      version          = "0.0.1",
      description      = "Calculates the permanent of a Numpy matrix upto given accuracy using random montecarlo algorithm",
      author           = "Sandip De, Michele Ceriotti",
      author_email     = "1sandipde@gmail.com",
      maintainer       = "1sandipde@gmail.com",
      url              = "https://github.com/sandipde/MCpermanent",
      packages         = ["mcpermanent"],
      ext_modules      = [
          Extension(
              'mcpermanent.mcpermanent', ['./src/mcpermanent.cpp'],
              extra_compile_args=["-O3","-Ofast", "-march=native","-std=c++11"],
              include_dirs=[numpy.get_include()]),
      ],

)

