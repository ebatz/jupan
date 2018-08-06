#!/usr/bin/env bash

PYBIND_PATH=/path/to/pybind11/
EIGEN_PATH=/path/to/eigen3/
MINUIT_PATH=/path/to/Minuit2/

g++ -O3 -Wall -shared -std=c++11 -fPIC `python3-config --includes` -I${EIGEN_PATH} -I${PYBIND_PATH}/include -I${MINUIT_PATH}/include/Minuit2/ -L${MINUIT_PATH}/lib/ bind.cc -lMinuit2 -o cppana`python3-config --extension-suffix`
