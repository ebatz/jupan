# Jupyter Analysis Notebook for Lattice QCD Spectroscopy

## Introduction

This package contains utilities and a frontend for the determination of spectra and scattering amplitudes from Lattice QCD correlation functions. A Jupyter notebook is used to illustrate the analysis choices leading to the spectrum.

## Setup
### Prerequisites

The following analysis libraries need to be accessible for all features of this code to work properly:

* Eigen3
* [Minuit2](https://github.com/GooFit/Minuit2)
* [Pybind11](https://github.com/pybind/pybind11)
* [pythib](https://github.com/ebatz/pythib)

In addition,

* Jupyter
* ipywidgets
* ipympl

are required for the Jupyter notebook. If the analysis widgets do not show any plots, calling

`jupyter nbextension enable --py --sys-prefix ipympl`

may solve the issue.

### Installation

Only manual installation is available at this point.

1. Build analysis kernel in cpp/ using the build script provided. Change paths as necessary. This should produce a Python module called cppana in the same directory.
2. Launch the Jupyter notebook provided and adjust the path to pythib.

## Usage

Evaluating the cells in the Jupyter notebook from top to bottom goes through the following list of tasks,

1. Determine single-hadron spectra, e.g. the pion mass.
2. Determine multi-hadron spectra for the relevant irreps.
3. Compute the scattering amplitude from the spectrum, and perform fits to customizable K matrix parametrizations.
4. Determine matrix elements of the vector current.
5. Compute the timelike pion form factor.

The spectroscopy is performed in a widget that allows one to explore the data interactively. Various kinds of diagnostic output is provided between those tasks.

## Further reading

The physics results of this analysis are described in the following references
