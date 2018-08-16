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

Evaluating the cells in the Jupyter notebook from top to bottom (with Shift+Enter) goes through the following list of tasks,

1. Determine single-hadron spectra, e.g. the pion mass.
2. Determine multi-hadron spectra for the relevant irreps.
3. Compute the scattering amplitude from the spectrum, and perform fits to customizable K matrix parametrizations.
4. Determine matrix elements of the vector current.
5. Compute the timelike pion form factor.
6. Perform fits to dispersive parametrizations of the form factor.

Various kinds of diagnostic output are provided between those tasks, some of which can be skipped without impeding the analysis workflow.

The spectroscopy is performed in a widget that allows one to explore the data interactively. The widget shows a list of available data channels (single-hadrons, irreducible representations etc.) read from the data files. For each data channel, plots detailing various sorts of systematics can be explored interactively. Fit values are chosen by clicking on a data point to select. Clicking on the same data point again unselects.

Each widget has to be closed using the *Save & Close* button before the analysis can proceed. After closing the widget,

1. the analysis state is written to disk,
2. bootstrap samples for the chosen analysis are generated in the background.

In the widget used to extract matrix elements of the vector current, a plateau average is performed instead of using only a single data point. The plateau average region is selected by keeping the mouse button pressed while selecting all (consecutive) time separations to be included. Selecting the same range again deselects.

## Further reading

The physics results of this analysis are described in [1808.05007](https://arxiv.org/abs/1808.05007), which can be cited as

@article{Andersen:2018mau,
      author         = "Andersen, Christian and Bulava, John and {H\"{o}rz}, Ben and
                        Morningstar, Colin",
      title          = "{The $I=1$ pion-pion scattering amplitude and timelike
                        pion form factor from $N_{\rm f} = 2+1$ lattice QCD}",
      year           = "2018",
      eprint         = "1808.05007",
      archivePrefix  = "arXiv",
      primaryClass   = "hep-lat",
      reportNumber   = "CP3-Origins-2018-029 DNRF90 MITP/18-073",
      SLACcitation   = "%%CITATION = ARXIV:1808.05007;%%"
}
