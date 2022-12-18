# BlipFinder
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2212.XXXXX%20-green.svg)](https://arxiv.org/abs/2212.XXXXX)

This GitHub page is the home of BlipFinder; the analysis pipeline and mock catalog generator described in [2212.XXXXX](https://arxiv.org/abs/2212.XXXXX) capable of searching for dark compact objects in the Milky Way through astrometric lensing in real or mock Gaia DR4 data.

![RingFlux](/PaperPlots/source_dynamics_plot.png "An example of astrometric lensing. A gravitional lens (purple) approaches a source freely propagating across the sky (red). If the lens is in the foreground relative to the faraway observer, the lens will deflect the apparent path of the star (black).")

If this pipeline is used in published work, please cite [2212.XXXXX](https://arxiv.org/abs/2212.XXXXX).

## Authors

- I-Kai Chen
- Marius Kongsore
- Ken Van Tilburg

## Pipeline

The `Pipeline` folder contains the analysis pipeline itself. It also contains several helper scripts and data used by both the analysis software and the mock catalog generator. These are:
- `dynamics_fcns.py`: functions for modeling astrometric source and lens trajectories, with and without lensing;
- `bh_prior_fcns.py`: astrophysical black hole prior functions based on most recent observations;
- `dm_prior_fcns.py`: compact dark matter prior functions based on a NFW profile;
- `rotation.py`: functions for performing astrophysical coordinate transformations;
- `coordinate_error.csv`: data for mapping the G magnitude of each source to a Gaussian error.
