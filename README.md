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

The `Pipeline` folder contains the analysis pipeline itself. It has several subfolders:
- `Data` contains mock Gaia DR4 AL scan data, without Gaussian noise added. To illustrate the format of the data, we have inserted two mock data file containing AL coordinate data for 100 source trajectories each, with 199 trajectories being entirely free, and the 200th being significantly perturbed by lensing. Note that the size and number of data files is drastically smaller than the mock catalogs we used in [2212.XXXXX](https://arxiv.org/abs/2212.XXXXX), which consist of ~3000 seperate data files containing ~500,000 source trajectories each.
- `SourceInfo` contains two types of files. The first ends with `seeds` and simply contains random seeds lists corresponding to the mock sources in `Data` in order to consistently generate the same Gaussian noise for each source trajectory. The other type of file begins with `gaia_info` and contains information about the sources in `Data` like G magntiude, distance, and all other information that is not the source trajectory.
- `Results` contains the results of the analysis. This folder has several subfolders. `FreeScipy` and `FreeMultinest` contain the results from fitting the free model to the data using SciPy and PyMultinest, respectively. These results include source IDs, best fit parameters, and test statistics. The `FreePostsamples` folder contains the MCMC generated postsamples (covariance information) for each significant fit. The other six folders contain the same information, but for the acceleration and blip models.
- `Analysis` contains the core analysis scripts that are executed to run the analysis. It contains several subscripts. The file`analysis_fcns.py` contains various helper functions for executing the data analysis, like test statistic calculations.

![RingFlux](/PaperPlots/pipeline.png "A flowchart depiction of the analysis pipeline.")


`Pipeline` also contains several helper scripts and data used by both the analysis software and the mock catalog generator. These are:
- `dynamics_fcns.py`: functions for modeling astrometric source and lens trajectories, with and without lensing;
- `bh_prior_fcns.py`: astrophysical black hole prior functions based on most recent observations;
- `dm_prior_fcns.py`: compact dark matter prior functions based on a NFW profile;
- `rotation.py`: functions for performing astrophysical coordinate transformations;
- `coordinate_error.csv`: data for mapping the G magnitude of each source to a Gaussian error.

## PaperPlots

The `PaperPlots` folder contains the Jupyter Notebook scripts used to generate the plots shown in [2212.XXXXX](https://arxiv.org/abs/2212.XXXXX).
