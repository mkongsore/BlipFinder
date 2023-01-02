# BlipFinder
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2301.XXXXX%20-green.svg)](https://arxiv.org/abs/2301.XXXXX)

This GitHub page is the home of BlipFinder; the analysis pipeline and mock catalog generator described in [2301.XXXXX](https://arxiv.org/abs/2301.XXXXX) capable of searching for dark compact objects in the Milky Way through astrometric lensing in real or mock Gaia DR4 data.

![RingFlux](/PaperPlots/source_dynamics_plot.png "An example of astrometric lensing. A gravitional lens (purple) approaches a source freely propagating across the sky (red). If the lens is in the foreground relative to the faraway observer, the lens will deflect the apparent path of the star (black).")

If this pipeline is used in published work, please cite [2301.XXXXX](https://arxiv.org/abs/2212.XXXXX).

## Authors

- I-Kai Chen
- Marius Kongsore
- Ken Van Tilburg

## Pipeline

The `Pipeline` folder contains the analysis pipeline itself. It has several subfolders:
- `Data` contains mock Gaia DR4 AL scan data, without Gaussian noise added. To illustrate the format of the data, we have inserted two mock data file containing AL coordinate data for 100 source trajectories each, with 199 trajectories being entirely free, and the 200th being significantly perturbed by lensing. Note that the size and number of data files is drastically smaller than the mock catalogs we used in [2301.XXXXX](https://arxiv.org/abs/2301.XXXXX), which consist of ~3000 seperate data files containing ~500,000 source trajectories each.
- `SourceInfo` contains two types of files. The first ends with `seeds` and simply contains random seeds lists corresponding to the mock sources in `Data` in order to consistently generate the same Gaussian noise for each source trajectory. The other type of file begins with `gaia_info` and contains information about the sources in `Data` like G magntiude, distance, and all other information that is not the source trajectory.
- `Results` contains the results of the analysis. This folder has several subfolders. `FreeScipy` and `FreeMultinest` contain the results from fitting the free model to the data using SciPy and PyMultinest, respectively. These results include source IDs, best fit parameters, and test statistics. The `FreePostsamples` folder contains the nested sampling generated postsamples (covariance information) for each significant fit. The other six folders contain the same information, but for the acceleration and blip models.
- `Analysis` contains the core analysis scripts that are executed to run the analysis. It contains several subscripts. The file `analysis_fcns.py` contains various helper functions for executing the data analysis, like test statistic calculations. `constraint_fcns.py` contains the constraints that are imposed on SciPy when fitting the free, acceleration, and blip model to the data. `free_fit.py` reads trajectory data and source information from the `Data` and `SourceInfo` folders and then fits the free model to the data. The results are then saved to `FreeScipy` under the `Results` folder. `free_multinest.py` takes the results from `FreeScipy` and refit to the data, while also imposing the 5 sigma cutoff in free log likelihood. The results from fitting the model using nested sampling are then saved to `FreeMultinest` and `FreePostsamples` in the `Results` folder. The four similarly named scripts function that same way, but for the acceleration and blip models. Finally, the `significant_plot.ipynb` displays the results of the analysis.
- `mock_generator` contains the files to generate the null catalog and also the lensed catalog. It contains several subscripts. The file `null_trajectory.py` computes and generate the one dimensional AL trajectory of the sources. The file `lens_generator_x1.py` generates and astrophysical BHs with the prioris in `bh_prior_fcns.py` and save their parameters. The file `lens_correction_x1.py` takes the parameter from the BHs we generated using `lens_generator_x1.py` and then updates the null trajectories with the pertubed ones.

`Pipeline` also contains several helper scripts and data used by both the analysis software and the mock catalog generator. These are:
- `dynamics_fcns.py`: functions for modeling astrometric source and lens trajectories, with and without lensing;
- `bh_prior_fcns.py`: astrophysical black hole prior functions based on most recent observations;
- `dm_prior_fcns.py`: compact dark matter prior functions based on a NFW profile;
- `rotation.py`: functions for performing astrophysical coordinate transformations;
- `coordinate_error.csv`: data for mapping the G magnitude of each source to a Gaussian error.
- `yellin.py`: contains the function that compute the exclusion confidence level following the optimal interval method specified in [0203002] (https://arxiv.org/abs/physics/0203002)

To run the analysis, data and source information must be placed with the correct format in the `Data` and `SourceInfo` folders (see the example files in these folders for the correct formatting). Then, scripts must be run in the following order:
1. `free_fit.py`;
2. `free_multinest.py`;
3. `accel_fit.py`;
4. `accel_multinest.py`;
5. `blip_fit.py`;
6. `blip_multinest.py`,

where the only line that needs to be changed in each script is the `job_idx` variable. This indexes over all files in the `Data` folder and is by default set to 0. To parallelize the pipeline on a slurm-based computing cluster, one may replace `Data` with int(sys.argv[1]) and instead batch submit a job array, e.g. via the script

````
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1GB
#SBATCH --job-name=gaia-free-fit
#SBATCH --mail-type=NONE
#SBATCH --array=0-1

module load python/intel/3.8.6
python free_fit.py ${SLURM_ARRAY_TASK_ID}`
````

with the exact formatting depending on the cluster in use.

Once 1-6 have been run, the pipeline's output can then be viewed using the `significant_plot.ipynb` Jupyter notebook in the `Analysis` folder.

![RingFlux](/PaperPlots/blipfinder-pipeline.png "A flowchart depiction of the analysis pipeline.")

## PaperPlots

The `PaperPlots` folder contains the Jupyter Notebook scripts used to generate the plots shown in [2301.XXXXX](https://arxiv.org/abs/2301.XXXXX).
