# SEWAA-forecasts

Scripts for operational rainfall forecasts, using the cGAN model, in the South East Africa region.


## Installation

#### Installing Git and Conda

To use these scripts, Git and Conda are required. These are standard tools that can be downloaded from the internet, and have detailed instructions for downloading and installing them.

For Windows, you can download Git from [here](https://gitforwindows.org/) and Conda from [here](https://repo.anaconda.com/archive/Anaconda3-2026.07-1-Windows-x86_64.exe).

For Mac, you should already have Git installed (try `git version` in terminal to see if it is there already), but it can be downloaded [here](https://sourceforge.net/projects/git-osx-installer/files/git-2.23.0-intel-universal-mavericks.dmg/download?use_mirror=autoselect) if needed. You can download Conda from [here](https://repo.anaconda.com/archive/Anaconda3-2026.07-1-MacOSX-arm64.pkg).

For Linux, you can download Git by running `sudo apt-get install git-all`, and download Conda from [here](https://www.anaconda.com/download/success?reg=skipped). 

#### ECMWF data

Data from the [European Centre for Medium-range Weather Forecasts (ECMWF)](ecmwf.int) are used as inputs into the cGAN model.

The scripts are currently set up to download data from Oxford's server (rain.physics.ox.ac.uk), with a backup server which is accessed if those downloads are unsuccessful.

As ECMWF transitions to [open data](https://www.ecmwf.int/en/forecasts/datasets/open-data), we expect to transition to downloading data directly from ECMWF. This is anticipated to happen by the end of 2026. 

#### Conda environment

To run the scripts, a specific conda environment is required. The default name (as per the instructions below) is _sewaa-forecasts_

To set this up, run the following commands (you may have to input _y_ or _yes_ for some commands to proceed):

First, set up the environment:

	conda config --add channels conda-forge
	conda config --set channel_priority strict
	conda create -n sewaa-forecasts python=3.11

Then activate the environment:

	conda activate sewaa-forecasts

Install tensorflow:

	python -m pip install tensorflow==2.15

Then install other packages that are required (you can copy and paste these all in one go if you want):

	pip install numba
	pip install matplotlib
	pip install seaborn
	pip install cartopy
	pip install jupyter
	pip install xarray
	pip install netcdf4
	pip install scikit-learn
	pip install cfgrib
	pip install dask
	pip install tqdm
	pip install properscoring
	pip install climlab
	pip install iris
	pip install ecmwf-api-client
	pip install xesmf
	pip install flake8
	pip install regionmask
	pip install schedule

Check that tensor flow is working:

	python -c "import tensorflow as tf; print(tf.config.list_physical_devices('CPU'))"

This should return something like the below, and not raise an error:

	[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]

#### Downloading the forecast scripts

We recommend using Git to ensure you can easily update your scripts when the code is updated.

In a terminal, navigate to the folder where you would like to place the forecasting scripts
"Clone" the GitHub repository using the following command:

	git clone -b South_East_Africa https://github.com/Fenwick-Cooper/SEWAA-forecasts.git

This should download the code for you; check the code has downloaded successfully using:

	git status

This should return:

	On branch South_East_Africa
	Your branch is up to date with 'origin/South_East_Africa'.

You have now successfully downloaded the code and can start forecasting!

Alternatively, if you are not able to use Git, you can download the code as a _.zip_ file.

- Go to https://github.com/Fenwick-Cooper/SEWAA-forecasts/South_East_Africa
- Click on the big green button labelled "<> Code "
- Select "Download ZIP"
- Uncompress the SEWAA-forecasts-main.zip into the folder where you would like to place the forecasting scipts

## Updating an installation

#### For people who use Git:

Navigate to the folder _SEWAA-forecasts_. Once inside the folder, run:

	git pull

This will "pull" all updates from the GitHub, whilst maintaining your data in the _data_ directory.

#### For people downloading as a _.zip_:

Download the latest version by following the instructions above.
To keep your data move the following directories:

Copy all folders from the *new* _SEWAA-forecasts_ folder into the current _SEWAA-forecasts_ folder, except _data_.

## How to make forecasts

#### To make a single forecast

Run the script `run_forecast.py` that is located in the SEWAA-forecasts directory.
To get usage information run:

	python run_forecast.py --help

This shows the options you can change when running the script:



To run the script using the default options:

	conda activate sewaa-forecasts
	python -u run_forecast.py 

This will
1. Download the ECMWF data for 6h and 24h accumulations from the Oxford server.
2. Run the forecasts.
3. Process the forecast data for viewing.

You can also alter the default options, including:

- To change the accumulation to just 24 hours, use `--accumulation 24h`; to change the accumulation to just 6 hours, use `--accumulation 6h`; to change the accumulation to just 7 days, use `--accumulation 7d`

- To run forecasts for a different date, use `--date YYYYMMDD` where YYYY is the year, MM is the month and DD is the day you want to run forecasts e.g., `--date 20200715`
- By default, forecast files are delted (only histograms are saved for plotting). To keep the raw forecasts, use `--keep-forecasts`

- By default, forecasts are produced with 1000 ensemble members. To change this to X ensemble members, use `--n_ens X` e.g., `--n_ens 50` for 50 ensemble members. Using ensemble members makes the forecast quicker to run, but gives a less good forecast.  

Add these arguments after `python -u run_forecast.py`, with a space between each one. There are a few other options you can change -- see the help for details.
#### To automatically run forecasts

Run the script `start_forecasting.py` that is located in the SEWAA-forecasts-main directory.

Running 

	conda activate sewaa-forecasts
	python -u start_forecasting.py

will
1. Check every 15 minutes to see if all forecasts from the last two days are done.
2. Complete any forecasts from the last two days that are found to be missing.
3. Delete the forecasts from the last two days, keeping the histogram data for viewing.

#### To view the forecasts

In a terminal change to the SEWAA-forecasts directory and run

	python -m http.server 8080
   
Then in a browser window go to the address

> http://localhost:8080/
