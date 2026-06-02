# SEWAA-forecasts

Scripts for operational rainfall forecasts in the ICPAC region.


## Installation

#### ECMWF data

To download the ECMWF data pre-processed for these scripts you will need access to the
ECMWF machine gbmc. Contact the Oxford or ICPAC team for further information.

#### Conda environment

A conda environment is required, to set this up run the following commands:

	conda config --add channels conda-forge
	conda config --set channel_priority strict
	conda create -n tf215gpu python=3.11
	conda activate tf215gpu
	python -m pip install tensorflow==2.15
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
	conda install conda-forge::curl

Check that tensor flow is working

	python -c "import tensorflow as tf; print(tf.config.list_physical_devices('CPU'))"

#### Forecast scripts

- Go to https://github.com/Fenwick-Cooper/SEWAA-forecasts.
- Click on the big green button labelled "<> Code ".
- Select "Download ZIP".
- Uncompress the SEWAA-forecasts-main.zip.


## Updating an installation

#### For people who use git:

Update using git.

#### For everyone else:

Download the latest version as in the "Forecast scripts" section above.
To keep your data move the following directories:

From:

	SEWAA-forecasts-main-OLD/6h_accumulations/IFS_forecast_data
	SEWAA-forecasts-main-OLD/24h_accumulations/IFS_forecast_data
	SEWAA-forecasts-main-OLD/6h_accumulations/cGAN_forecasts
	SEWAA-forecasts-main-OLD/24h_accumulations/cGAN_forecasts
	SEWAA-forecasts-main-OLD/interface/view_forecasts/data
	
To:

	SEWAA-forecasts-main-NEW/6h_accumulations/IFS_forecast_data
	SEWAA-forecasts-main-NEW/24h_accumulations/IFS_forecast_data
	SEWAA-forecasts-main-NEW/6h_accumulations/cGAN_forecasts
	SEWAA-forecasts-main-NEW/24h_accumulations/cGAN_forecasts
	SEWAA-forecasts-main-NEW/interface/view_forecasts/data


## How to use

#### To make a forecast

Run the script `run_forecast.py` that is located in the SEWAA-forecasts-main directory.
To get usage information run

	python run_forecast.py --help

Running 

	conda activate tf215gpu
	python run_forecast.py

will
1. Download the ECMWF data for 6h accumulations from gbmc.
2. Run the forecasts.
3. Process the forecast data for viewing.

#### To automatically run forecasts

Run the script `start_forecasting.py` that is located in the SEWAA-forecasts-main directory.

Running 

	conda activate tf215gpu
	python start_forecasting.py

will
1. Check every 15 minutes to see if all forecasts from the last two days are done.
2. Complete any forecasts from the last two days that are found to be missing.
3. Delete the forecasts from the last two days, keeping the histogram data for viewing.

#### To view the forecasts

In a terminal change to the SEWAA-forecasts-main directory and run

	python -m http.server 8080
   
Then in a browser window go to the address

> http://localhost:8080/
