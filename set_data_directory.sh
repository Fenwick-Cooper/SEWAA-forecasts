#!/usr/bin/env bash

# Where your forecast data is stored:
data_directory="/Users/cooperf/Documents/WFP/Operational/Madagascar_north-forecasts_data"


# Create the data directories
echo Creating directories

mkdir ${data_directory}
mkdir ${data_directory}/6h_accumulations
mkdir ${data_directory}/6h_accumulations/IFS_forecast_data
mkdir ${data_directory}/6h_accumulations/cGAN_forecasts

mkdir ${data_directory}/24h_accumulations
ln -F -s ${data_directory}/6h_accumulations/IFS_forecast_data ${data_directory}/24h_accumulations/IFS_forecast_data
mkdir ${data_directory}/24h_accumulations/cGAN_forecasts

mkdir ${data_directory}/interface/
mkdir ${data_directory}/interface/view_forecasts
mkdir ${data_directory}/interface/view_forecasts/data
mkdir ${data_directory}/interface/ensemble_logistic_regression
mkdir ${data_directory}/interface/ensemble_logistic_regression/ELR_predictions


# Create the links:
echo Creating links

# 6h IFS data
ln -F -s ${data_directory}/6h_accumulations/IFS_forecast_data 6h_accumulations/IFS_forecast_data

# 6h forecasts
ln -F -s ${data_directory}/6h_accumulations/cGAN_forecasts 6h_accumulations/cGAN_forecasts

# 24h IFS data
ln -F -s ${data_directory}/24h_accumulations/IFS_forecast_data 24h_accumulations/IFS_forecast_data

# 24h forecasts
ln -F -s ${data_directory}/24h_accumulations/cGAN_forecasts 24h_accumulations/cGAN_forecasts

# Forecast histograms
ln -F -s ${data_directory}/interface/view_forecasts/data interface/view_forecasts/data

# ELR forecasts
ln -F -s ${data_directory}/interface/ensemble_logistic_regression/ELR_predictions interface/ensemble_logistic_regression/ELR_predictions


# Done
echo Done!