#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import json
import pathlib
import yaml
from datetime import datetime, date, timedelta
import properscoring as ps

import netCDF4 as nc
import numpy as np
from tensorflow.keras.utils import Progbar

from pyproj import CRS, Transformer

log_precip = True

# Open and parse forecast.yaml
parser = argparse.ArgumentParser()
parser.add_argument(
    "yaml_file",
    nargs="?",
    default="forecast.yaml",
    help="Path to forecast configuration YAML file."
)
parser.add_argument(
    "config_file",
    nargs="?",
    default="config.yaml",
    help="Path to forecast configuration YAML file."
)
args = parser.parse_args()

with open(args.yaml_file, "r") as f:
    try:
        fcst_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)
        raise

# ------------------------------------------------------------
# Select local config for this process
# ------------------------------------------------------------
model_folder = fcst_params["MODEL"]["folder"]
# Open and parse GAN config file
config_path = os.path.join(model_folder, "setup_params.yaml")
with open(config_path, "r") as f:
    try:
        setup_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

local_config_path = setup_params["GENERAL"]["local_config_path"]

# Resolve relative to experiment config file
if not os.path.isabs(local_config_path):
    local_config_path = os.path.join(
        os.path.dirname(os.path.abspath(args.config_file)),
        local_config_path,
    )

os.environ["CGAN_LOCAL_CONFIG"] = local_config_path

data_cfg = setup_params["DATA"]

os.environ["CGAN_ALL_FCST_FIELDS"] = json.dumps(
    data_cfg["all_fcst_fields"]
)

os.environ["CGAN_ACCUMULATED_FIELDS"] = json.dumps(
    data_cfg["accumulated_fields"]
)

os.environ["CGAN_NONNEGATIVE_FIELDS"] = json.dumps(
    data_cfg["nonnegative_fields"]
)

os.environ["CGAN_CROP_TO_BOUNDS"] = str(
    data_cfg["crop_to_bounds"]
)

os.environ["CGAN_BOUNDS"] = ",".join(
    str(x) for x in data_cfg["bounds"]
)

print(f"Experiment config: {os.path.abspath(args.config_file)}")
print(f"Local config:      {local_config_path}")

#Imports
from data import HOURS, fcst_norm, denormalise, load_hires_constants, load_fcst, load_truth_and_mask, load_fcst_norm, crop_to_bounds, bounds
import read_config
from noise import NoiseGenerator
from setupmodel import setup_model

# Setup
read_config.set_gpu_mode()  # set up whether to use GPU, and mem alloc mode
data_paths = read_config.get_data_paths()  # need the constants directory
downscaling_steps = read_config.read_downscaling_factor()["steps"]

checkpoint = fcst_params["MODEL"]["checkpoint"]
fcst_input_folder = fcst_params["INPUT"]["fcst_folder"]
truth_input_folder = fcst_params["INPUT"]["truth_folder"]
mask_input_folder = fcst_params["INPUT"]["mask_folder"]
constants_folder = fcst_params["INPUT"]["constants_folder"]
normalisation_folder = fcst_params["INPUT"]["normalisation_folder"]
output_folder = fcst_params["OUTPUT"]["folder"]
ensemble_members = fcst_params["OUTPUT"]["ensemble_members"]
save_crps_only = fcst_params["OUTPUT"]["save_crps_only"]
leadtime = fcst_params["MODEL"]["leadtime"]
accumulation = fcst_params["MODEL"]["accumulation"]
start_date_in = fcst_params["OUTPUT"]["start_date"]
end_date_in = fcst_params["OUTPUT"]["end_date"]
hour = fcst_params["MODEL"]["hour"]
all_fcst_fields = fcst_params["DATA"]["all_fcst_fields"]
accumulated_fields = fcst_params["DATA"]["accumulated_fields"]
nonnegative_fields = fcst_params["DATA"]["nonnegative_fields"]

local_fcst_norm = load_fcst_norm(year=2018, normalisation_path=normalisation_folder)
assert local_fcst_norm is not None

start_date = date(start_date_in[2], start_date_in[1], start_date_in[0])
end_date = date(end_date_in[2], end_date_in[1], end_date_in[0])

# Open and parse GAN config file
config_path = os.path.join(model_folder, "setup_params.yaml")
with open(config_path, "r") as f:
    try:
        setup_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

mode = setup_params["GENERAL"]["mode"]
arch = setup_params["MODEL"]["architecture"]
padding = setup_params["MODEL"]["padding"]
filters_gen = setup_params["GENERATOR"]["filters_gen"]
noise_channels = setup_params["GENERATOR"]["noise_channels"]
latent_variables = setup_params["GENERATOR"]["latent_variables"]
filters_disc = setup_params["DISCRIMINATOR"]["filters_disc"]
constants_list = setup_params["CONSTANTS"]["constants_list"]

assert mode == "GAN", "standalone forecast script only for GAN, not VAE-GAN or deterministic model"

# Set up pre-trained GAN
weights_fn = os.path.join(model_folder, "models", f"gen_weights-{checkpoint:07}.h5")
input_channels = 2*len(all_fcst_fields)

model = setup_model(mode=mode,
                    arch=arch,
                    downscaling_steps=downscaling_steps,
                    input_channels=input_channels,
                    constant_fields=len(constants_list),
                    filters_gen=filters_gen,
                    filters_disc=filters_disc,
                    noise_channels=noise_channels,
                    latent_variables=latent_variables,
                    padding=padding)
gen = model.gen
print(weights_fn)
gen.load_weights(weights_fn)

network_const_input = load_hires_constants(batch_size=1, constants_path=constants_folder, constants_list=constants_list)  # 1 x lats x lons x 2


def create_output_file(nc_out_path):
    netcdf_dict = {}
    rootgrp = nc.Dataset(nc_out_path, "w", format="NETCDF4")
    netcdf_dict["rootgrp"] = rootgrp
    rootgrp.description = f"GAN {accumulation}-hour rainfall ensemble members."

    # Create output file dimensions
    rootgrp.createDimension("y", len(y))
    rootgrp.createDimension("x", len(x))
    rootgrp.createDimension("time", None)
    rootgrp.createDimension("valid_time", None)

    if not save_crps_only:
        rootgrp.createDimension("member", ensemble_members)

    # Create coordinate variables
    y_data = rootgrp.createVariable("y", "f4", ("y",))
    y_data.units = "meters"
    y_data[:] = y

    x_data = rootgrp.createVariable("x", "f4", ("x",))
    x_data.units = "meters"
    x_data[:] = x

    if not save_crps_only:
        ensemble_data = rootgrp.createVariable("member", "i4", ("member",))
        ensemble_data.units = "ensemble member"
        ensemble_data[:] = range(1, ensemble_members + 1)

    netcdf_dict["time_data"] = rootgrp.createVariable("time", "f4", ("time",))
    netcdf_dict["time_data"].units = "hours since 1900-01-01 00:00:00.0"

    netcdf_dict["valid_time_data"] = rootgrp.createVariable(
        "fcst_valid_time", "f4", ("time", "valid_time")
    )
    netcdf_dict["valid_time_data"].units = "hours since 1900-01-01 00:00:00.0"

    # Ensemble precipitation output
    if not save_crps_only:
        netcdf_dict["precipitation"] = rootgrp.createVariable(
            "precipitation",
            "f4",
            ("time", "member", "valid_time", "y", "x"),
            compression="zlib",
            chunksizes=(1, 1, 1, len(y), len(x)),
        )
        netcdf_dict["precipitation"].units = "mm/h"
        netcdf_dict["precipitation"].long_name = "Precipitation"

    # CRPS output
    netcdf_dict["crps"] = rootgrp.createVariable(
        "crps",
        "f4",
        ("time", "valid_time", "y", "x"),
        compression="zlib",
        chunksizes=(1, 1, len(y), len(x)),
    )
    netcdf_dict["crps"].units = "mm/h"
    netcdf_dict["crps"].long_name = "CRPS for precipitation"

    if radar_centre_lon is not None:
        radar_centre_lat_data = rootgrp.createVariable(
            "radar_centre_lat",
            "f4",
            (),
        )
        radar_centre_lat_data.units = "degrees_east"
        radar_centre_lat_data.long_name = "Central latitude of radar instrument"
        radar_centre_lat_data[:] = radar_centre_lat
        
        radar_centre_lon_data = rootgrp.createVariable(
                "radar_centre_lon",
                "f4",
                (),
            )
        radar_centre_lon_data.units = "degrees_east"
        radar_centre_lon_data.long_name = "Central longitude of radar instrument"
        radar_centre_lon_data[:] = radar_centre_lon

        radar_lat_data = rootgrp.createVariable(
                "radar_lat",
                "f4",
                ("y", "x"),
            )
        radar_lat_data.units = "degrees_east"
        radar_lat_data.long_name = "Central latitude of radar instrument"
        radar_lat_data[:] = radar_lat
        
        radar_lon_data = rootgrp.createVariable(
                "radar_lon",
                "f4",
                ("y", "x"),
            )
        radar_lon_data.units = "degrees_east"
        radar_lon_data.long_name = "Central longitude of radar instrument"
        radar_lon_data[:] = radar_lon
    
    return netcdf_dict

def iter_dates(start, end):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)

## Open first available truth file to get y and x of forecast
example_truth_file = None

for d_tmp in iter_dates(start_date, end_date):
    candidate = os.path.join(
        truth_input_folder,
        str(d_tmp.year),
        f"{d_tmp.strftime('%Y%m%d')}_06.nc"
    )

    if os.path.exists(candidate):
        example_truth_file = candidate
        break

if example_truth_file is None:
    raise FileNotFoundError(
        f"No truth files found between {start_date} and {end_date}"
    )

print(f"Using truth file for grid: {example_truth_file}")

with nc.Dataset(example_truth_file, mode="r") as nc_in:
    y = nc_in["y"][:]
    x = nc_in["x"][:]

    radar_centre_lat, radar_centre_lon = None, None
    if "central_lat" in nc_in.variables:
        radar_centre_lat = nc_in["central_lat"][:]
        radar_centre_lon = nc_in["central_lon"][:]
    if "lat" in nc_in.variables:
        radar_lat = nc_in["lat"][:]
        radar_lon = nc_in["lon"][:]
    elif radar_centre_lat is not None:
        #Calculate radar lat/lon from radar x/y
        # Local azimuthal equidistant projection centred on radar
        radar_crs = CRS.from_proj4(
            f"+proj=aeqd "
            f"+lat_0={radar_centre_lat} "
            f"+lon_0={radar_centre_lon} "
            f"+datum=WGS84 +units=m"
        )
        to_lonlat = Transformer.from_crs(
            radar_crs,
            "EPSG:4326",
            always_xy=True
        )
        # x and y are 1-D arrays in metres
        radar_xx, radar_yy = np.meshgrid(x, y)
        radar_lon, radar_lat = to_lonlat.transform(
            radar_xx,
            radar_yy
        )

#Iterate through all dates that we want to forecast/CRPS
for d in iter_dates(start_date, end_date):

    print(f"\n{'='*70}")
    print(f"Processing forecast date: {d}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # 1. Find this forecast date in the IFS file
    # ------------------------------------------------------------------
    file_name = os.path.join(
        fcst_input_folder,
        str(d.year),
        "tp.nc"
    )

    with nc.Dataset(file_name, mode="r") as nc_in:

        start_times = nc_in["time"][:]
        valid_times = nc_in["fcst_valid_time"][:]

        time_var = nc_in["time"]

        # Only convert valid/unmasked time entries
        valid_indices = np.where(~np.ma.getmaskarray(start_times))[0]

        matching_indices = []

        for idx in valid_indices:

            t = nc.num2date(
                start_times[idx],
                units=time_var.units,
                calendar=getattr(time_var, "calendar", "standard")
            )

            if (
                t.year == d.year
                and t.month == d.month
                and t.day == d.day
            ):
                matching_indices.append(idx)

    # No valid IFS initialisation for this date
    if len(matching_indices) == 0:
        print(
            f"WARNING: No valid IFS forecast found for {d}. "
            f"Skipping date."
        )
        continue

    if len(matching_indices) > 1:
        raise ValueError(
            f"Multiple IFS forecasts found for {d}: "
            f"indices {matching_indices}"
        )

    fcst_idx = matching_indices[0]

    # ------------------------------------------------------------------
    # 2. Get forecast valid time
    # ------------------------------------------------------------------
    valid_time_idx = ([int(leadtime / HOURS)],)

    valid_times_forecast = valid_times[
        fcst_idx,
        valid_time_idx
    ]

    print(
        f"Forecast index = {fcst_idx}, "
        f"valid time shape = {np.shape(valid_times_forecast)}"
    )

    # ------------------------------------------------------------------
    # 3. Load ALL forecast fields BEFORE creating output file
    #
    # If any field does not contain this forecast date, skip the entire
    # forecast date cleanly.
    # ------------------------------------------------------------------
    field_arrays = []

    try:

        for field in all_fcst_fields:

            print(f"Loading {field} for {d}")

            data = load_fcst(
                field,
                d.strftime('%Y%m%d'),
                hour=hour,
                leadtime=leadtime,
                accumulation=accumulation,
                log_precip=log_precip,
                norm=True,
                fcst_path=fcst_input_folder,
                fcst_norm_dict=local_fcst_norm
            )

            field_arrays.append(data)

    except Exception as e:

        print(f"WARNING: {e}")
        print(f"Skipping forecast date {d}")
        continue

    # ------------------------------------------------------------------
    # 4. All forecast inputs are valid.
    #    Now create output NetCDF.
    # ------------------------------------------------------------------
    pathlib.Path(output_folder).mkdir(
        parents=True,
        exist_ok=True
    )

    if not save_crps_only:

        nc_out_path = os.path.join(
            output_folder,
            f"GAN_fcst_crps_"
            f"{d.year}{d.month:02d}{d.day:02d}_00Z.nc"
        )

    else:

        nc_out_path = os.path.join(
            output_folder,
            f"GAN_crps_"
            f"{d.year}{d.month:02d}{d.day:02d}_00Z.nc"
        )

    print(f"Creating output: {nc_out_path}")

    netcdf_dict = create_output_file(nc_out_path)

    netcdf_dict["time_data"][0] = start_times[fcst_idx]

    netcdf_dict["valid_time_data"][0, :] = (
        valid_times_forecast
    )

    # ------------------------------------------------------------------
    # 5. Build GAN forecast input
    #
    # This does not need to be inside the valid-time loop because
    # load_fcst() above has already selected the required LEADTIME.
    # ------------------------------------------------------------------
    network_fcst_input = np.concatenate(
        field_arrays,
        axis=-1
    )

    network_fcst_input = np.expand_dims(
        network_fcst_input,
        axis=0
    )

    print(
        "network_fcst_input shape:",
        np.shape(network_fcst_input)
    )

    # ------------------------------------------------------------------
    # 6. Process forecast valid times
    # ------------------------------------------------------------------
    for valid_time_num in range(
        len(valid_times_forecast)
    ):

        noise_shape = (
            network_fcst_input.shape[1:-1]
            + (noise_channels,)
        )

        noise_gen = NoiseGenerator(
            noise_shape,
            batch_size=1
        )

        progbar = Progbar(ensemble_members)

        # --------------------------------------------------------------
        # Generate GAN ensemble
        # --------------------------------------------------------------
        ens_cgan_preds = []

        for ii in range(ensemble_members):

            gan_inputs = [
                network_fcst_input,
                network_const_input,
                noise_gen()
            ]

            gan_prediction = gen.predict(
                gan_inputs,
                verbose=False
            )

            pred = denormalise(
                gan_prediction[0, :, :, 0]
            )

            if not save_crps_only:

                netcdf_dict["precipitation"][
                    0,
                    ii,
                    valid_time_num,
                    :,
                    :
                ] = pred

            ens_cgan_preds.append(pred)

            progbar.add(1)

        ens_cgan_preds_stacked = np.stack(
            ens_cgan_preds,
            axis=0
        )

        # --------------------------------------------------------------
        # 7. Truth corresponds to forecast VALID date
        # --------------------------------------------------------------
        valid_dt = (
            datetime.combine(
                d,
                datetime.min.time()
            )
            + timedelta(hours=leadtime)
        )

        truth_file = os.path.join(
            truth_input_folder,
            str(valid_dt.year),
            f"{valid_dt.strftime('%Y%m%d')}_06.nc"
        )

        # --------------------------------------------------------------
        # 8. Calculate CRPS if truth exists
        # --------------------------------------------------------------
        if os.path.exists(truth_file):

            truth_data, mask = load_truth_and_mask(
                d.strftime('%Y%m%d'),
                leadtime=leadtime,
                accumulation=accumulation,
                log_precip=log_precip,
                truth_path=truth_input_folder,
                mask_path=mask_input_folder
            )

            truth_data = np.squeeze(
                truth_data,
                axis=0
            )

            print(
                f"shape truth = "
                f"{np.shape(truth_data)}"
            )

            print(
                f"shape ens_cgan_preds_stacked = "
                f"{np.shape(ens_cgan_preds_stacked)}"
            )

            truth_for_crps = truth_data.copy()

            truth_for_crps[mask] = np.nan

            crps = ps.crps_ensemble(
                truth_for_crps,
                ens_cgan_preds_stacked,
                axis=0
            )

            print(
                "NaNs in truth_data:",
                np.isnan(truth_data).sum()
            )

            print(
                "NaNs in truth_for_crps:",
                np.isnan(truth_for_crps).sum()
            )

            print(
                "NaNs in crps:",
                np.isnan(crps).sum()
            )

        else:

            print(
                f"WARNING: No truth for forecast {d}; "
                f"valid date is {valid_dt.date()}: "
                f"{truth_file}"
            )

            print(
                "Setting CRPS to NaN and continuing."
            )

            crps = np.full(
                ens_cgan_preds_stacked.shape[1:],
                np.nan,
                dtype=np.float32
            )

        # --------------------------------------------------------------
        # 9. Write CRPS
        # --------------------------------------------------------------
        netcdf_dict["crps"][
            0,
            valid_time_num,
            :,
            :
        ] = crps

    # ------------------------------------------------------------------
    # 10. Diagnostics
    # ------------------------------------------------------------------
    print(
        "network_fcst_input finite:",
        np.isfinite(network_fcst_input).all()
    )

    print(
        "gan_prediction finite:",
        np.isfinite(gan_prediction).all()
    )

    print(
        "pred finite:",
        np.isfinite(pred).all()
    )

    print(
        "pred min/max:",
        np.nanmin(pred),
        np.nanmax(pred)
    )

    # ------------------------------------------------------------------
    # 11. Close output file ONLY when we're completely finished
    # ------------------------------------------------------------------
    netcdf_dict["rootgrp"].close()

    print(f"Finished forecast date: {d}")