# Script to compute the histogram data from 1000 member ensembles

import sys
import numpy as np
from datetime import datetime
import netCDF4 as nc
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    "--date",
    help="Forecast date in YYYYMMDD format"
)

parser.add_argument(
    "--time",
    type=int,
    choices=[0, 6, 12, 18],
    help="Forecast initialisation hour"
)

parser.add_argument(
    "--accumulation",
    choices=[6, 24],
    type=int,
    default=6,
    help="Accumulation of forecasts in hours (default: 6)"
)

args = parser.parse_args()

accumulation = args.accumulation

time_str = args.date
year = int(time_str[0:4])
month = int(time_str[4:6])
day = int(time_str[6:8])
hour = args.time

# Where the counts are saved to
output_dir = f'../../data/ifs+cgan_radar_{accumulation}h/histograms/{year}'
os.makedirs(output_dir, exist_ok=True)
# Where the forecasts are downloaded to
if accumulation == 6:
    data_dir = "../../data/ifs+cgan_radar_6h/cGAN_forecasts"
elif accumulation == 24:
    data_dir = "../../data/ifs+cgan_radar_24h/cGAN_forecasts"
else:
    raise ValueError("Accumulation must be 6 or 24.")

# Smaller chunk size means slower but less RAM usage
# Larger chunk size means faster but more RAM usage
chunk_size = 50


# Define the bins we will use on an approximate log scale (mm/h)
bin_spec_1h = np.array([ 0.        ,  0.04166667,  0.08333333,  0.20833333,  0.41666667,
                        0.625     ,  0.83333333,  1.        ,  1.25      ,  1.5       ,
                        1.8       ,  2.2       ,  2.6       ,  3.        ,  3.5       ,
                        4.        ,  4.7       ,  5.4       ,  6.1       ,  7.        ,
                        8.        ,  9.       , 10.       , 11.5       , 13.25      ,
                       15.        ,1000])

# Load some details
file_name = f"{data_dir}/GAN_{year}{month:02d}{day:02d}_{hour:02d}Z.nc"
nc_file = nc.Dataset(file_name, "r")
x = np.array(nc_file["x"][:])
y = np.array(nc_file["y"][:])

latitude = np.array(nc_file["latitude"][:])
longitude = np.array(nc_file["longitude"][:])

time = np.array(nc_file["time"][:])
# All forecast valid times are stored in the same file
valid_time = np.array(nc_file["fcst_valid_time"][0, :])
num_valid_times = len(valid_time)
# Number of ensemble members
num_ensemble_members = nc_file["precipitation"].shape[1]
        
counts = np.zeros((num_valid_times, len(y), len(x), len(bin_spec_1h)-1), dtype=int)

for valid_time_num in range(num_valid_times):

    print(
        f"Computing histograms for valid time "
        f"{valid_time_num + 1}/{num_valid_times}"
    )
    
    # Compute the counts at each valid time, latitude and longitude
    for j in range(0,len(y),chunk_size):
        # Load a chunk of the precip from the netCDF file
        precip = np.array(nc_file["precipitation"][0,:,valid_time_num,j:np.min([j+chunk_size,len(y)]),:])
        for k in range(precip.shape[1]):  # Iterate over the chunked dimension
            for i in range(len(x)):
                counts[valid_time_num,j+k,i,:],_ = np.histogram(precip[:,k,i], bin_spec_1h)

# Close the netCDF file
nc_file.close()


lead_hours = np.rint(valid_time - time[0]).astype(int)
print("Forecast lead times:", lead_hours)

# Save each valid time in a different file
for valid_time_num in range(len(valid_time)):
    
    valid_hour = lead_hours[valid_time_num]

    file_name = (
        f"{output_dir}/"
        f"counts_{year}{month:02d}{day:02d}_"
        f"{hour:02d}_{valid_hour}h.nc"
    )

    # Create a new NetCDF file
    rootgrp = nc.Dataset(file_name, "w", format="NETCDF4")

    # Describe where this data comes from
    rootgrp.description = "cGAN forecast histogram counts"

    # Create dimensions
    rootgrp.createDimension("y", len(y))
    rootgrp.createDimension("x", len(x))
    rootgrp.createDimension("time", 1)
    rootgrp.createDimension("valid_time", 1)
    rootgrp.createDimension(
        "bins",
        len(bin_spec_1h) - 2,
    )

    # Projected radar-grid coordinates
    y_data = rootgrp.createVariable(
        "y",
        "f4",
        ("y",),
        zlib=False,
    )
    y_data.units = "meters"
    y_data[:] = y

    x_data = rootgrp.createVariable(
        "x",
        "f4",
        ("x",),
        zlib=False,
    )
    x_data.units = "meters"
    x_data[:] = x

    # Geographic coordinates corresponding to the radar grid.
    # These are 2-D because this is a projected y/x grid.
    latitude_data = rootgrp.createVariable(
        "latitude",
        "f4",
        ("y", "x"),
        zlib=False,
    )
    latitude_data.units = "degrees_north"
    latitude_data[:] = latitude

    longitude_data = rootgrp.createVariable(
        "longitude",
        "f4",
        ("y", "x"),
        zlib=False,
    )
    longitude_data.units = "degrees_east"
    longitude_data[:] = longitude

    # Create the time variable
    time_data = rootgrp.createVariable("time", "f4", ("time"), zlib=False)
    time_data.units = "hours since 1900-01-01 00:00:00.0"
    time_data.description = "Time corresponding to forecast model start"
    time_data[:] = time         # Write the forecast model start time

    # Create the valid_time variable
    valid_time_data = rootgrp.createVariable("valid_time", "f4", ("valid_time"), zlib=False)
    valid_time_data.units = "hours since 1900-01-01 00:00:00.0"
    valid_time_data.description = "Time corresponding to forecast prediction"
    valid_time_data[:] = valid_time[valid_time_num]  # Write the forecast model valid times

    # Bin specification. First bin is zero, final bin is infinity.
    bins_data = rootgrp.createVariable("bins", "f4", ("bins"), zlib=False)
    bins_data.units = "mm/h"
    bins_data.description = "Histogram bin edges"
    bins_data[:] = bin_spec_1h[1:-1]  # Write histogram bin specification

    # Create the counts variable
    counts_data = rootgrp.createVariable("counts", "i2", ("bins","y","x"), zlib=True, complevel=9)
    counts_data.description = "Histogram bin counts"
    counts_data.num_members = num_ensemble_members
    # Compression is better if we move the axis order
    counts_data[:] = np.moveaxis(counts,[0,1,2,3],[0,2,3,1])[valid_time_num,1:,:,:]

    # Close the netCDF file
    rootgrp.close()
    