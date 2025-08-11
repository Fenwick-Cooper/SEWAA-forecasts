# Script to compute the histogram data from 1000 member ensembles

import sys
import numpy as np
from datetime import datetime
import netCDF4 as nc

# Where the forecasts are downloaded to
data_dir = "cGAN_forecasts"

# Where the counts are saved to
output_dir = '../interface/view_forecasts/data/counts_6h'

# Smaller chunk size means slower but less RAM usage
# Larger chunk size means faster but more RAM usage
chunk_size = 500

# Get the date from the command line argument
time_str = sys.argv[1]
year = int(time_str[0:4])
month = int(time_str[4:6])
day = int(time_str[6:8])

hour_str = sys.argv[2]
hour = int(hour_str)

# # Choose the date
# forecast_init_date = datetime(year=2024, month=5, day=21, hour=0)
# # Pick today instead:
# #forecast_init_date = datetime.now()
# year = forecast_init_date.year
# month = forecast_init_date.month
# day = forecast_init_date.day
# hour = forecast_init_date.hour

# Define the bins we will use on an approximate log scale (mm/h)
bin_spec_1h = np.array([0,0.04,0.1,0.25,0.4,0.6,0.8,1,1.25,1.5,
                        1.8,2.2,2.6,3,3.5,4,4.7,5.4,6.1,7,
                        8,9.1,10.3,11.7,13.25,15,1000])

# New bins. Decide in workshop.
bin_spec_1h = np.array([ 0.        ,  0.04166667,  0.08333333,  0.20833333,  0.41666667,
                        0.625     ,  0.83333333,  1.        ,  1.25      ,  1.5       ,
                        1.8       ,  2.2       ,  2.6       ,  3.        ,  3.5       ,
                        4.        ,  4.7       ,  5.4       ,  6.1       ,  7.        ,
                        8.        ,  9.       , 10.       , 11.5       , 13.25      ,
                       15.        ,1000])

# To get 24h bin edges.
# bins = np.array([  0.  ,   1.,   2. ,   5.  ,   10. ,  15. ,  20. ,  24.  ,
#                     30.  ,  36.  ,  43.2 ,  52.8 ,  62.4 ,  72.  ,  84.  ,  96.  ,
#                    112.8 , 129.6 , 146.4 , 168.  , 192.  , 218.4 , 247.2 , 280.8 ,
#                    318.  , 360.  , 24000.])

# Open a NetCDF file for reading
file_name = f"{data_dir}/GAN_{year}{month:02d}{day:02d}_{hour:02d}Z.nc"
nc_file = nc.Dataset(file_name, "r")
latitude = np.array(nc_file["latitude"][:])
longitude = np.array(nc_file["longitude"][:])
time = np.array(nc_file["time"][:])
valid_time = np.array(nc_file["fcst_valid_time"][:])[0]

# Compute the counts at each valuid time, latitude and longitude
counts = np.zeros((len(valid_time), len(latitude), len(longitude), len(bin_spec_1h)-1), dtype=int)
for valid_time_num in range(len(valid_time)):
    for j in range(0,len(latitude),chunk_size):
        # Load a chunk of the precip from the netCDF file
        precip = np.array(nc_file["precipitation"][0,:,valid_time_num,j:np.min([j+chunk_size,len(latitude)]),:])
        for k in range(precip.shape[1]):  # Iterate over the chunked dimension
            for i in range(len(longitude)):
                counts[valid_time_num,j+k,i,:],_ = np.histogram(precip[:,k,i], bin_spec_1h)

# Close the netCDF file
nc_file.close()

num_ensemble_members = precip.shape[0]

# Save each valid time in a different file
for valid_time_num in range(len(valid_time)):

    # counts in bin zero are not stored.
    file_name = f"{output_dir}/{year}/counts_{year}{month:02d}{day:02d}_{hour:02d}_{valid_time_num*6+30}h.nc"

    # Create a new NetCDF file
    rootgrp = nc.Dataset(file_name, "w", format="NETCDF4")

    # Describe where this data comes from
    rootgrp.description = "cGAN forecast histogram counts"

    # Create dimensions
    longitude_dim = rootgrp.createDimension("longitude", len(longitude))
    latitude_dim = rootgrp.createDimension("latitude", len(latitude))
    time_dim = rootgrp.createDimension("time", 1)
    valid_time_dim = rootgrp.createDimension("valid_time", 1)
    bins_dim = rootgrp.createDimension("bins", len(bin_spec_1h)-2)

    # Create the longitude variable
    longitude_data = rootgrp.createVariable("longitude", "f4", ("longitude"), zlib=False)
    longitude_data.units = "degrees_east"
    longitude_data[:] = longitude   # Write the longitude data

    # Create the latitude variable
    latitude_data = rootgrp.createVariable("latitude", "f4", ("latitude"), zlib=False)
    latitude_data.units = "degrees_north"
    latitude_data[:] = latitude     # Write the latitude data

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
    counts_data = rootgrp.createVariable("counts", "i2", ("bins","latitude","longitude"), zlib=True, complevel=9)
    counts_data.description = "Histogram bin counts"
    counts_data.num_members = num_ensemble_members
    # Compression is better if we move the axis order
    counts_data[:] = np.moveaxis(counts,[0,1,2,3],[0,2,3,1])[valid_time_num,1:,:,:]

    # Close the netCDF file
    rootgrp.close()
    
