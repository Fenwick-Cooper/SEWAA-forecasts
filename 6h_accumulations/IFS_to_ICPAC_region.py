#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load and interpolate a grib file on a reduced Gaussian grid

import sys
import numpy as np
import netCDF4 as nc
import h5py
from datetime import datetime, date, timedelta
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

# For debugging only
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


# In[2]:


# Where the forecast data GRIB files are
# GRIB_dir = f"/network/group/aopp/predict/TIP021_MCRAECOOPER_IFS/IFSens-EUAF_idx/2018/"  # XXX To prototype
GRIB_dir = "/home/c/cooperf/IFS/ICPAC_data/operational/grib_historical"

# Where to save the NetCDF files
output_data_dir = "/home/c/cooperf/IFS/ICPAC_data/operational/processed_IFS"

# A single IMERG data file to get latitude and longitude
IMERG_data_dir = "/home/c/cooperf/IFS/IMERG_V07"
IMERG_file_name = f"{IMERG_data_dir}/2018/Jan/3B-HHR.MS.MRG.3IMERG.20180101-S000000-E002959.0000.V07B.HDF5"

# HDF5 in the ICPAC region
h5_file = h5py.File(IMERG_file_name)
latitude = h5_file['Grid']["lat"][763:1147]
longitude = h5_file['Grid']["lon"][1991:2343]
h5_file.close()

# Time of forecast initialisation for the command line argument
time_str = f"{int(sys.argv[1]):02d}"


# In[3]:


# Convert numpy.datetime64 to datetime
def datetime64_to_datetime(datetime64):
    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')
    seconds_since_epoch = (datetime64 - unix_epoch) / one_second
    return datetime.utcfromtimestamp(seconds_since_epoch)


# In[4]:


# Get today's date
if (time_str == "18"):
    todays_date = date.today() - timedelta(days=1)
else:
    todays_date = date.today()

# Use an old date for now
#todays_date = datetime(2018,6,6)
#todays_date = datetime(2024,3,9)
#todays_date = datetime(2024,3,10)

date_str = todays_date.strftime("%Y%m%d")


# In[5]:


# Load the data on pressure levels
#file_name = f"{GRIB_dir}/20180606_pl700_7d.grib"  # XXX To prototype
file_name = f"{GRIB_dir}/{todays_date.year}{todays_date.month:02}{todays_date.day:02}_{time_str}Z_pl700_54h.grib"
ds_pl = xr.load_dataset(file_name, engine="cfgrib")

# Load the data on single levels
#file_name = GRIB_dir + f"20180606_sfc_7d.grib"  # XXX To prototype
file_name = f"{GRIB_dir}/{todays_date.year}{todays_date.month:02}{todays_date.day:02}_{time_str}Z_sfc_54h.grib"
ds_sfc = xr.load_dataset(file_name, engine="cfgrib")

# Load the mucape
file_name = f"{GRIB_dir}/{todays_date.year}{todays_date.month:02}{todays_date.day:02}_{time_str}Z_MUCAPE_54h.grib"
ds_MUCAPE = xr.load_dataset(file_name, engine="cfgrib")

# The number of ensemble members in each forecast
num_ensemble_members = ds_sfc["number"].shape[0]

# Construct an array of all of the points where data exists
# points = np.array([ds["latitude"].values, ds["longitude"].values-360]).T
# points_sfc = np.array([ds_sfc["longitude"].values-360, ds_sfc["latitude"].values]).T
# points_pl = np.array([ds_pl["longitude"].values-360, ds_pl["latitude"].values]).T
# XXX Remove the -360 for operational data
points_sfc = np.array([ds_sfc["longitude"].values, ds_sfc["latitude"].values]).T
points_pl = np.array([ds_pl["longitude"].values, ds_pl["latitude"].values]).T
points_MUCAPE = np.array([ds_MUCAPE["longitude"].values, ds_MUCAPE["latitude"].values]).T

# Perform a one off triangulation of all the points
tri_sfc = Delaunay(points_sfc)
tri_pl = Delaunay(points_pl)
tri_MUCAPE = Delaunay(points_MUCAPE)

# Generate the points we are interpolating to
# lat_dest, lon_dest = np.meshgrid(latitude, longitude)
lon_dest, lat_dest = np.meshgrid(longitude, latitude)


# In[6]:


# Check that the data is in the correct place
if False:

    # Create the interpolation function
    interp = LinearNDInterpolator(tri_sfc, np.mean(ds_sfc["sp"].values[:,0,:], axis=0))

    # Interpolate
    sp_interp = interp(lon_dest, lat_dest)

    # Plot
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    c = plt.pcolormesh(longitude, latitude, sp_interp, transform=ccrs.PlateCarree())
    cb = plt.colorbar(c, fraction=0.02)
    cb.ax.tick_params(labelsize=18)
    cb.set_label('Pa',size=18)
    plt.title(f"Surface pressure")
    plt.show()


# In[7]:


# Time that the forecast starts
start_time = datetime64_to_datetime(ds_sfc["time"].values)

# Convert numpy datetime64 to timedelta since 1900
dt = start_time - datetime(1900,1,1)

# Convert timedelta to hours (ignoring microseconds)
dt_hours = (dt.days * 24*60*60 + dt.seconds) / (60*60)

# Create the list of valid times
num_valid_times = 5  # 30, 36, 42, 48, 54 hours
valid_hours = np.zeros(num_valid_times)
for i in range(num_valid_times):
    
    # Convert numpy datetime64 to timedelta since 1900
    valid_time = datetime64_to_datetime(ds_sfc['time'].values + ds_sfc['step'][i+5].values)
    valid_dt = valid_time - datetime(1900,1,1)
    
    # Convert timedelta to hours (ignoring microseconds)
    valid_hours[i] = (valid_dt.days * 24*60*60 + valid_dt.seconds) / (60*60)


# In[8]:


# the_keys = ['cape','tclw','tciw','tcrw','sp','tcw','tcwv','lsp','cp','u10','v10','t2m','ssr','mcc','tp','u','v']
# A subset of the keys are required for cGAN
the_keys = ['cape','tclw','tciw','tcrw','sp','tcw','tcwv','cp','t2m','ssr','mcc','tp','u','v']
the_vars = {
    "cape":{
        "long_name":"Convective available potential energy",
        "units":"J kg**-1",
        "accumulated":False,
        "single_level":True,
    },
    "tclw":{
        "long_name":"Total column cloud liquid water",
        "units":"kg m**-2",
        "accumulated":False,
        "single_level":True,
    },
    "tciw":{
        "long_name":"Total column cloud ice water",
        "units":"kg m**-2",
        "accumulated":False,
        "single_level":True,
    },
    "tcrw":{
        "long_name":"Total column rain water",
        "units":"kg m**-2",
        "accumulated":False,
        "single_level":True,
    },
    "sp":{
        "long_name":"Surface pressure",
        "units":"Pa",
        "accumulated":False,
        "single_level":True,
    },
    "tcw":{
        "long_name":"Total column water",
        "units":"kg m**-2",
        "accumulated":False,
        "single_level":True,
    },
    "tcwv":{
        "long_name":"Total column vertically-integrated water vapour",
        "units":"kg m**-2",
        "accumulated":False,
        "single_level":True,
    },
    "lsp":{
        "long_name":"Large-scale precipitation",
        "units":"m hour**-1",
        "accumulated":True,
        "accumulation_per_h":1,
        "single_level":True,
    },
    "cp":{
        "long_name":"Convective precipitation",
        "units":"m hour**-1",
        "accumulated":True,
        "accumulation_per_h":1,
        "single_level":True,
    },
    "u10":{
        "long_name":"10 metre U wind component",
        "units":"m s**-1",
        "accumulated":False,
        "single_level":True,
    },
    "v10":{
        "long_name":"10 metre V wind component",
        "units":"m s**-1",
        "accumulated":False,
        "single_level":True,
    },
    "t2m":{
        "long_name":"2 metre temperature",
        "units":"K",
        "accumulated":False,
        "single_level":True,
    },
    "ssr":{
        "long_name":"Surface net solar radiation",
        "units":"J m**-2 s**-1",
        "accumulated":True,
        "accumulation_per_h":3600,
        "single_level":True,
    },
    "mcc":{
        "long_name":"Medium cloud cover",
        "units":"(0 - 1)",
        "accumulated":False,
        "single_level":True,
    },
    "tp":{
        "long_name":"Total precipitation",
        "units":"m hour**-1",
        "accumulated":True,
        "accumulation_per_h":1,
        "single_level":True,
    },
    "u":{
        "long_name":"700 hPa U wind component",
        "units":"m s**-1",
        "accumulated":False,
        "single_level":False,
    },
    "v":{
        "long_name":"700 hPa V wind component",
        "units":"m s**-1",
        "accumulated":False,
        "single_level":False,
    },
}


# In[9]:


# Create the NetCDF files
file_name = f"IFS_{date_str}_{time_str}Z.nc"

print(f"Saving data to {output_data_dir}/{file_name}")

# Create a new NetCDF file
rootgrp = nc.Dataset(f"{output_data_dir}/{file_name}", "w", format="NETCDF4")

# Describe where this data comes from
rootgrp.description = "IFS forecast mean and variance with a lead time of 30h - 54h in the ICPAC region."

# Create dimensions
longitude_dim = rootgrp.createDimension("longitude", len(longitude))
latitude_dim = rootgrp.createDimension("latitude", len(latitude))
time_dim = rootgrp.createDimension("time", None)
valid_time_dim = rootgrp.createDimension("valid_time", num_valid_times)

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
time_data[:] = dt_hours         # Write the forecast model start time

# Create the valid_time variable
valid_time_data = rootgrp.createVariable("valid_time", "f4", ("valid_time"), zlib=False)
valid_time_data.units = "hours since 1900-01-01 00:00:00.0"
valid_time_data.description = "Time corresponding to forecast prediction"
valid_time_data[:] = valid_hours  # Write the forecast model valid times

for key_no in range(len(the_keys)):
    print(f"Interpolating {the_keys[key_no]}")
    
    # Which .grib file should we be reading
    if (the_keys[key_no] == "cape"):
        # XXX The cape variable changed to mucape, this is a temporary fix
        ds = ds_MUCAPE
        tri = tri_MUCAPE
    else:
        if the_vars[the_keys[key_no]]['single_level']:
            ds = ds_sfc
            tri = tri_sfc
        else:
            ds = ds_pl
            tri = tri_pl
    
    data_mean_interp = np.zeros((num_valid_times,len(latitude),len(longitude)))
    data_std_interp = np.zeros((num_valid_times,len(latitude),len(longitude)))
    for valid_time_idx in range(num_valid_times):
        
        # XXX The cape variable changed to mucape, this is a temporary fix
        if (the_keys[key_no] == "cape"):
            
            values = ds["mucape"].values[:,valid_time_idx+5,:]
            values_mean = np.mean(values, axis=0)
            # XXX Should be np.var but is np.std for consistency with training data
            values_std = np.std(values, axis=0, ddof=1)
            
        else:
            if (the_vars[the_keys[key_no]]["accumulated"]):
                
                # If this is the final accumulated value
                if (valid_time_idx == num_valid_times-1):
                    values_mean = np.zeros(ds[the_keys[key_no]].values[0,0,:].shape)
                    values_std  = np.zeros(ds[the_keys[key_no]].values[0,0,:].shape)
                else:
                    values = ds[the_keys[key_no]].values[:,valid_time_idx+5:valid_time_idx+7,:]
                    values_mean = np.mean(values[:,1,:] - values[:,0,:], axis=0)
                    # XXX Should be np.var but is np.std for consistency with training data
                    values_std = np.std(values[:,1,:] - values[:,0,:], axis=0, ddof=1)
            else:
                values = ds[the_keys[key_no]].values[:,valid_time_idx+5,:]
                values_mean = np.mean(values, axis=0)
                # XXX Should be np.var but is np.std for consistency with training data
                values_std = np.std(values, axis=0, ddof=1)
        
        # Create the interpolation functions
        interp_mean = LinearNDInterpolator(tri, values_mean)
        interp_std = LinearNDInterpolator(tri, values_std)
        
        # Interpolate
        data_mean_interp[valid_time_idx,:,:] = interp_mean(lon_dest, lat_dest)
        data_std_interp[valid_time_idx,:,:] = interp_std(lon_dest, lat_dest)
    
    # Label 700 hPa fields as being at 700 hPa
    if the_keys[key_no] in ['u','v']:
        key_name = the_keys[key_no] + '700'  # On the 700 hPa level
    else:
        key_name = the_keys[key_no]
    
    # Create the mean variable
    var_name = f"{key_name}_ensemble_mean"
    ensemble_mean_data = rootgrp.createVariable(var_name, "f4", ("valid_time","latitude","longitude"), zlib=True)
    ensemble_mean_data.units = the_vars[the_keys[key_no]]["units"]
    ensemble_mean_data.long_name = f'{the_vars[the_keys[key_no]]["long_name"]} ensemble mean'
    
    # XXX The cape variable changed to mucape, this is a temporary fix
    if (the_keys[key_no] == "cape"):
        ensemble_mean_data.warning = "WARNING: This is actually most-unstable CAPE (mucape)"
    
    ensemble_mean_data[:] = data_mean_interp

    # Create the variance variable
    var_name = f"{key_name}_ensemble_standard_deviation"
    ensemble_std_data = rootgrp.createVariable(var_name, "f4", ("valid_time","latitude","longitude"), zlib=True)
    ensemble_std_data.units = f'({the_vars[the_keys[key_no]]["units"]})**2'
    ensemble_std_data.long_name = f'{the_vars[the_keys[key_no]]["long_name"]} ensemble standard deviation'
    
    # XXX The cape variable changed to mucape, this is a temporary fix
    if (the_keys[key_no] == "cape"):
        ensemble_mean_data.warning = """WARNING: The ensemble standard deviation was interpolated not the ensemble variance.
WARNING: This is actually most-unstable CAPE (mucape)"""
    else:
        ensemble_std_data.warning = "WARNING: The ensemble standard deviation was interpolated not the ensemble variance."
    
    ensemble_std_data[:] = data_std_interp
    
# Close the netCDF file
rootgrp.close()


# In[10]:


# Test that values are the same as in Andrew's files
if False:

    # Load the new axes and times
    file_name = f"IFS_{date_str}_00Z.nc"
    nc_file = nc.Dataset(f"{output_data_dir}/{file_name}")
    time = np.array(nc_file["time"][:])
    valid_time = np.array(nc_file["valid_time"][:])
    latitude = np.array(nc_file["latitude"][:])
    longitude = np.array(nc_file["longitude"][:])
    nc_file.close()

    # For each variable
    for key_no in range(len(the_keys)):
    #for key_no in [4]:  # 'sp'

        # Find the key name
        key = the_keys[key_no]
        if key == 'u':
            key = 'u700'
        if key == 'v':
            key = 'v700'
    
        # Load the new data
        file_name = f"IFS_{date_str}_00Z.nc"
        nc_file = nc.Dataset(f"{output_data_dir}/{file_name}")
        data_mean = np.array(nc_file[f"{key}_ensemble_mean"][:])
        data_std = np.array(nc_file[f"{key}_ensemble_standard_deviation"][:])
        nc_file.close()

        # Load the original data
        file_name = f"/home/c/cooperf/IFS/IFS-regICPAC-meansd/2018/{key}.nc"
        nc_file = nc.Dataset(file_name)
        time_A = np.array(nc_file["time"][156])
        valid_time_A = np.array(nc_file["fcst_valid_time"][156,5:10])
        latitude_A = np.array(nc_file["latitude"][:])
        longitude_A = np.array(nc_file["longitude"][:])
        data_mean_A = np.array(nc_file[f"{key}_mean"][156,5:10,:,:])  # time, valid_time, latitude, longitude
        data_std_A = np.array(nc_file[f"{key}_sd"][156,5:10,:,:])
        nc_file.close()

        print(key)
        print(f"time       : {time_A - time}")
        print(f"valid_time : {np.max(np.abs(valid_time_A - valid_time))}")
        print(f"latitude   : {np.max(np.abs(latitude_A - latitude))}")
        print(f"longitude  : {np.max(np.abs(longitude_A - longitude))}")
        print(f"{key}_mean : {np.max(np.abs(data_mean_A - data_mean)) / np.max(np.abs(data_mean_A))}")
        print(f"{key}_std  : {np.max(np.abs(data_std_A - data_std)) / np.max(np.abs(data_std_A))}")
        print()


# In[ ]:




