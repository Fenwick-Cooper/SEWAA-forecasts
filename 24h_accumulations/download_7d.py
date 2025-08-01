# Python script to download todays forecast data for cGAN

# Parameter code list:
#
# sfc fields:
# 59.128 - CAPE (Replace with MUCAPE)
# 228235 - MUCAPE, Most-unstable CAPE
# 78.128 - total column cloud liquid water
# 79.128 - total column cloud ice water
# 89.228 - total column rain water
# 134.128 - surface pressure
# 136.128 - total column water
# 137.128 - total column v.i. water vapour
# 143.128 - convective precipitation
# 167.128 - 2m temperature
# 176.128 - surface net solar radiation
# 187.128 - medium cloud cover
# 228.128 - total precipitation
#
# Not needed for cGAN
# 142.128 - large-scale precipitation
# 165.128 - 10m u component
# 166.128 - 10m v component
# 212.128 - TOA incident solar radiation; NOT AVAILABLE IN enfo, ONLY IN oper
#
# pl fields:
# 131 - U component of wind
# 132 - V component of wind

import os
import sys
from datetime import date, timedelta
from ecmwfapi import ECMWFService

# Get command line arguments
time_str = f"{int(sys.argv[1]):02d}"

# Where to store the forecast data from ECMWF
IFS_data_dir = "/home/c/cooperf/IFS/ICPAC_data/operational/7d/grib"

# The server that will give us the data
server = ECMWFService("mars")

# Get today's date
if (time_str == "18"):
    todays_date = date.today() - timedelta(days=1)
else:
    todays_date = date.today()
# Use an old date for now
# todays_date = date(2024,3,8)
date_str = todays_date.strftime("%Y%m%d")

print(f"time_str = {time_str}, todays_date = {todays_date}")
print(f"date_str = {date_str}")

# Where to store the pressure level data
file_name = f"{IFS_data_dir}/{date_str}_{time_str}Z_pl700_174h.grib"

# If the file exists don't download again
if os.path.isfile(file_name):
    print(f"{file_name} already exists.")
else:

    # Create request specification for data on pressure levels
    pl_req = {"class": "od",
              "stream": "enfo",
              "expver": "1",
              "type": "pf",
              "time": time_str,
              "step": "0/to/174/by/6",
              "area": "25/19/-14/54.5",  # ICPAC region + a bit for interpolation (format N/W/S/E)
              "number": "/".join([str(i) for i in range(1, 50+1)]),  # concise way of writing "1/2/3/.../49/50"
              "grid": "O640", # "O640" is what Andrew used. "av" specifies the archived grid (no interpolation)
              "levtype": "pl",
              "levelist": "700",
              "param": "131/132",
              "date": date_str,
             }
    
    print(f"pl_req = {pl_req}")
    
    # Request the data from ECMWF
    server.execute(pl_req, file_name)
    
# Where to store the pressure level data
file_name = f"{IFS_data_dir}/{date_str}_{time_str}Z_sfc_174h.grib"

# If the file exists don't download again
if os.path.isfile(file_name):
    print(f"{file_name} already exists.")
else:
    
    # Create request specification for data on a single level
    sfc_req = {"class": "od",
               "stream": "enfo",
               "expver": "1",
               "type": "pf",
               "time": time_str,
               "step": "0/to/174/by/6",
               "area": "25/19/-14/54.5",  # ICPAC region + a bit for interpolation (format N/W/S/E)
               "number": "/".join([str(i) for i in range(1, 50+1)]),  # concise way of writing "1/2/3/.../49/50"
               "grid": "O640", # "O640" is what Andrew used. "av" specifies the archived grid (no interpolation)
               "levtype": "sfc",
               "param": "78.128/79.128/89.228/134.128/136.128/137.128/143.128/165.128/166.128/167.128/176.128/187.128/228.128",
               "date": date_str,
              }

    print(f"sfc_req = {sfc_req}")

    # Request the data from ECMWF
    server.execute(sfc_req, file_name)

# Where to store the pressure level data
file_name = f"{IFS_data_dir}/{date_str}_{time_str}Z_MUCAPE_174h.grib"

# If the file exists don't download again
if os.path.isfile(file_name):
    print(f"{file_name} already exists.")
else:
    
    # Create request specification for data on a single level
    mucape_req = {"class": "od",
				  "stream": "enfo",
				  "expver": "1",
				  "type": "pf",
				  "time": time_str,
				  "step": "0/to/174/by/6",
				  "area": "25/19/-14/54.5",  # ICPAC region + a bit for interpolation (format N/W/S/E)
				  "number": "/".join([str(i) for i in range(1, 50+1)]),  # concise way of writing "1/2/3/.../49/50"
				  "grid": "O640", # "O640" is what Andrew used. "av" specifies the archived grid (no interpolation)
				  "levtype": "sfc",
				  "param": "228235",
				  "date": date_str,
				 }

    print(f"mucape_req = {mucape_req}")

    # Request the data from ECMWF
    server.execute(mucape_req, file_name)
