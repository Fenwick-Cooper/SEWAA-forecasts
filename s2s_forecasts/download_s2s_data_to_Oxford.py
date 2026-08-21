"""
File to download ECMWF S2S forecast data for a given date,
process it into weekly mean and std NetCDF files,
and save to Oxford server.
Use on command line using python download_s2s_data_to_Oxford.py --date YYYYMMDD.
"""

import numpy as np
from ecmwfapi import ECMWFService
import os
import xarray as xr
import glob
import argparse
from pathlib import Path

OUT_FOLDER = "/nf2/web/rain/ICPAC/operational/s2s_forecasts/s2s_forecast_data"

def parse_date_yyyymmdd(date_str):
    """
    Parse a date string in YYYYMMDD format.

    Parameters
    ----------
    date_str : str
        Date string of the form YYYYMMDD.

    Returns
    -------
    tuple[int, int, int]
        Year, month, and day as integers.

    Raises
    ------
    argparse.ArgumentTypeError
        If the input is not exactly 8 digits long or contains non-digits.
    """
    if len(date_str) != 8 or not date_str.isdigit():
        raise argparse.ArgumentTypeError("date must be in YYYYMMDD format")
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    return year, month, day

def sort_lat_lon(ds):
    """
    Standardize latitude/longitude coordinate names and ordering.

    Renames coordinates from 'latitude'/'longitude' to 'lat'/'lon' when needed,
    converts longitudes to the [-180, 180] range if they are in [0, 360],
    and sorts latitude in ascending order.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with latitude/longitude coordinates.

    Returns
    -------
    xarray.Dataset
        Dataset with standardized and sorted spatial coordinates.
    """
    #Convert latitude/longitude to lat/lon
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})

    #If longitudes are in 0-360, convert to -180 to 180
    if ds.lon.max().item() > 180: 
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    #If latitudes are in descending order, sort to ascending order
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.sortby("lat")

    return ds

def calculate_steps(lead_times_weeks):
    """
    Build ECMWF step ranges for the requested lead weeks.

    Each lead week is converted into a 168-hour interval, returned in the
    MARS step syntax used by ECMWF requests.

    Parameters
    ----------
    lead_times_weeks : iterable of int
        Lead weeks to request, e.g. [1, 2, 3].

    Returns
    -------
    str
        ECMWF step string such as '0-168/168-336/336-504'.
    """
    steps = str()
    for lead in lead_times_weeks:
        steps += f"{168*(lead-1)}-{168*lead}/"
    steps = steps[:-1]  # Remove trailing slash
    
    return steps

def delete_grib_and_index(fname):
    """
    Delete a GRIB file and any matching ECMWF index files.

    Parameters
    ----------
    fname : str
        Path to the GRIB file. Any files matching fname + '*.idx' are also removed.

    Returns
    -------
    None
    """
    try: #Delete grib file
        os.remove(fname)
        print("Deleted grib file: ", fname)
    except Exception as e:
        print(f"Error deleting file {fname}: {e}")
    for path in glob.glob(fname + "*.idx"): #Try and remove any index files with matching name
        try:
            os.remove(path)
            print("Deleted index file: ", path)
        except Exception as e:
            print(f"Error deleting file {path}: {e}")

def download_s2s_data_ecmwf(year, month, day, lead_times_weeks=[1,2,3], OUT_FOLDER="./s2s_data"):
    """
    Download ECMWF subseasonal-to-seasonal precipitation data for a given date.

    The request downloads total precipitation forecast mean data from the ECMWF
    MARS archive and saves it as a GRIB file in OUT_FOLDER.

    Parameters
    ----------
    year : int
        Forecast initialization year.
    month : int
        Forecast initialization month.
    day : int
        Forecast initialization day.
    lead_times_weeks : list[int], optional
        Lead weeks to download, by default [1, 2, 3].
    OUT_FOLDER : str, optional
        Directory where the GRIB file will be saved.

    Returns
    -------
    None
    """
    server = ECMWFService("mars")
    fname = os.path.join(OUT_FOLDER, f"{year}-{month:02d}-{day:02d}_tprate.grib")
    #Check path exists and make if not
    os.makedirs(OUT_FOLDER, exist_ok=True)

    #Calculate step string
    steps = calculate_steps(lead_times_weeks)

    #Build MARS request in correct format for ECMWF API
    request = {
        "class": "od",
        "date": f"{year}-{month:02d}-{day:02d}",
        "expver": 1,
        "levtype": "sfc",
        "number": "0/1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50/51/52/53/54/55/56/57/58/59/60/61/62/63/64/65/66/67/68/69/70/71/72/73/74/75/76/77/78/79/80/81/82/83/84/85/86/87/88/89/90/91/92/93/94/95/96/97/98/99/100",
        "param": 228.172,
        "step": steps,
        "stream": "eefo",
        "time": "00:00:00",
        "type": "fcmean",
        "grid": "0.4/0.4",
    }
    #Send request
    server.execute(request, fname)

    print("Dowloaded ECMWF S2S data for date: ", f"{year}-{month:02d}-{day:02d}")


def process_s2s_data(year, month, day, lead_times_weeks=[1,2,3], delete_grib=True, IN_FOLDER="./s2s_data", OUT_FOLDER="./s2s_data"):
    """
    Process downloaded ECMWF S2S GRIB data into weekly NetCDF files.

    The function opens the GRIB file, standardizes coordinates, converts total
    precipitation into mm/week, computes ensemble mean and standard deviation
    for each requested lead week, and writes one NetCDF file per week.

    Parameters
    ----------
    year : int
        Forecast initialization year.
    month : int
        Forecast initialization month.
    day : int
        Forecast initialization day.
    lead_times_weeks : list[int], optional
        Lead weeks to process, by default [1, 2, 3].
    delete_grib : bool, optional
        Whether to delete the GRIB and index files after processing.
    IN_FOLDER : str, optional
        Directory containing the downloaded GRIB file.
    OUT_FOLDER : str, optional
        Directory where processed NetCDF files will be written.

    Returns
    -------
    None
    """
    #Open GRIB file with xarray
    fname = os.path.join(IN_FOLDER, f"{year}-{month:02d}-{day:02d}_tprate.grib")
    try:
        ds = xr.open_dataset(fname)
    except Exception as e:
        print(f"Error opening file {fname}: {e}. Please check download was successful and file is not corrupted.")
        return

    #Tidy latitude/longitude and convert units
    ds = sort_lat_lon(ds) 
    ds = ds*3600*168 #Convert to m/week from m/s
    ds = ds*1000 #Convert to mm/week from m/week

    for i, week in enumerate(lead_times_weeks):
        ds_week = ds.isel(step=i) #Select the correct step for this lead week. This assumes the steps are in the same order as the lead_times_weeks list, which should be true if the request is built correctly.
        ds_week_mean = ds_week.mean('number').rename({'tprate': 'tp_mean'}) #Renaming variable to avoid confusion with std
        ds_week_mean['tp_std'] = ds_week.std('number').tprate #Calculate std across ensemble members and add as new variable in same dataset
        ds_week_mean.attrs['units'] = 'mm/week'
        ds_week_mean["valid_time"] = ds_week_mean["valid_time"] - pd.Timedelta(days=7) #Change valid_time convention to be start of accumulation period, not end
        ds_week_mean["step"] = ds_week_mean["step"] - pd.Timedelta(days=7) #Change step convention to be start of accumulation period, not end
        ds_week_mean = ds_week_mean.squeeze()

        #Save file
        ds_week_mean.to_netcdf(os.path.join(OUT_FOLDER, f"{year}-{month:02d}-{day:02d}_tp_meanstd_{week}wklead.nc")) #naming convention is week 1 for hours 0-168 etc.
        print(f"Processed and saved S2S data for date: {year}-{month:02d}-{day:02d}, lead time: {week} weeks")
        print(f"Saved to: {os.path.join(OUT_FOLDER, f'{year}-{month:02d}-{day:02d}_tp_meanstd_{week}wklead.nc')}")

    # Delete grib file and index files to clean up
    if delete_grib:
        delete_grib_and_index(fname)

def download_and_process_s2s_data(year, month, day, lead_times_weeks=[1,2,3], delete_grib=True, RAW_FOLDER="./s2s_data", PROC_FOLDER="./s2s_data"):
    """
    Download ECMWF S2S data and process it into NetCDF outputs.

    This is a convenience wrapper that first downloads the GRIB file and then
    processes it into weekly mean/std NetCDF files.

    Parameters
    ----------
    year : int
        Forecast initialization year.
    month : int
        Forecast initialization month.
    day : int
        Forecast initialization day.
    lead_times_weeks : list[int], optional
        Lead weeks to download and process, by default [1, 2, 3].
    delete_grib : bool, optional
        Whether to delete the raw GRIB file after processing.
    RAW_FOLDER : str, optional
        Directory for the downloaded GRIB file.
    PROC_FOLDER : str, optional
        Directory for processed NetCDF files.

    Returns
    -------
    None
    """
    
    print("Starting download and processing of S2S data for date: ", f"{year}-{month:02d}-{day:02d}")
    
    download_s2s_data_ecmwf(year, month, day, lead_times_weeks=lead_times_weeks, OUT_FOLDER=RAW_FOLDER)
    process_s2s_data(year, month, day, lead_times_weeks=lead_times_weeks, delete_grib=delete_grib, IN_FOLDER=RAW_FOLDER, OUT_FOLDER=PROC_FOLDER)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        required=True,
        help="Date in YYYYMMDD format, e.g. 20260527",
    )
    parser.add_argument(
        "--no-delete-grib",
        action="store_true",
        help="Keep the downloaded GRIB file",
    )
    args = parser.parse_args()

    year, month, day = parse_date_yyyymmdd(args.date)

    FULL_OUT_FOLDER = os.path.join(OUT_FOLDER, str(year))

    download_and_process_s2s_data(
        year,
        month,
        day,
        delete_grib=not args.no_delete_grib,
        RAW_FOLDER=OUT_FOLDER,
        PROC_FOLDER=FULL_OUT_FOLDER,
    )