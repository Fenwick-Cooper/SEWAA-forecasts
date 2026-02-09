#!/usr/bin/env python

# Python script to run the forecasts for a given date and time
#
# Requirements:
#
#    Access to the ECMWF computer gbmc@136.156.130.165.
#    Conda environment for cGAN installed and activated.
#
# Before use:
# 
#    conda activate tf215gpu
#
# Usage examples:
#
# Run todays 6h forecasts initialised at 0000
#    python run_forecast.py
#    python run_forecast.py --accumulation 6h
#
# Run the 6h forecasts for the 10th of February 2025 initialised at 1800
#    python run_forecast.py --date 20250210 --time 1800
#
# Run the most recent 24h forecasts
#    python run_forecast.py --accumulation 24h
#
# Run the 24h forecasts for the 10th of February 2025
#    python run_forecast.py --accumulation 24h --date 20250210
#
# Run todays 6h forecasts initialised at 0000 and delete the forecasts once
# statistics have been computed
#    python run_forecast.py --delete_forecasts Y

import argparse
import sys
import os
import subprocess
import pathlib
import datetime


# Parse arguments to this script
def parseArguments():

    parser = argparse.ArgumentParser(description="""Requirements:

    Access to the ECMWF computer gbmc@136.156.130.165.
    Conda environment for cGAN installed and activated.

 Before use:
 
    conda activate tf215gpu

 Usage examples:

 Run todays 6h forecasts initialised at 0000
    python run_forecast.py
    python run_forecast.py --accumulation 6h

 Run the 6h forecasts for the 10th of February 2025 initialised at 1800
    python run_forecast.py --date 20250210 --time 1800

 Run the most recent 24h forecasts
    python run_forecast.py --accumulation 24h

 Run the 24h forecasts for the 10th of February 2025
    python run_forecast.py --accumulation 24h --date 20250210  
    
 Run todays 6h forecasts initialised at 0000 and delete the forecasts once
 statistics have been computed
    python run_forecast.py --delete_forecasts Y  
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--accumulation', help='How long rainfall is accumulated for, either 6h or 24h',default=None,type=str)
    parser.add_argument('--date', help='Forecast initialisation date (YYYYMMDD)',default=None,type=str)
    parser.add_argument('--time', help='Forecast initialisation time (HHMM)',default=None,type=str)    
    parser.add_argument('--delete_forecasts', help='Should forecasts be deleted or not (Y/N)',default=None,type=str)
    #parser.add_argument('--disable_ELR', help='If this option is selected ELR forecasts are not run',nargs='*',type=str)
    args = parser.parse_args()
    
    # Parse the accumulation
    if (args.accumulation is not None):
        
        if (args.accumulation == '6h') or (args.accumulation == '6'):
            accumulation_time = 6
            
        elif (args.accumulation == '24h') or (args.accumulation == '24'):
            accumulation_time = 24
            
        else:
            print("ERROR: Incorrect accumulation.")
            print("       Available accumulations 6h, 24h.")
            parser.print_help()
            sys.exit()
        
    else:
        
        # Default 6h accumulation
        accumulation_time = 6
    
    # Parse the date
    if (args.date is not None):
    
        if (len(args.date) != 8):
            print("ERROR: Incorrect date.")
            parser.print_help()
            sys.exit()
        
        year = int(args.date[0:4])
        month = int(args.date[4:6])
        day = int(args.date[6:8])
        
    else:
    
        # Default today
        d = datetime.datetime.today()
        year = d.year
        month = d.month
        day = d.day
    
    # Parse the time
    if (args.time is not None):
    
        if (len(args.time) != 4):
            print("ERROR: Incorrect time.")
            parser.print_help()
            sys.exit()
            
        hour = int(args.time[0:2])
        minute = int(args.time[2:4])
        
        if (accumulation_time == 6):
            if (hour not in [0,6,12,18]) or (minute != 0):
                print("ERROR: Incorrect time.")
                print("       Available initialisation times are 0000, 0600, 1200, 1800 for 6h accumulations.")
                parser.print_help()
                sys.exit()
                
        elif (accumulation_time == 24):
            if (hour != 0):
                print("ERROR: Incorrect time.")
                print("       Available initialisation time is 0000 for 24h accumulations.")
                parser.print_help()
                sys.exit()
                
    else:
        
        # Default 0000
        hour = 0
        minute = 0
        
    # Parse delete_forecasts
    delete_forecasts = False  # Default
    if (args.delete_forecasts is not None):
        
        if ((args.delete_forecasts == "T") or (args.delete_forecasts == "t") or
            (args.delete_forecasts == "Y") or (args.delete_forecasts == "y")):
            delete_forecasts = True
        
    # Parse disable_ELR
    run_ELR = False  # ELR is currently not available for Madagascar north
#     run_ELR = True  # Default
#     if (args.disable_ELR is not None):
#         run_ELR = False
    
    return accumulation_time, year, month, day, hour, minute, delete_forecasts, run_ELR


# Checks that all of the histogram counts files for this date and time are there or not.
# Arguments:
#    counts_path - The directory to check.
#    date_str    - A string containing the initialisation date (YYYYMMDD).
#    hour        - An integer corresponding to initialisation hour of the day.
#    valid_hours - A list of the valid_hours that the forecast is computed at.
# Returns:
#    If all files that should be there, are there.
def check_counts_files(counts_path, date_str, hour, valid_hours):
    
    # Extract the year from date_str
    year = date_str[0:4]
    
    # Check to see if each file exists
    num_files_exist = 0
    for i in valid_hours:
        file_name = f"counts_{date_str}_{hour:02d}_{i}h.nc"
        exists = os.path.isfile(f"{counts_path}/{year}/{file_name}")
        if (exists):
            num_files_exist += 1
            # print(f"{counts_path}/{year}/{file_name} already exists.")
    
    # Does the number of files that exist equal the number that should exist?
    correct_num_files_exist = (num_files_exist == len(valid_hours))
    
    return correct_num_files_exist


# Checks that all of the ELR output files for this date and time are there or not.
# Returns:
#    If all files that should be there, are there.
def check_ELR_files(model_path, save_path, accumulation_time, countries, 
                    country_admin_regions, date, model='GAN', day=[1]):
    num_files_expected = len(ELR_countries)*len(day)
    num_files_actual = 0
    for country in ELR_countries:
        for d in day:
            if os.path.exists(\
                save_path+f'{accumulation_time}h_accumulations/{country}/{country_admin_regions[country]}/{model}_{date}_ELR_v{d}.nc'):
                num_files_actual+=1
    
    correct_num_files_exist = (num_files_expected==num_files_actual)
    return correct_num_files_exist


if __name__=='__main__':
    
    # Parse arguments to this script
    accumulation_time, year, month, day, hour, minute, delete_forecasts, run_ELR = parseArguments()
    
    print(f"Producing forecasts of {accumulation_time}h accumulations")
    print(f"initialised on {year}-{month:02d}-{day:02d} at {hour:02d}{minute:02d}.")
    
    # What are the valid hours of the forecast
    valid_hours_6h = [30, 36, 42, 48]
    valid_hours_24h = [6, 30, 54, 78, 102, 126, 150]
    
    # Shorthand
    date_str = f"{year}{month:02d}{day:02d}"
    time_str = f"{hour:02d}{minute:02d}"
    
    # The SEWAA-forecasts directory
    root_dir = "."
    
    # Where IFS data for 6 hour accumulations will be stored
    IFS_data_path_6h = f"{root_dir}/6h_accumulations/IFS_forecast_data"
    
    # Where IFS data for 24 hour accumulations will be stored
    IFS_data_path_24h = f"{root_dir}/24h_accumulations/IFS_forecast_data"
    
    # Where cGAN 6h forecasts will be stored
    cGAN_forecast_path_6h = f"{root_dir}/6h_accumulations/cGAN_forecasts"
    
    # Where cGAN 24h forecasts will be stored
    cGAN_forecast_path_24h = f"{root_dir}/24h_accumulations/cGAN_forecasts"
    
    # Where the 6h cGAN model forecast script is located
    cGAN_forecast_script_path_6h = f"{root_dir}/6h_accumulations/cGAN/dsrnngan"
    
    # Where the 6h cGAN model forecast script is located
    cGAN_forecast_script_path_24h = f"{root_dir}/24h_accumulations/cGAN/dsrnngan"

    # Where the ELR model script is located
    ELR_script_path = f"{root_dir}/ELR/"

    # Where the ELR models are located
    ELR_model_path = f"{root_dir}/ELR/models/"

    # Where the ELR predictions are saved
    ELR_predictions_path = f"{root_dir}/interface/ensemble_logistic_regression/ELR_predictions/"

    # Countries for ELR
    ELR_countries = ["Rwanda","Kenya","Ethiopia"]
    ELR_country_admin_regions = {"Rwanda":"county","Kenya":"subcounty","Ethiopia":"subcounty"}
    
    # Where all of the cGAN histogram counts will be stored
    cGAN_counts_path = f"{root_dir}/interface/view_forecasts/data"
    
    # Where the cGAN 6h histogram counts will be stored
    cGAN_counts_path_6h = f"{cGAN_counts_path}/counts_6h"
    
    # Where the cGAN 24h histogram counts will be stored
    cGAN_counts_path_24h = f"{cGAN_counts_path}/counts_24h"
    
    
    # Download from gbmc
    
    if (accumulation_time == 6):
        
        # Create the directory for the IFS downloads if it doesn't exist
        pathlib.Path(IFS_data_path_6h).mkdir(exist_ok=True)
        
        file_name = f"IFS_{date_str}_{hour:02d}Z.nc"
        
        # Check to see if the file is here first
        if os.path.isfile(f"{IFS_data_path_6h}/{file_name}"):
            print(f"{IFS_data_path_6h}/{file_name} already exists.")
    
        else:
            print(f"Copying 6h accumulation data, {file_name}, from gbmc")
            print(f"to {IFS_data_path_6h}/.")
            
            # Madagascar_north data can be obtained using curl
            cp = subprocess.run(["curl",
                                 f"http://megacorr.dynu.net/ICPAC/Madagascar_north_IFS/{file_name}",
                                 "-o",f"{IFS_data_path_6h}/{file_name}"])
            if (cp.returncode != 0):
                print(f"Unable to copy {file_name} from megacorr.dynu.net.")
                sys.exit()
        
    elif (accumulation_time == 24):
    
        # Create the directory for the IFS downloads if it doesn't exist
        pathlib.Path(IFS_data_path_24h).mkdir(exist_ok=True)
        
        file_name = f"IFS_{date_str}_{hour:02d}Z.nc"
        
        # Check to see if the file is here first
        if os.path.isfile(f"{IFS_data_path_24h}/{file_name}"):
            print(f"{IFS_data_path_24h}/{file_name} already exists.")
    
        else:
            print(f"Copying 24h accumulation data, {file_name}, from gbmc")
            print(f"to {IFS_data_path_24h}/.")
            
            # Madagascar_north data can be obtained using curl
            cp = subprocess.run(["curl",
                                 f"http://megacorr.dynu.net/ICPAC/Madagascar_north_IFS/{file_name}",
                                 "-o",f"{IFS_data_path_6h}/{file_name}"])
            if (cp.returncode != 0):
                print(f"Unable to copy {file_name} from megacorr.dynu.net.")
                sys.exit()
        
    else:
        # Error should have been caught before, but if it wasn't catch it now.
        print("ERROR: Incorrect accumulation time.")
        sys.exit()
    
    
    # Run cGAN on this data
        
    if (accumulation_time == 6):
        
        # Check to see if the counts are there first
        correct_num_counts_files = check_counts_files(cGAN_counts_path_6h,
                                                      date_str, hour, valid_hours_6h)
        
        
        # If the counts files are there and delete_forecasts is true don't run the forecasts
        if not (correct_num_counts_files and delete_forecasts):
            
            # Create the directory for the cGAN forecasts if it doesn't exist
            pathlib.Path(cGAN_forecast_path_6h).mkdir(exist_ok=True)
            
            file_name = f"GAN_{date_str}_{hour:02d}Z.nc"
            
            # Check to see if the forecast is there first
            if os.path.isfile(f"{cGAN_forecast_path_6h}/{file_name}"):
                print(f"{cGAN_forecast_path_6h}/{file_name} already exists.")
            
            else:
                print(f"Running 6h cGAN: forecast_date.py {date_str} {time_str}")
                run_dir = cGAN_forecast_script_path_6h
                subprocess.run(["python", "forecast_date.py", date_str, str(hour)], cwd=run_dir)
                
        else:
            print("Counts and ELR files exist and delete_forecasts is True; no forecast required.")
    
    elif (accumulation_time == 24):
        
        # Check to see if the counts are there first
        correct_num_counts_files = check_counts_files(cGAN_counts_path_24h,
                                                      date_str, hour, valid_hours_24h)
        
        # Check to see if the ELR files are there first
        correct_num_ELR_files = check_ELR_files(ELR_model_path, ELR_predictions_path, accumulation_time,
                                               ELR_countries, ELR_country_admin_regions, date_str)
        
        # If the counts and ELR files are there and delete_forecasts is true don't run the forecasts
        if not (correct_num_counts_files and correct_num_ELR_files and delete_forecasts):
            
            # Create the directory for the cGAN forecasts if it doesn't exist
            pathlib.Path(cGAN_forecast_path_24h).mkdir(exist_ok=True)
            
            # Perform a separate forecast for each lead time
            for lead_time_idx in range(7):
            
                file_name = f"GAN_{date_str}_{hour:02d}Z_v{lead_time_idx}.nc"
                
                # Check to see if the forecast is there first
                if os.path.isfile(f"{cGAN_forecast_path_24h}/{file_name}"):
                    print(f"{cGAN_forecast_path_24h}/{file_name} already exists.")
                
                else:
                    print(f"Running 24h cGAN: forecast_date.py {lead_time_idx} {date_str}")
                    run_dir = cGAN_forecast_script_path_24h
                    subprocess.run(["python", "forecast_date.py", str(lead_time_idx), date_str], cwd=run_dir)

        else:
            print("Counts and ELR files exist and delete_forecasts is True; no forecast required.")

    # Compute the histogram counts
    
    # Create the directory for the data if it doesn't exist
    pathlib.Path(cGAN_counts_path).mkdir(exist_ok=True)
    
    if (accumulation_time == 6):

        # Create the directory for the data if it doesn't exist
        pathlib.Path(cGAN_counts_path_6h).mkdir(exist_ok=True)
        
        # Check to see if the counts are there first
        correct_num_counts_files = check_counts_files(cGAN_counts_path_6h, date_str, hour, valid_hours_6h)
        
        # Check to see if the counts are there first
#         num_files_exist = 0
#         for i in [30,36,42,48]:
#             file_name = f"counts_{date_str}_{hour:02d}_{i}h.nc"
#             exists = os.path.isfile(f"{cGAN_counts_path_6h}/{year}/{file_name}")
#             if (exists):
#                 num_files_exist += 1
#                 print(f"{cGAN_counts_path_6h}/{year}/{file_name} already exists.")
#         
#         # If a file isn't there
#         if (num_files_exist < 4):
        if not correct_num_counts_files:
            print(f"Computing 6h histograms for {date_str} {time_str}.")
            
            # Create the directory for the year if it doesn't exist
            pathlib.Path(f"{cGAN_counts_path_6h}/{year}").mkdir(exist_ok=True)
            
            run_dir = f"{root_dir}/6h_accumulations"
            subprocess.run(["python", "forecast2histogram_lowRAM.py", date_str, str(hour)], cwd=run_dir)
        
        else:
            print("Histogram counts files already exist.")
        
    elif (accumulation_time == 24):
        
        # Create the directory for the data if it doesn't exist
        pathlib.Path(cGAN_counts_path_24h).mkdir(exist_ok=True)
        
        # Check to see if the counts are there first
        correct_num_counts_files = check_counts_files(cGAN_counts_path_24h, date_str, hour, valid_hours_24h)
        
#         # Check to see if the counts are there first
#         num_files_exist = 0
#         for i in [6,30,54,78,102,126,150]:
#             file_name = f"counts_{date_str}_{hour:02d}_{i}h.nc"
#             exists = os.path.isfile(f"{cGAN_counts_path_24h}/{year}/{file_name}")
#             if (exists):
#                 num_files_exist += 1
#                 print(f"{cGAN_counts_path_24h}/{year}/{file_name} already exists.")
#         
#         # If a file isn't there
#         if (num_files_exist < 7):
        if not correct_num_counts_files:
            print(f"Computing 24h histograms for {date_str} {time_str}.")
            
            # Create the directory for the year if it doesn't exist
            pathlib.Path(f"{cGAN_counts_path_24h}/{year}").mkdir(exist_ok=True)
            
            run_dir = f"{root_dir}/24h_accumulations"
            subprocess.run(["python", f"forecast2histogram_7d_lowRAM.py", date_str, str(hour)], cwd=run_dir)
        
        else:
            print("Histogram counts files already exist.")
    
    # Run ELR forecasts (only when time is 0)
    if ((hour == 0) and (run_ELR)):
                  
        if (accumulation_time == 24):
                                                          
            # Check to see if the ELR files are there first
            correct_num_ELR_files = check_ELR_files(ELR_model_path, ELR_predictions_path, accumulation_time,
                                               ELR_countries, ELR_country_admin_regions, date_str)
        
            # If the relevant files are there and delete_forecasts is true don't run the ELR
            if not correct_num_ELR_files:
        
                print("Running ELR 24h forecasts.")
                run_dir=ELR_script_path
                subprocess.run(["python", f"run_ELR.py", "--date", date_str, "--model", "GAN", 
                                "--accumulation", "24h_accumulations"], cwd=run_dir)
            
            else:
                print("ELR files already exist.")
                
    else:
        if (run_ELR):
            print("Skipping ELR forecasts for time not equal to 00:00.")
        else:
            print("ELR forecasts disabled.");
    
    # Update .JSON files for the interface. Overwrite if files exist.
    
    if (accumulation_time == 6):
    
        # Histogram counts
        print("Listing 6h counts for the interface.")
        run_dir = "6h_accumulations"
        subprocess.run(["python", f"find_available_dates.py"], cwd=run_dir)
        
    elif (accumulation_time == 24):
    
        # Histogram counts
        print("Listing 24h counts for the interface.")
        run_dir = "24h_accumulations"
        subprocess.run(["python", f"find_available_dates.py"], cwd=run_dir)
    
    # ELR forecasts run only when time is 0
    if ((hour == 0) and (run_ELR)):
        print("Listing ELR available dates.")
        run_dir = "ELR"
        subprocess.run(["python", f"ELR_available_dates.py"], cwd=run_dir)
    
    # Delete the cGAN forecast
    if delete_forecasts:
    
        if (accumulation_time == 6):
            file_to_delete = f"{cGAN_forecast_path_6h}/GAN_{date_str}_{hour:02d}Z.nc"
            if os.path.isfile(file_to_delete):  # Check to see if the forecast is there first
                print(f"Deleting {file_to_delete}")
                subprocess.run(["rm", file_to_delete])
        
        elif (accumulation_time == 24):
            for lead_time_idx in range(7):
                file_to_delete = f"{cGAN_forecast_path_24h}/GAN_{date_str}_{hour:02d}Z_v{lead_time_idx}.nc"
                if os.path.isfile(file_to_delete):  # Check to see if the forecast is there first
                    print(f"Deleting {file_to_delete}")
                    subprocess.run(["rm", file_to_delete])
    
    
    # Show that we are done (and haven't crashed)
    print("Script run_forecast.py is done!")
