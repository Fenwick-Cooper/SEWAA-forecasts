#!/usr/bin/env python

# Python script to start running forecasts automatically.
#
# To run this script:
#
#       conda activate tf215gpu
#       python start_forecasting.py
#
# Fault tolerance is delegated to run_forecasts.py.
# run_forecasts.py checks for existing files.

import argparse
import subprocess
from datetime import datetime, timedelta
import time
import schedule

# Number of minutes to wait before checking for another forecast
minutes_to_wait = 15

# Number of minutes to wait before checking the schedule
minutes_between_schedule_checks = 1

# Number of days in the past to check for forecasts. Starting at 00:00 every day.
days_to_check = 2
# If the script was stopped, people might want these forecasts to be computed when it is
# restarted again.

# Minimum wait time for IFS data starting when it was initialised in hours.
IFS_wait_hours_for_6h_accumulations = 7
IFS_wait_minutes_for_6h_accumulations = 30
IFS_wait_hours_for_24h_accumulations = 9
IFS_wait_minutes_for_24h_accumulations = 30

#Not sure on these values -- guessed for now
IFS_wait_hours_for_7d_accumulations = 12
IFS_wait_minutes_for_7d_accumulations = 30


# 6h accumulations
def run_6h_accumulation_forecasts():

    # Time of most recent forecast to check for
    d_end = datetime.utcnow() - (timedelta(hours = IFS_wait_hours_for_6h_accumulations) +
                                 timedelta(minutes = IFS_wait_minutes_for_6h_accumulations))
    
    # Time of first forecast to check for
    d_start = d_end - timedelta(days = days_to_check)
    
    # Always start at 00:00
    d_start = datetime(d_start.year, d_start.month, d_start.day)
    
    # Run all 6h forecasts days_to_check days in the past
    d = d_start
    while (d < d_end):
        
        # Check for the 6h forecast
        print(f"Running: run_forecast.py --accumulation 6h --date {d.year}{d.month:02d}{d.day:02d} --time {d.hour:02d}{d.minute:02d} --delete_forecasts Y")
        subprocess.call(["python", f"run_forecast.py",
                         "--accumulation", "6h",
                         "--date", f"{d.year}{d.month:02d}{d.day:02d}",
                         "--time", f"{d.hour:02d}{d.minute:02d}",
                         "--delete_forecasts", "Y"])
        
        # Move to the next forecast
        d += timedelta(hours=6)


# 24h accumulations
def run_24h_accumulation_forecasts():

    # Time of most recent forecast to check for
    d_end = datetime.utcnow() - (timedelta(hours = IFS_wait_hours_for_24h_accumulations) +
                                 timedelta(minutes = IFS_wait_minutes_for_24h_accumulations))
    
    # Time of first forecast to check for
    d_start = d_end - timedelta(days = days_to_check)
    
    # Always start at 00:00
    d_start = datetime(d_start.year, d_start.month, d_start.day)
    
    # Run all 6h forecasts days_to_check days in the past
    d = d_start
    while (d < d_end):
        
        # Check for the 6h forecast
        print(f"Running: run_forecast.py --accumulation 24h --date {d.year}{d.month:02d}{d.day:02d} --time {d.hour:02d}{d.minute:02d} --delete_forecasts Y")
        subprocess.call(["python", f"run_forecast.py",
                         "--accumulation", "24h",
                         "--date", f"{d.year}{d.month:02d}{d.day:02d}",
                         "--time", f"{d.hour:02d}{d.minute:02d}",
                         "--delete_forecasts", "Y"])
        
        # Move to the next forecast
        d += timedelta(days=1)


# 7d accumulations
def run_7d_accumulation_forecasts():

    # Time of most recent forecast to check for
    d_end = datetime.utcnow() - (timedelta(hours = IFS_wait_hours_for_7d_accumulations) +
                                 timedelta(minutes = IFS_wait_minutes_for_7d_accumulations))
    
    # Time of first forecast to check for
    d_start = d_end - timedelta(days = days_to_check)
    
    # Always start at 00:00
    d_start = datetime(d_start.year, d_start.month, d_start.day)
    
    # Run all 7d forecasts days_to_check days in the past
    d = d_start
    while (d < d_end):
        
        # Check for the 7d forecast
        print(f"Running: run_forecast.py --accumulation 7d --date {d.year}{d.month:02d}{d.day:02d} --delete_forecasts Y")
        subprocess.call(["python", f"run_forecast.py",
                         "--accumulation", "7d",
                         "--date", f"{d.year}{d.month:02d}{d.day:02d}",
                         "--delete_forecasts", "Y"])
        
        # Move to the next forecast
        d += timedelta(days=1)


def run_all_forecasts():
    run_6h_accumulation_forecasts()
    run_24h_accumulation_forecasts()
    run_7d_accumulation_forecasts()


if __name__=='__main__':

    # There are no command line arguments, but the user might want help anyway.
    parser = argparse.ArgumentParser(description="""Python script to start running forecasts automatically.

To run this script:

      conda activate tf215gpu
      python start_forecasting.py
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.parse_args()

    # Start immediately
    run_all_forecasts()

    # Schedule the forecasts to run every minutes_to_wait minutes.
    schedule.every(minutes_to_wait).minutes.do(run_all_forecasts)
    
    # Check the schedule every minutes_between_schedule_checks minutes.
    while True:
        print(f"Checking shedule at {datetime.utcnow()} UTC.")
        schedule.run_pending()
        time.sleep(minutes_between_schedule_checks * 60)
