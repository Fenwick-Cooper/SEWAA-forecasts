#!/usr/bin/env python

# Python script to run a set of historical forecasts.
#
# To run this script:
#
#       conda activate tf215gpu
#       python run_some_forecasts.py -h

import argparse
import subprocess
from datetime import datetime, timedelta
import sys

# Parse arguments to this script
def parseArguments():

    parser = argparse.ArgumentParser(description="""Before use:
 
    conda activate tf215gpu

 Usage examples:

 Run all 6h and 24h forecasts between 2026-02-10 and 2026-02-20
    python run_some_forecasts.py --start_date 20260210 --end_date 20260220
    
 Run only 6h forecasts initialised at 00:00 UTC
    python run_some_forecasts.py --start_date 20260210 --end_date 20260220 --run_6h_forecasts_starting_00
                                     
 Run only 6h forecasts initialised at 06:00 UTC
    python run_some_forecasts.py --start_date 20260210 --end_date 20260220 --run_6h_forecasts_starting_06

 Run only 6h forecasts initialised at 12:00 UTC
    python run_some_forecasts.py --start_date 20260210 --end_date 20260220 --run_6h_forecasts_starting_12
                                     
 Run only 6h forecasts initialised at 18:00 UTC
    python run_some_forecasts.py --start_date 20260210 --end_date 20260220 --run_6h_forecasts_starting_18
                                     
 Run only 24h forecasts initialised at 00:00 UTC
    python run_some_forecasts.py --start_date 20260210 --end_date 20260220 --run_24h_forecasts_starting_00

 Delete forecasts after computing histograms
    python run_some_forecasts.py --start_date 20260210 --end_date 20260220 --delete_forecasts
                                     
 Disable ELR when running forecasts
    python run_some_forecasts.py --start_date 20260210 --end_date 20260220 --disable_ELR

 Arguments can be combined. For example:
 Run only 24h forecasts initialised at 00:00 UTC, deleting forecasts after computing histograms
    python run_some_forecasts.py --start_date 20260210 --end_date 20260220 --run_24h_forecasts_starting_00 --delete_forecasts 
    """, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--start_date', help='First initialisation date (YYYYMMDD)',default=None,type=str,required=True)
    parser.add_argument('--end_date', help='Run forecasts up to this date (YYYYMMDD)',default=None,type=str,required=True)    
    parser.add_argument('--run_6h_forecasts_starting_00', help='Run only 6h forecasts initialised at 00:00 UTC',nargs='*',type=str)
    parser.add_argument('--run_6h_forecasts_starting_06', help='Run only 6h forecasts initialised at 00:00 UTC',nargs='*',type=str)
    parser.add_argument('--run_6h_forecasts_starting_12', help='Run only 6h forecasts initialised at 00:00 UTC',nargs='*',type=str)
    parser.add_argument('--run_6h_forecasts_starting_18', help='Run only 6h forecasts initialised at 00:00 UTC',nargs='*',type=str)
    parser.add_argument('--run_24h_forecasts_starting_00', help='Run only 24h forecasts initialised at 00:00 UTC',nargs='*',type=str)
    parser.add_argument('--delete_forecasts', help='Should forecasts be deleted',nargs='*',type=str)
    parser.add_argument('--disable_ELR', help='If this option is selected ELR forecasts are not run',nargs='*',type=str)
    args = parser.parse_args()
    
    # Parse the start date
    if (args.start_date is not None):
    
        if (len(args.start_date) != 8):
            print("ERROR: Incorrect date.")
            parser.print_help()
            sys.exit()
        
        year = int(args.start_date[0:4])
        month = int(args.start_date[4:6])
        day = int(args.start_date[6:8])

        start_date = datetime(year=year, month=month, day=day)
    
    # Parse the end date
    if (args.end_date is not None):
    
        if (len(args.end_date) != 8):
            print("ERROR: Incorrect date.")
            parser.print_help()
            sys.exit()
        
        year = int(args.end_date[0:4])
        month = int(args.end_date[4:6])
        day = int(args.end_date[6:8])

        end_date = datetime(year=year, month=month, day=day)
    
    # Parse run_6h_forecasts_starting_00
    run_6h_forecasts_starting_00 = False  # Default
    if (args.run_6h_forecasts_starting_00 is not None):
        run_6h_forecasts_starting_00 = True

    # Parse run_6h_forecasts_starting_06
    run_6h_forecasts_starting_06 = False  # Default
    if (args.run_6h_forecasts_starting_06 is not None):
        run_6h_forecasts_starting_06 = True

    # Parse run_6h_forecasts_starting_12
    run_6h_forecasts_starting_12 = False  # Default
    if (args.run_6h_forecasts_starting_12 is not None):
        run_6h_forecasts_starting_12 = True

    # Parse run_6h_forecasts_starting_18
    run_6h_forecasts_starting_18 = False  # Default
    if (args.run_6h_forecasts_starting_18 is not None):
        run_6h_forecasts_starting_18 = True

    # Parse run_24h_forecasts_starting_00
    run_24h_forecasts_starting_00 = False  # Default
    if (args.run_24h_forecasts_starting_00 is not None):
        run_24h_forecasts_starting_00 = True

    # If there are no specific forecasts to run
    if (run_6h_forecasts_starting_00 == False and run_6h_forecasts_starting_06 == False and
        run_6h_forecasts_starting_12 == False and run_6h_forecasts_starting_18 == False and
        run_24h_forecasts_starting_00 == False):

        # Run them all
        run_6h_forecasts_starting_00 = True
        run_6h_forecasts_starting_06 = True
        run_6h_forecasts_starting_12 = True
        run_6h_forecasts_starting_18 = True
        run_24h_forecasts_starting_00 = True

    # Parse delete_forecasts
    delete_forecasts = False  # Default
    if (args.delete_forecasts is not None):
        delete_forecasts = True
        
    # Parse disable_ELR
    disable_ELR = False  # Default
    if (args.disable_ELR is not None):
        disable_ELR = True
    
    return (start_date, end_date,
            run_6h_forecasts_starting_00, run_6h_forecasts_starting_06,
            run_6h_forecasts_starting_12, run_6h_forecasts_starting_18,
            run_24h_forecasts_starting_00,
            delete_forecasts, disable_ELR)


# 6h accumulations
def run_6h_accumulation_forecasts(start_date, end_date,
                                  run_6h_forecasts_starting_00, run_6h_forecasts_starting_06,
                                  run_6h_forecasts_starting_12, run_6h_forecasts_starting_18,
                                  disable_ELR, delete_forecasts):
    
    # Keep forecasts by default
    delete_forecasts_str = ""
    if (delete_forecasts):
        # But if delete_forecasts is True, set them to be deleted
        delete_forecasts_str = "--delete_forecasts Y"

    # Run all 6h forecasts
    d = start_date
    while (d < end_date):
        
        # If we need to check for this hour
        if ((d.hour == 0 and run_6h_forecasts_starting_00) or 
            (d.hour == 6 and run_6h_forecasts_starting_06) or 
            (d.hour == 12 and run_6h_forecasts_starting_12) or 
            (d.hour == 18 and run_6h_forecasts_starting_18)):

            # Run the 6h forecast
            print(f"Running: run_forecast.py --accumulation 6h --date {d.year}{d.month:02d}{d.day:02d} --time {d.hour:02d}{d.minute:02d} {delete_forecasts_str}")
            run_command = ["python", f"run_forecast.py",
                            "--accumulation", "6h",
                            "--date", f"{d.year}{d.month:02d}{d.day:02d}",
                            "--time", f"{d.hour:02d}{d.minute:02d}"]
            if (disable_ELR):
                run_command.append("--disable_ELR")
            if (delete_forecasts):
                run_command.append(["--delete_forecasts", "Y"])
            subprocess.call(run_command)
        
        # Move to the next forecast
        d += timedelta(hours=6)


# 24h accumulations
def run_24h_accumulation_forecasts(start_date, end_date, disable_ELR, delete_forecasts):
    
    # Keep forecasts by default
    delete_forecasts_str = ""
    if (delete_forecasts):
        # But if delete_forecasts is True, set them to be deleted
        delete_forecasts_str = "--delete_forecasts"

    # Run all 24h forecasts
    d = start_date
    while (d < end_date):
        
        # Check for the 6h forecast
        print(f"Running: run_forecast.py --accumulation 24h --date {d.year}{d.month:02d}{d.day:02d} --time {d.hour:02d}{d.minute:02d} {delete_forecasts_str}")
        run_command = ["python", f"run_forecast.py",
                       "--accumulation", "24h",
                       "--date", f"{d.year}{d.month:02d}{d.day:02d}",
                       "--time", f"{d.hour:02d}{d.minute:02d}"]
        if (disable_ELR):
            run_command.append("--disable_ELR")
        if (delete_forecasts):
            run_command.append(["--delete_forecasts", "Y"])
        subprocess.call(run_command)
        
        # Move to the next forecast
        d += timedelta(days=1)


if __name__=='__main__':

    # Parse arguments to this script
    (start_date, end_date,
     run_6h_forecasts_starting_00, run_6h_forecasts_starting_06,
     run_6h_forecasts_starting_12, run_6h_forecasts_starting_18,
     run_24h_forecasts_starting_00,
     delete_forecasts, disable_ELR) = parseArguments()

    # Start running
    if (run_6h_forecasts_starting_00 or run_6h_forecasts_starting_06 or
        run_6h_forecasts_starting_12 or run_6h_forecasts_starting_18):
        run_6h_accumulation_forecasts(start_date, end_date,
                                      run_6h_forecasts_starting_00, run_6h_forecasts_starting_06,
                                      run_6h_forecasts_starting_12, run_6h_forecasts_starting_18,
                                      disable_ELR, delete_forecasts)
        
    if (run_24h_forecasts_starting_00):
        run_24h_accumulation_forecasts(start_date, end_date, disable_ELR, delete_forecasts)
