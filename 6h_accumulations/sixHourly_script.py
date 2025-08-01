# Python script to run continuously in the background to download IFS data.
# In the calling terminal must run
#    conda activate tf215gpu
#    eval $(ssh-agent -s)
#    ssh-add ~/.ssh/id_rsa_kenya2
# before running
#    nohup python -u sixHourly_script.py > nohup.out 2>&1 &

# If running on my ubuntu desktop
#    screen -S processIFS
#    longjob 365 python -u daily_script.py > nohup.out
# Then detach from screen with "ctrl+A" "D"

import subprocess
from datetime import datetime, timedelta, UTC
import schedule
import time

# This function is run every six hours at some point between
# 07:01 and 07:02, 13:01 and 13:02, 19:01 and 19:02, 01:01 and 01:02.
def sixHourly_work(time):
    print(f"Running sixHourly_work(time = {time}) function at {datetime.now()}")
    
    # Download the .grib data from ECMWF
    print(f"Running download_sixHourly.py {time}")
    subprocess.call(["python", f"download_sixHourly.py", f"{time}"])

    # Convert to a lon/lat grid in the ICPAC region
    print(f"Running IFS_to_ICPAC_region.py {time}")
    subprocess.call(["python", "IFS_to_ICPAC_region.py", f"{time}"])

    # Upload to the remote server
    print("Uploading to remote server")
    if (time == 18):
        todays_date_str = (datetime.now()-timedelta(days=1)).strftime("%Y%m%d")
    else:
        todays_date_str = datetime.now().strftime("%Y%m%d")
    data_path = "/home/c/cooperf/IFS/ICPAC_data/operational/processed_IFS"
    subprocess.call(["scp", f"{data_path}/IFS_{todays_date_str}_{time:02d}Z.nc", "gbmc@136.156.130.165:/data/Operational"])

    # Run cGAN on this data
    print(f"Running forecast_date.py {todays_date_str} {time:02d}:00")
    run_dir = "/home/c/cooperf/data/cGAN/ICPAC/operational/Jurre_brishti/ensemble-cgan/dsrnngan"
    subprocess.call(["python", "forecast_date.py", todays_date_str, str(time)], cwd=run_dir)

    # Compute the histogram counts
    print("Computing histogram counts")
    subprocess.call(["python", "forecast2histogram.py", todays_date_str, str(time)])
    
    # Delete the cGAN forecast
    if (time != 0):
        forecast_dir = "/home/c/cooperf/IFS/ICPAC_data/operational/GAN_forecasts_large"
        file_to_delete = f"{forecast_dir}/GAN_{todays_date_str}_{time:02d}Z.nc"
        print(f"Deleting {file_to_delete}")
        subprocess.call(["rm", file_to_delete])

# Run up to one minute after 07:01 every day
# This uses the local time which currently is BST (UTC+1)
schedule.every().day.at("08:01").do(sixHourly_work, time = 0)
# The fix for dealing with daylight saving time I believe is (NEEDS TESTING):
# schedule.every().day.at("07:01", UTC).do(sixHourly_work, time = 0)

# Run up to one minute after 13:01 every day
schedule.every().day.at("14:01").do(sixHourly_work, time = 6)

# Run up to one minute after 19:01 every day
schedule.every().day.at("20:01").do(sixHourly_work, time = 12)

# Run up to one minute after 01:01 every day
schedule.every().day.at("02:01").do(sixHourly_work, time = 18)

while True:
    print(f"Checking shedule at {datetime.now()}")
    schedule.run_pending()
    time.sleep(60)
