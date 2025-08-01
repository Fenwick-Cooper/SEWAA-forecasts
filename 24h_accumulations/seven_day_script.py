# Python script to run continuously in the background to download IFS data.
# In the calling terminal must run
#    eval $(ssh-agent -s)
#    ssh-add ~/.ssh/id_rsa_kenya2
# before running
#    nohup python -u seven_day_script.py > nohup.out &

# If running on my ubuntu desktop
#    screen -S processIFS
#    longjob 365 python -u daily_script.py > nohup.out
# Then detach from screen with "ctrl+A" "D"

import subprocess
from datetime import datetime, timedelta, UTC
import schedule
import time

# This function is run every six hours at some point between
# 08:01 and 08:02, 14:01 and 14:02, 20:01 and 20:02, 02:01 and 02:02.
def sixHourly_7d_work(time):
    print(f"Running sixHourly_7d_work(time = {time}) function at {datetime.now()}")
    
    # Download the .grib data from ECMWF
    print(f"Running download_7d.py {time}")
    subprocess.call(["python", f"download_7d.py", f"{time}"])

    # Convert to a lon/lat grid in the ICPAC region
    print(f"Running IFS_to_ICPAC_region.py {time}")
    subprocess.call(["python", "IFS_to_ICPAC_region.py", f"{time}"])

    # Upload to the remote server
    print("Uploading to remote server")
    if (time == 18):
        todays_date_str = (datetime.now()-timedelta(days=1)).strftime("%Y%m%d")
    else:
        todays_date_str = datetime.now().strftime("%Y%m%d")
    data_path = "/home/c/cooperf/IFS/ICPAC_data/operational/7d/processed_IFS"
    subprocess.call(["scp", f"{data_path}/IFS_{todays_date_str}_{time:02d}Z.nc", "gbmc@136.156.130.165:/data/Operational_7d"])

    # Run cGAN on this data
    print(f"Running forecast_date.py {todays_date_str}")
    run_dir = "/home/c/cooperf/data/cGAN/ICPAC/operational/Mvua_kubwa/ensemble-cgan/dsrnngan/"
    subprocess.call(["python", "forecast_date.py", todays_date_str], cwd=run_dir)

    # Compute the histogram counts
    subprocess.call(["python", "forecast2histogram_7d.py", todays_date_str])


# Run up to one minute after 08:01 every day
# This uses the local time which currently is BST (UTC+1)
schedule.every().day.at("09:01").do(sixHourly_7d_work, time = 0)
# The fix for dealing with daylight saving time I believe is (NEEDS TESTING):
# schedule.every().day.at("08:01", UTC).do(sixHourly_7d_work, time = 0)

# Run up to one minute after 14:01 every day
#schedule.every().day.at("14:01").do(sixHourly_7d_work, time = 6)

# Run up to one minute after 20:01 every day
#schedule.every().day.at("20:01").do(sixHourly_7d_work, time = 12)

# Run up to one minute after 02:01 every day
#schedule.every().day.at("02:01").do(sixHourly_7d_work, time = 18)

while True:
    print(f"Checking shedule at {datetime.now()}")
    schedule.run_pending()
    time.sleep(60)
