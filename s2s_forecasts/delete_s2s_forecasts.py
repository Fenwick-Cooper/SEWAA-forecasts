from download_s2s_data import delete_grib_and_index
import os

def delete_forecast_files(year, month, day, lead_times_weeks, RAW_S2S_DATA_DIR, PROC_S2S_DATA_DIR, REG_MEAN_DATA_DIR, regionmask_name):
    """Delete forecast files for a given date and lead times.
    args:        year, month, day: date of forecast initialization
        lead_times_weeks: list of lead weeks to delete, e.g. [1, 2, 3]
        forecast_folder: folder where forecasts are stored
    """

    #First see if grib still exists, and if so delete it and any matching index files
    grib_fname = os.path.join(RAW_S2S_DATA_DIR, f"{year}-{month:02d}-{day:02d}_tprate.grib")
    delete_grib_and_index(grib_fname)

    for week in lead_times_weeks:
        proc_fname = os.path.join(PROC_S2S_DATA_DIR, f"{year}-{month:02d}-{day:02d}_tp_meanstd_{week}wklead.nc")

        #Check if file exists before trying to delete
        if os.path.isfile(proc_fname):
            print("Forecast file found: ", proc_fname)
            print("Deleting forecast file: ", proc_fname)
            try: #Delete file
                os.remove(proc_fname)
                print("Deleted processed file: ", proc_fname)
            except Exception as e:
                print(f"Error deleting file {proc_fname}: {e}")

        reg_mean_fname = os.path.join(REG_MEAN_DATA_DIR, f"{year}-{month:02d}-{day:02d}_tp_meanstd_{week}wklead_{regionmask_name.split('.')[0]}.nc")

        #Check if file exists before trying to delete
        if os.path.isfile(reg_mean_fname):
            print("Regional mean file found: ", reg_mean_fname)
            print("Deleting regional mean file: ", reg_mean_fname)
            try: #Delete file
                os.remove(reg_mean_fname)
                print("Deleted regional mean file: ", reg_mean_fname)
            except Exception as e:
                print(f"Error deleting file {reg_mean_fname}: {e}")
