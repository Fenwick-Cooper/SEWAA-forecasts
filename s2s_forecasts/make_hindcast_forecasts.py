
from pathlib import Path
import os
from datetime import datetime
from make_s2s_regional_means import make_regional_means
from make_s2s_forecast import produce_s2s_idr_forecasts
import argparse
import xarray as xr
from datetime import datetime
import pandas as pd

# Paths are anchored to this file, not the current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent          # .../s2s_forecasts
PROJECT_ROOT = SCRIPT_DIR.parent                       # .../

# Define directories
RAW_S2S_DATA_DIR = SCRIPT_DIR / "s2s_data" / "raw"
PROC_S2S_DATA_DIR = SCRIPT_DIR / "s2s_data" / "processed"
REG_MEAN_DATA_DIR = SCRIPT_DIR / "s2s_data" / "regional_means"
SHAPEFILE_DIR = SCRIPT_DIR / "regionmasks"
IDR_MODEL_FOLDER = SCRIPT_DIR / "idr_models"
FCST_OUT_DIR = PROJECT_ROOT / "interface" / "view_forecasts" / "data" / "counts_s2s"

#Global settings
regionmask_name = "admin1_merged_KeEtRwUg_region_masks_1x1_bool.nc"

os.makedirs(RAW_S2S_DATA_DIR, exist_ok=True)
os.makedirs(PROC_S2S_DATA_DIR, exist_ok=True)
os.makedirs(REG_MEAN_DATA_DIR, exist_ok=True)
os.makedirs(FCST_OUT_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Run S2S download, regional means, and forecast pipeline.")
    parser.add_argument(
        "--date",
        required=True,
        help="TARGET date (start of week) in YYYYMMDD format, e.g. 20240527",
    )
    parser.add_argument('--delete_forecasts', help='Should forecasts be deleted or not (Y/N)',default=None,type=str)
    parser.add_argument(
        "--lead",
        required=True,
        help="Lead time as integer",
    )
    return parser.parse_args()

def main():
    #Parse command line arguments
    args = parse_args()
    date_str = args.date
    lead_times_weeks = [int(args.lead)]

    try:
        date = datetime.strptime(date_str, "%Y%m%d")
        year, month, day = date.year, date.month, date.day
    except ValueError:
        print("Invalid date format. Please use YYYYMMDD.")
        return

    #Process hindcast data into right place
    for lead in lead_times_weeks:
        print(f"Processing data for lead {lead} weeks")
        hind = xr.load_dataset(f"/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/processed_s2s_data/tprate_sfc_{lead-1}wklead.nc")
        hind = hind.sel(
            time=datetime(year,month,day)
        )
        out = hind.mean("number").rename({"precipitation": "tp_mean"})
        out["tp_std"] = hind.precipitation.std("number")
        out.attrs['units'] = 'mm/week'
        out.attrs['type'] = 'from hindcast'
        out["step"] = out["step"] - pd.Timedelta(days=7) #Change step convention to be start of accumulation period, not end
        out = out.squeeze().assign_coords(valid_time=out.time)
        out["time"] = out["time"] - pd.Timedelta(days=7*(lead-1))
        out_year = out.time.dt.year.values
        out_month = out.time.dt.month.values
        out_day = out.time.dt.day.values
        os.makedirs(f"/home/m/matthewwright/SEWAA-forecasts/s2s_forecasts/s2s_data/processed/{out_year}", exist_ok=True)
        out.to_netcdf(f"/home/m/matthewwright/SEWAA-forecasts/s2s_forecasts/s2s_data/processed/{out_year}/{out_year}-{out_month:02d}-{out_day:02d}_tp_meanstd_{lead}wklead.nc")
        print(f"Saved hindcast data to /home/m/matthewwright/SEWAA-forecasts/s2s_forecasts/s2s_data/processed/{out_year}/{out_year}-{out_month:02d}-{out_day:02d}_tp_meanstd_{lead}wklead.nc")

    #Make the regional means for this date
    make_regional_means(
        out_year,
        out_month,
        out_day,
        lead_times_weeks=lead_times_weeks,
        IN_FOLDER=f"{str(PROC_S2S_DATA_DIR)}/{out_year}",
        OUT_FOLDER=f"{str(REG_MEAN_DATA_DIR)}/{out_year}",
        REGIONMASK_FOLDER=str(SHAPEFILE_DIR),
        regionmask_name=regionmask_name,
        region_subset=None,
    )

    #Run forecast script to make IDR forecast for this date and save histograms
    produce_s2s_idr_forecasts(
        out_year,
        out_month,
        out_day,
        lead_times_weeks=lead_times_weeks,
        bins="default",
        IDR_MODEL_FOLDER=str(IDR_MODEL_FOLDER),
        regionmask_name=regionmask_name,
        REGIONAL_MEAN_FOLDER=f"{str(REG_MEAN_DATA_DIR)}/{out_year}",
        OUT_FOLDER=f"{str(FCST_OUT_DIR)}/{out_year}",
    )

if __name__ == "__main__":
    main()