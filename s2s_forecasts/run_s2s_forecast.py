#imports
import argparse
from datetime import datetime

from ensure_required_packages import ensure_packages
from download_s2s_data import download_and_process_s2s_data
from make_s2s_regional_means import make_regional_means
from make_s2s_forecast import produce_s2s_idr_forecasts
from delete_s2s_forecasts import delete_forecast_files

#Define directories
RAW_S2S_DATA_DIR = "./s2s_data/raw"
PROC_S2S_DATA_DIR = "./s2s_data/processed"
REG_MEAN_DATA_DIR = "./s2s_data/regional_means"
SHAPEFILE_DIR = "./shapefiles"
FCST_OUT_DIR = "../interface/view_forecasts/data/counts_s2s"
# FCST_OUT_DIR = "/nf2/web/rain/ICPAC/operational/s2s_forecasts/s2s_counts" #FOR OXFORD USE


#Global settings
lead_time_weeks = [1,2,3]
data_source = 'Oxford'
SHAPEFILE_NAME = "admin1_merged_KeEtRwUg.gpkg"

def parse_args():
    parser = argparse.ArgumentParser(description="Run S2S download, regional means, and forecast pipeline.")
    parser.add_argument(
        "--date",
        required=True,
        help="Run date in YYYYMMDD format, e.g. 20240527",
    )
    parser.add_argument('--delete_forecasts', help='Should forecasts be deleted or not (Y/N)',default=None,type=str)
    return parser.parse_args()

def main():
    #Parse command line arguments
    args = parse_args()
    date_str = args.date

    try:
        date = datetime.strptime(date_str, "%Y%m%d")
        year, month, day = date.year, date.month, date.day
    except ValueError:
        print("Invalid date format. Please use YYYYMMDD.")
        return
    
    # Parse delete_forecasts
    delete_forecasts = False  # Default
    if (args.delete_forecasts is not None):
        
        if ((args.delete_forecasts == "T") or (args.delete_forecasts == "t") or
            (args.delete_forecasts == "Y") or (args.delete_forecasts == "y")):
            delete_forecasts = True

    #Check package requirements
    ensure_packages()

    #Run download script to get S2S data for this date
    download_and_process_s2s_data(
        year,
        month,
        day,
        data_source=data_source,
        lead_time_weeks=lead_time_weeks,
        delete_grib=True,
        RAW_FOLDER=RAW_S2S_DATA_DIR,
        PROC_FOLDER=PROC_S2S_DATA_DIR
        )

    #Make the regional means for this date
    make_regional_means(
        year,
        month,
        day,
        lead_time_weeks=lead_time_weeks,
        IN_FOLDER=PROC_S2S_DATA_DIR,
        OUT_FOLDER=PROC_S2S_DATA_DIR,
        SHAPEFILE_FOLDER=SHAPEFILE_DIR,
        shapefile_name=SHAPEFILE_NAME,
        region_subset=None
    )

    #Run forecast script to make IDR forecast for this date and save histograms
    produce_s2s_idr_forecasts(
        year,
        month,
        day,
        lead_time_weeks=lead_time_weeks,
        bins='default',
        IDR_MODEL_FOLDER='./idr_models',
        shapefile_name=SHAPEFILE_NAME,
        REGIONAL_MEAN_FOLDER=REG_MEAN_DATA_DIR,
        OUT_FOLDER=FCST_OUT_DIR
    )

    #Delete forecast files if requested
    if delete_forecasts:
        delete_forecast_files(
        year,
        month,
        day,
        lead_time_weeks,
        RAW_S2S_DATA_DIR,
        PROC_S2S_DATA_DIR,
        REG_MEAN_DATA_DIR,
        SHAPEFILE_NAME
        )

if __name__ == "__main__":
    main()