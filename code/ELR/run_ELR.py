import os
import glob
import argparse
from datetime import datetime,timedelta
import warnings

import numpy as np
from cftime import date2num
from tqdm import tqdm
import xarray as xr
import joblib

from file_paths import paths
from helper_functions import get_geometry
import shapefile
from shapely.geometry.polygon import Polygon
import cartopy.io.shapereader as shpreader
import regionmask

from sklearn.exceptions import InconsistentVersionWarning
import time

OUT_PATH = paths['OUT_PATH']
FCST_PATH = paths['FCST_PATH']
MODEL_PATH = paths['MODEL_PATH']

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

countries = ['Kenya','Ethiopia','Rwanda']
county = {'Ethiopia':False,'Kenya':True,'Rwanda':True}
subcounty = {'Ethiopia':True,'Kenya':True,'Rwanda':False}

bounding_box = {'Ethiopia':(32.95418, 47.78942, 3.45, 14.837),'Kenya':(33.935689697, 41.5550830926, -4.559, 5.4877),
                'Rwanda':(28.87, 30.90, -2.81, -1.151)}

counties = None
subcounties = None

def time_standardised_to_since_1900(times,time_delta=None):
    times_standardised = []
    for day in times:
        day = datetime(day.astype('datetime64[D]').astype(object).year,
                     day.astype('datetime64[D]').astype(object).month,
                     day.astype('datetime64[D]').astype(object).day)
        if time_delta is not None:
            day = day+timedelta(hours=int(time_delta))
        times_standardised.append(date2num(day,units="hours since 1900-01-01 00:00:00.0"))
    return times_standardised

def get_model_output(date, model="GAN", day=1):

    if model == 'GAN':
        fcst_root_dir = f'{FCST_PATH}/24h_accumulations/cGAN_forecasts/'
    elif model=='IFS':
        fcst_root_dir = f'{FCST_PATH}/24h_accumulations/{model}_forecast_data/'

    ds_fcst = xr.open_dataset(fcst_root_dir+f'{model}_{date}_00Z_v{day}.nc')
    ds_fcst = ds_fcst.isel({"valid_time":0})

    return ds_fcst
    
def get_region(Location, geometry_all, ds):
    region_vectorised = regionmask.Regions(geometry_all, names=[Location])
    
    ## follows syntax of lat/lon
    mask_list = region_vectorised.mask_3D(ds.rename({'longitude':'lon','latitude':'lat'}))
    mask_list = np.ma.masked_invalid(mask_list)  

    temp = ds.precipitation.where(mask_list[0]).stack(latlon=('longitude','latitude'))
    fcst_valid_time = ds.fcst_valid_time
    ds_sel = temp.drop_vars(\
                ['latlon', 'longitude', 'latitude']
            ).assign_coords({'latlon':np.arange(temp[0].latlon.values.shape[0]),
                            'latitude':('latlon',[val[1] for val in temp[0].latlon.values]),
                            'longitude':('latlon',[val[0] for val in\
                                                   temp[0].latlon.values])}).dropna('latlon').reset_index('latlon')
    if Location=='Kajiado-East':
        ds_sel = ds_sel.isel({'latlon':slice(0,10)})
    ds_sel['fcst_valid_time']=fcst_valid_time

    return ds_sel

def get_model_checkpoint(Location, country, day, model):

    if country == 'Kenya':

        return '1'
        
    elif country=='Ethiopia':
        return str(day)

    else:
        return '1'
    

def get_ELR_predictions(logreg_model, model, ds_sel, day, longitude, latitude, Location, date, save_path):

    file_name = save_path+f'{model}_{date}_ELR_v{day}.nc'
    if os.path.exists(file_name):
        print('file already exists under,',file_name,'delete first then retry')
        return

    thresholds = np.asarray([key for key in logreg_model.keys()])[:4]
    latitude_reg = ds_sel.latitude.values
    longitude_reg = ds_sel.longitude.values
    date = ds_sel.time.values[0].astype('datetime64[D]').astype(object).strftime("%Y%M%d")

    lons, lats = np.meshgrid(np.unique(longitude_reg),np.unique(latitude_reg))
    predictions = np.full([1,1,thresholds.shape[0],lats.shape[0],lats.shape[1]],np.nan)
    
    mask = np.full([lats.shape[0],lats.shape[1]],False)
    
    for lat_reg,lon_reg in zip(latitude_reg,longitude_reg):
    
        idx_2d = np.ma.asarray(lats==lat_reg) * np.ma.asarray(lons==lon_reg)
        mask[idx_2d] = True
    
    lons_full, lats_full = np.meshgrid(np.unique(longitude),np.unique(latitude))
    mask_full = np.full([lats_full.shape[0],lats_full.shape[1]],False)
    for lat_f,lon_f in zip(latitude_reg,longitude_reg):
    
        idx_2d = np.ma.asarray(lats_full==lat_f) * np.ma.asarray(lons_full==lon_f)
        mask_full[idx_2d] = True
    
    X = np.sort(np.squeeze(24*ds_sel.values),axis=0).T

    if model == 'GAN':

        X = np.percentile(X, np.linspace(1,100,50), axis=1, method='weibull').T
    
    for i, threshold in enumerate(thresholds):
        predictions[0,0,i,mask] = logreg_model[threshold].predict_proba(X)[:,1]

    return predictions, mask_full, mask

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='IFS or GAN',default=None)
    parser.add_argument('--model', help='IFS or GAN',default='GAN',type=str)
    parser.add_argument('--day', help='lead time (in days)',action='append',nargs='+',default=None,type=int)
    parser.add_argument('--accumulation', help='6h- or 24h- accumulation',default="24h_accumulations",type=str)
    parser.add_argument('--store_netcdf', help='Store as netcdf',default=True,action='store_true')
    
    args = parser.parse_args()

    start_time = time.time()
    date = args.date
    if date is None:
        date = np.array(['2025-10-02'],dtype='datetime64[D]')[0].astype(object).strftime("%Y%m%d")#datetime.now().strftime("%Y%M%d")
    model = args.model
    day = args.day
    if day is None:
        day = np.arange(1,2)
    else:
        day = day[0]
    store_netcdf = args.store_netcdf
    accumulation = args.accumulation

    for country in countries:
        county_loop = county[country]
        subcounty_loop = subcounty[country]
        counties_loop = []
        subcounties_loop = []

        if counties == None and county[country]:
            counties_loop = glob.glob(MODEL_PATH+f'{country}/counties/*')
            counties_loop = [c.split('counties')[-1].replace('/','').replace('\\','').split('Region_bin_')[-1].split('_')[0] for c in counties_loop]
            for d in day:
                all_days_exist=True
                if not os.path.exists(OUT_PATH+f'{accumulation}/{country}/county/{model}_{date}_ELR_v{d}.nc'):
                    all_days_exist=False
            if all_days_exist:
                print(f"All ELR predictions already made for {country} at {accumulation}, skipping")
                continue   

        if len(counties_loop)==0 and county[country]:
            print("No county-level models found for:",country)
            county_loop=False
            skip_county=True     
    
        if subcounties == None and subcounty[country]:
            subcounties_loop = glob.glob(MODEL_PATH+f'{country}/subcounties/*')
            subcounties_loop = [c.split('subcounties')[-1].replace('/','').replace('\\','').split('Region_bin_')[-1].split('_')[0]\
                                for c in subcounties_loop]
            for d in day:
                all_days_exist=True
                if not os.path.exists(OUT_PATH+f'{accumulation}/{country}/county/{model}_{date}_ELR_v{d}.nc'):
                    all_days_exist=False
            if all_days_exist:
                print(f"All ELR predictions already made for {country} at {accumulation}, skipping")
                continue   

    
        if len(subcounties_loop)==0 and subcounty[country]:
            print("No subcounty-level models found for:",country)
            subcounty_loop=False
            skip_subcounty=True
        ## Get the country mask
        sf_region = shapefile.Reader(shpreader.natural_earth(resolution='110m',
                                          category='cultural',
                                          name='admin_0_countries'))
        features = sf_region.shapeRecords()
        geometry_all = [Polygon(sf_region.shape(i).points)  for i in range(len(features)) if features[i].record[3] in [country]]
        
        region_vectorised = regionmask.Regions(geometry_all, overlap=True)
        
        print(f'Calculating ELR output for {accumulation} in {country}')
        for d in day:
            d = int(d)
            assert isinstance(d,int)
            ds = get_model_output(date, model=model, day=d)
            mask_list = region_vectorised.mask_3D(ds.rename({'longitude':'lon','latitude':'lat'}))
            mask_list = np.ma.masked_invalid(mask_list)
            #emp_probs = np.stack([np.mean(np.squeeze(np.searchsorted([t],ds.precipitation)),axis=0)\
            #                                      for t in [20,30,40,50]])[None,None,...]
            #emp_probs[:,:,:,~np.squeeze(mask_list)] = np.nan
            if county_loop:
                full_predictions_county = np.full([1,1,4,384,352],np.nan)
                if not os.path.exists(OUT_PATH+f'{accumulation}/{country}/county/'):
                    os.makedirs(OUT_PATH+f'{accumulation}/{country}/county/')
                
                for Location in counties_loop:
                    #print("Getting ELR predictions for", Location)
                    try:
                        geometry_all = get_geometry(Location, region_type='county', country=country)
                    except:
                        continue
                    ds_sel = get_region(Location, geometry_all, ds)
                    checkpoint = get_model_checkpoint(Location, country, d, model)
                    if model=='GAN':
                        warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
                        logreg_model = joblib.load(MODEL_PATH+f'{country}/counties/Region_bin_{Location}_logreg_models.pkl')['cGAN']
                    else:
                        logreg_model = joblib.load(MODEL_PATH+f'{country}/counties/Region_bin_{Location}_logreg_models.pkl')[model]
    
                    preds, mask_full, mask_reg = get_ELR_predictions(logreg_model, model, ds_sel, d, ds.longitude.values, ds.latitude.values, 
                                                                Location, date, OUT_PATH+f'{accumulation}/{country}/county/')
                    full_predictions_county[:,:,:,mask_full] = preds[:,:,:,mask_reg]
            
            if subcounty_loop:
                full_predictions_subcounty = np.full([1,1,4,384,352],np.nan)
                
                if not os.path.exists(OUT_PATH+f'{accumulation}/{country}/subcounty/'):
                    os.makedirs(OUT_PATH+f'{accumulation}/{country}/subcounty/')
                with tqdm(total=len(subcounties_loop)) as pbar:
                    for Location in subcounties_loop:
                        #print("Getting ELR predictions for", Location)
                        try:
                            geometry_all = get_geometry(Location, region_type='subcounty', country=country)
                        except:
                            print(Location)
                            continue
                        ds_sel = get_region(Location, geometry_all, ds)
                        checkpoint = get_model_checkpoint(Location, country, d, model)
                        if model=='GAN':
                            warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
                            logreg_model = joblib.load(\
                                MODEL_PATH+f'{country}/subcounties/Region_bin_{Location}_logreg_models.pkl')['cGAN']
                        else:
                            logreg_model = joblib.load(\
                                MODEL_PATH+f'{country}/subcounties/Region_bin_{Location}_logreg_models.pkl')[model]
        
                        preds, mask_full, mask_reg = get_ELR_predictions(logreg_model, model, ds_sel, d, ds.longitude.values, ds.latitude.values, 
                                                                    Location, date,
                                                                              OUT_PATH+f'{accumulation}/{country}/subcounty/')
                        full_predictions_subcounty[:,:,:,mask_full] = preds[:,:,:,mask_reg]
                        pbar.update(1)

            # For some regions county level models converged but not subcounty
            # due to limited grid cells, so we merge the two
            if county_loop and subcounty_loop:
                nan_mask = np.isnan(full_predictions_subcounty)
                full_predictions_subcounty[nan_mask] = full_predictions_county[nan_mask]
                county_loop = False
            if store_netcdf:
                if subcounty_loop:
                    if not os.path.exists(OUT_PATH+f'{accumulation}/{country}/subcounty/'):
                        os.makedirs(OUT_PATH+f'{accumulation}/{country}/subcounty/')
                    file_name = OUT_PATH+f'{accumulation}/{country}/subcounty/{model}_{date}_ELR_v{d}.nc'
                    if os.path.exists(file_name):
                        continue
                    else:
                        time_delta = (d*24)+6
                        #nan_mask = np.isnan(full_predictions_subcounty)
                        #full_predictions_subcounty[nan_mask] = emp_probs[nan_mask]
                        ds_subcounty = xr.DataArray(full_predictions_subcounty, 
                                                    dims = ['time','fcst_valid_time','threshold','latitude','longitude'],
                                  coords = {\
                                      'time': ds.time.values,
                                      'fcst_valid_time': ds.time.values+time_delta,
                                      'threshold': [20,30,40,50],
                                      'latitude': np.unique(ds.latitude.values),
                                      'longitude': np.unique(ds.longitude.values),
                                  }
                                 ).rename('probability_exceedance')
                        #ds_subcounty.fcst_valid_time.attrs["units"]="hours since 1900-01-01 00:00:00.0"
                        #ds_subcounty.time.attrs["units"]="hours since 1900-01-01 00:00:00.0"
                        
                        ## If we want to crop to country un-comment
                        #left = bounding_box[country][0]-0.1
                        #right = bounding_box[country][1]+0.1
                        #bottom = bounding_box[country][2]-0.1
                        #top = bounding_box[country][3]+0.1
                        #ds_subcounty = ds_subcounty.sel({'longitude':slice(left,right),'latitude':slice(bottom,top)})
                        ds_subcounty.to_netcdf(file_name)
    
                if county_loop:
                    if not os.path.exists(OUT_PATH+f'{accumulation}/{country}/county/'):
                        os.makedirs(OUT_PATH+f'{accumulation}/{country}/county/')
                    file_name = OUT_PATH+f'{accumulation}/{country}/county/{model}_{date}_ELR_v{d}.nc'
                    if os.path.exists(file_name):
                        continue
                    else:
                        time_delta = (d*24)+6
                        #nan_mask = np.isnan(full_predictions_county)
                        #full_predictions_county[nan_mask] = emp_probs[nan_mask]
                        ds_county = xr.DataArray(full_predictions_county, 
                                                    dims = ['time','fcst_valid_time','threshold','latitude','longitude'],
                                  coords = {\
                                      'time': ds.time.values,
                                      'fcst_valid_time': ds.time.values+time_delta,
                                      'threshold': [20,30,40,50],
                                      'latitude': np.unique(ds.latitude.values),
                                      'longitude': np.unique(ds.longitude.values),
                                  }
                                 ).rename('probability_exceedance')
                        #ds_county.fcst_valid_time.attrs["units"]="hours since 1900-01-01 00:00:00.0"
                        #ds_county.time.attrs["units"]="hours since 1900-01-01 00:00:00.0"
                        ## If we want to crop to country un-comment
                        #left = bounding_box[country][0]-0.1
                        #right = bounding_box[country][1]+0.1
                        #bottom = bounding_box[country][2]-0.1
                        #top = bounding_box[country][3]+0.1
                        #ds_county = ds_county.sel({'longitude':slice(left,right),'latitude':slice(bottom,top)})
                        ds_county.to_netcdf(file_name)
                
            
                        
                    
