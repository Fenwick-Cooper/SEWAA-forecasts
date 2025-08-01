import os
import glob
import argparse
from datetime import datetime

import numpy as np
import xarray as xr
import joblib

from file_paths import paths
from helper_functions import get_geometry
import regionmask


OUT_PATH = paths['OUT_PATH']
FCST_PATH = paths['FCST_PATH']
MODEL_PATH = paths['MODEL_PATH']

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

countries = ['Ethiopia','Kenya']
county = True
subcounty = True

counties = None
subcounties = None

def get_cGAN_output(date, accumulation = "6h_accumulations", model="GAN", day=1):

    ds_fcst = xr.open_dataset(f'{FCST_PATH}/{accumulation}/{model}_{date}_00Z.nc')

    if accumulation == "6h_accumulations":

        ds_fcst = ds_fcst.mean("valid_time")
    else:
        ds_fcst = ds_fcst.isel({"valid_time":day})

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
    

def get_ELR_predictions(logreg_model, model, ds_sel, day, Location, date, save_path):

    file_name = save_path+f'{model}_{Location}_{date}_logreg.zarr'
    timedelta = np.timedelta64(day,'D')+np.timedelta64(6,'h')
    
    if os.path.exists(file_name):
        precheck = xr.open_zarr(file_name)
        if ds_sel.time.values+timedelta in precheck.fcst_valid_time.values:
            print('predictions already made, skipping')
            return

    thresholds = np.asarray([key for key in logreg_model.keys()])
    latitude_reg = ds_sel.latitude.values
    longitude_reg = ds_sel.longitude.values
    date = ds_sel.time.values[0].astype('datetime64[D]').astype(object).strftime("%Y%M%d")
    
    lons, lats = np.meshgrid(np.unique(longitude_reg),np.unique(latitude_reg))
    predictions = np.full([1,1,thresholds.shape[0],lats.shape[0],lats.shape[1]],np.nan)
    
    mask = np.full([lats.shape[0],lats.shape[1]],False)
    
    for lat_reg,lon_reg in zip(latitude_reg,longitude_reg):
    
        idx_2d = np.ma.asarray(lats==lat_reg) * np.ma.asarray(lons==lon_reg)
        mask[idx_2d] = True
    
    X = np.sort(np.squeeze(24*ds_sel.values),axis=0).T

    if model == 'GAN':

        X = np.percentile(X, np.linspace(1,100,50), axis=1, method='weibull').T
    
    for i, threshold in enumerate(thresholds):
        predictions[0,0,i,mask] = logreg_model[threshold].predict_proba(X)[:,1]

    ds_predictions = xr.DataArray(predictions, dims = ['time','fcst_valid_time','threshold','latitude','longitude'],
                                  coords = {\
                                      'time': ds_sel.time.values,
                                      'fcst_valid_time': ds_sel.time.values+timedelta,
                                      'threshold': thresholds,
                                      'latitude': np.unique(latitude_reg),
                                      'longitude': np.unique(longitude_reg),
                                  }
                                 ).rename('probability_exceedance')

    if os.path.exists(file_name):
        ds_predictions.to_zarr(file_name, mode='a-',append_dim='fcst_valid_time')
    else:
        ds_predictions.to_zarr(file_name, mode='w')

    

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='IFS or GAN',default=None)
    parser.add_argument('--model', help='IFS or GAN',default='GAN',type=str)
    parser.add_argument('--day', help='valid date',default=1,type=int)
    parser.add_argument('--accumulation', help='6h- or 24h- accumulation',default="24h_accumulations",type=str)
    args = parser.parse_args()

    
    date = args.date
    if date is None:
        date = np.array(['2024-04-23'],dtype='datetime64[D]')[0].astype(object).strftime("%Y%m%d")#datetime.now().strftime("%Y%M%d")
    model = args.model
    day = args.day
    accumulation = args.accumulation

    ds = get_cGAN_output(date, accumulation = accumulation, model=model, day=day)

    for country in countries:

        county_loop = county
        subcounty_loop = subcounty

        

        if not os.path.exists(OUT_PATH+f'{country}/'):
            os.makedirs(OUT_PATH+f'{country}/')

        if counties == None and county:
            counties_loop = glob.glob(MODEL_PATH+f'{country}/counties/*')
            counties_loop = [c.split('/')[-1].split('_')[0] for c in counties_loop]
        
        if len(counties_loop)==0 and county:
            print("No county-level models found for:",country)
            county_loop=False
    
        if subcounties == None and subcounty:
            subcounties_loop = glob.glob(MODEL_PATH+f'{country}/subcounties/*')
            subcounties_loop = [c.split('/')[-1].split('_')[0] for c in subcounties_loop]
    
        if len(subcounties_loop)==0 and subcounty:
            print("No subcounty-level models found for:",country)
            subcounty_loop=False

        if subcounty_loop:

            if not os.path.exists(OUT_PATH+f'{country}/subcounty/'):
                os.makedirs(OUT_PATH+f'{country}/subcounty/')
            
            for Location in subcounties_loop:

                print("Getting ELR predictions for", Location)

                geometry_all = get_geometry(Location, region_type='subcounty', country=country)
                ds_sel = get_region(Location, geometry_all, ds)
                checkpoint = get_model_checkpoint(Location, country, day, model)
                if model=='GAN':
                    logreg_model = joblib.load(MODEL_PATH+f'{country}/subcounties/{Location}_logreg_models.pkl')[checkpoint]['cGAN']
                else:
                    logreg_model = joblib.load(MODEL_PATH+f'{country}/subcounties/{Location}_logreg_models.pkl')[checkpoint][model]


                get_ELR_predictions(logreg_model, model, ds_sel, day, Location, date, OUT_PATH+f'{country}/subcounty/')

    
        if county_loop:

            if not os.path.exists(OUT_PATH+f'{country}/county/'):
                os.makedirs(OUT_PATH+f'{country}/county/')
            
            for Location in counties_loop:

                print("Getting ELR predictions for", Location)

                geometry_all = get_geometry(Location, region_type='county', country=country)
                ds_sel = get_region(Location, geometry_all, ds)
                checkpoint = get_model_checkpoint(Location, country, day, model)
                if model=='GAN':
                    logreg_model = joblib.load(MODEL_PATH+f'{country}/counties/{Location}_logreg_models.pkl')[checkpoint]['cGAN']
                else:
                    logreg_model = joblib.load(MODEL_PATH+f'{country}/counties/{Location}_logreg_models.pkl')[checkpoint][model]

                get_ELR_predictions(logreg_model, model, ds_sel, day, Location, date, OUT_PATH+f'{country}/county/')

                
            