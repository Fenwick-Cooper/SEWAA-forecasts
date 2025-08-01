import os
import glob
import argparse
import numpy as np
from datetime import datetime

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm

import cartopy.feature as cfeature
import cartopy.crs as ccrs

from file_paths import paths
from helper_functions import get_geometry
from shapely.ops import unary_union

def plot_exceedance(ds, Location, date, day, threshold, model, 
                    save_path, geometry, probability_bins=None):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #if os.path.exists(save_path+f'{model}_{Location}_{date}_{day}-day_leadtime_{threshold}_mmday.png'):
    #    print('Figure already created')
    #    return
    timedelta = np.timedelta64(day,'D')+np.timedelta64(6,'h')
    init_time = ds.time.values[0].astype('datetime64[h]').astype(object).strftime('%Y-%m-%d %H:00')
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Robinson())
    array_to_plot = ds.sel({'threshold':threshold,'fcst_valid_time':ds.time.values+timedelta})

    valid_time = array_to_plot.fcst_valid_time.values[0].astype('datetime64[h]').astype(object).strftime('%Y-%m-%d %H:00')

    ax.set_facecolor('white')  # For consistency with Harris et. al 2022
    if probability_bins is None:
        c = ax.pcolormesh(ds.longitude.values, ds.latitude.values, np.squeeze(array_to_plot.to_dataarray().values)*100,
                       transform=ccrs.Robinson(), cmap=plt.get_cmap('Reds',10),norm=LogNorm(vmin=1,vmax=100))
        cb = plt.colorbar(c, fraction=0.2,orientation='horizontal')
    else:
        array_to_plot = 100*np.squeeze(array_to_plot.to_dataarray().values)
        nan_mask = np.isnan(array_to_plot)
        binned_probabilities = np.searchsorted(probability_bins,array_to_plot,side='right').astype(np.float32)
        binned_probabilities[nan_mask] = np.nan
        
        # Unevenly-spaced bounds changes the colormapping.
        bounds = np.arange(-0.5, len(probability_bins)+1, 1)

        norm = BoundaryNorm(boundaries=bounds, ncolors=256)
        c = ax.pcolormesh(ds.longitude.values, ds.latitude.values, binned_probabilities,
                       transform=ccrs.Robinson(), cmap=plt.get_cmap('Reds'),norm=norm)
        cb = plt.colorbar(c, fraction=0.2, orientation='horizontal')
        cb.ax.set_xticks((bounds[1:]+bounds[:-1])/2)
        if len(probability_bins)==1:
            cb.ax.set_xticklabels([f'<{probability_bins[0]}',f'>={probability_bins[-1]}'])
        else:
            cb.ax.set_xticklabels([f'<{probability_bins[0]}']\
                                  +[f'{above}-{below}' for above,below in zip(probability_bins[:-1],probability_bins[1:])]\
                                  +[f'>={probability_bins[-1]}'])

    cb.set_label(f'Probability of Exceedance [%]')
    region_extent=ax.get_extent()
    ax.set_extent([region_extent[0]-.07,region_extent[1]+.07,
               region_extent[2]-.07,region_extent[3]+.07], crs=ccrs.Robinson())
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.LAKES, linewidth=1,linestyle='-',edgecolor='grey',facecolor='none')
    ax.add_geometries(geometry, crs=ccrs.Robinson(), facecolor='none', edgecolor='k', linewidth=3)
    

    plt.title(f'{model} forecast for {threshold} mm/day at {Location}:\n fcst time: {init_time}, valid time: {valid_time}') 

    plt.savefig(save_path+f'{model}_{Location}_{date}_{day}-day_leadtime_{threshold}_mmday.png')
    plt.show()

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='IFS or GAN',default=None,type=str)
    parser.add_argument('--model', help='IFS or GAN',default='GAN',type=str)
    parser.add_argument('--day', help='valid date',default=1,type=int)
    parser.add_argument('--country', help='Country within which to plot',default='Kenya',type=str)
    parser.add_argument('--threshold', help='threshold within which to plot',default=40,type=int)
    parser.add_argument('--loc', help='Location within country to plot',action='append',nargs='+',type=str,default=None)
    parser.add_argument('--probability_bins', help='probability bins to use',
                        action='append',nargs='+',type=int,default=None)
    
    args = parser.parse_args()

    date = args.date
    if date is None:
        date = np.array(['2024-04-23'],dtype='datetime64[D]')[0].astype(object).strftime("%Y%m%d")#datetime.now().strftime("%Y%M%d")
    
    probability_bins = args.probability_bins
    if probability_bins is not None:
        probability_bins = probability_bins[0]
       
    model = args.model
    day = args.day
    country = args.country
    Locations = args.loc
    if Locations is not None:
         Locations = Locations[0]

    threshold = args.threshold

    counties = glob.glob(paths['MODEL_PATH']+f'{country}/counties/*')
    counties = [c.split('/')[-1].split('_')[0] for c in counties]
    subcounties = glob.glob(paths['MODEL_PATH']+f'{country}/subcounties/*')
    subcounties = [c.split('/')[-1].split('_')[0] for c in subcounties]
   
    ds = []
    geometry_all = []
    for Location in Locations:
        if Location in counties:
            region_type = 'county'
        elif Location in subcounties:
            region_type = 'subcounty'
        else:
            print('Could not find corresponding model for Location,', Location, 'in country folder', country)
        in_path = paths['OUT_PATH']+f'{country}/{region_type}/'
        ds.append(xr.open_zarr(in_path+f'{model}_{Location}_{date}_logreg.zarr'))
        geometry_all.append(get_geometry(Location, region_type, country))
    if len(Locations)>1:
        save_path = paths['OUT_PATH']+f'plots/{country}/combined_regions/'
        ds = xr.merge(ds)
        geometry_all = unary_union(geometry_all)
    else:
        save_path = paths['OUT_PATH']+f'plots/{country}/{region_type}/'
        ds = ds[0]
        geometry_all = geometry_all[0]
    
    plot_exceedance(ds, Location, date, day, threshold, model, save_path, geometry_all, probability_bins=probability_bins)


