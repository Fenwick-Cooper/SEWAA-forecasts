import os
import glob
import argparse
import numpy as np
from datetime import datetime
import regionmask

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm

import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs

from file_paths import paths
from helper_functions import get_geometry
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union
import shapefile

country_region_type_available = {'Kenya':'subcounty','Ethiopia':'subcounty','Rwanda':'county'}
bounding_box = {'Ethiopia':(32.95418, 47.78942, 3.45, 14.837),'Kenya':(33.935689697, 41.5550830926, -4.559, 5.4877),
                'Rwanda':(28.87, 30.90, -2.81, -1.151)}

def plot_exceedance(ds, country, date, day, threshold, model, 
                    save_path, probability_bins=None,clim=False):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    timedelta = np.timedelta64(day,'D')+np.timedelta64(6,'h')
    my_dpi=3
    block_size=7
    if country=='Rwanda':
        block_size=30
    region_extent = bounding_box[country]

    if clim:
        month = (np.array([f'{date[:4]}-{date[4:6]}-{date[-2:]}'],
                           dtype='datetime64[ns]')+timedelta)[0].astype('datetime64[h]').astype(object).month
        ds_sel = xr.open_dataset(f'./climatological_exceedances/clim_exc_{threshold}mmday_{month}month.nc')
        ds_sel = ds_sel.sel({'longitude':slice(region_extent[0],region_extent[1]),
                         'latitude':slice(region_extent[2],region_extent[3])})
        save_file_name = save_path+f'{model}_{date}_{day}-day_leadtime_{threshold}_mmday_clim.png'
    else:
        ## Select sub-region and timestep to plot
        init_time = ds.time.values[0].astype('datetime64[h]').astype(object).strftime('%Y-%m-%d %H:00')
        ds_sel = ds.sel({'longitude':slice(region_extent[0],region_extent[1]),
                         'latitude':slice(region_extent[2],region_extent[3])}).sel({'threshold':threshold,
                                                                                    'fcst_valid_time':ds.time.values+timedelta}) 
        save_file_name = save_path+f'{model}_{date}_{day}-day_leadtime_{threshold}_mmday.png'
        
    ## need to select the country file and set other things to nan
    array_to_plot = np.squeeze(ds_sel.to_dataarray().values)*100
    sf_region = shapefile.Reader(shpreader.natural_earth(resolution='110m',
                                              category='cultural',
                                              name='admin_0_countries'))
    features = sf_region.shapeRecords()
    geometry_all = [Polygon(sf_region.shape(i).points)  for i in range(len(features)) if features[i].record[3] in [country]]
    
    region_vectorised = regionmask.Regions(geometry_all, overlap=True)
    mask_list = region_vectorised.mask_3D(ds_sel.rename({'longitude':'lon','latitude':'lat'}))
    mask_list = np.ma.masked_invalid(mask_list)
    array_to_plot[~np.squeeze(mask_list)]=np.nan

    ## Get array specifications to make it a whole number of pixels
    w = array_to_plot.shape[0]
    h = array_to_plot.shape[1]-1
    fig = plt.figure(figsize=(w*block_size/my_dpi,h*block_size/my_dpi),frameon=False,dpi=my_dpi)
    fig.set_size_inches(h*block_size/my_dpi,w*block_size/my_dpi)
    ax = plt.axes(projection=ccrs.Robinson(),frameon=False)
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.set_facecolor('none')  # For consistency with Harris et. al 2022
    
    if probability_bins is None:
        ax.pcolormesh(ds_sel.longitude.values, ds_sel.latitude.values, array_to_plot,
                       transform=ccrs.Robinson(), cmap=plt.get_cmap('Reds',10),norm=LogNorm(vmin=1,vmax=100))
    else:
        nan_mask = np.isnan(array_to_plot)
        binned_probabilities = np.searchsorted(probability_bins,array_to_plot,side='right').astype(np.float32)
        binned_probabilities[nan_mask] = np.nan
        # Unevenly-spaced bounds changes the colormapping.
        bounds = np.arange(-0.5, len(probability_bins)+1, 1)
        norm = BoundaryNorm(boundaries=bounds, ncolors=256)
        ax.pcolormesh(ds_sel.longitude.values, ds_sel.latitude.values, binned_probabilities,
                       transform=ccrs.Robinson(), cmap=plt.get_cmap('Reds'),norm=norm)
    
    ax.set_extent([region_extent[0],region_extent[1],
               region_extent[2],region_extent[3]], crs=ccrs.Robinson())
    #ax.add_feature(cfeature.BORDERS, linewidth=.1)
    #ax.add_feature(cfeature.COASTLINE, linewidth=1)
    #ax.add_feature(cfeature.LAKES, linewidth=1,linestyle='-',edgecolor='grey',facecolor='none')
    #ax.add_feature(geometry, facecolor='none', edgecolor='k', linewidth=.075)
    #cb = plt.colorbar(c, fraction=0.075, orientation='horizontal',aspect=40)
    #cb.set_label(f'Probability of Exceedance [%]')
    plt.axis('off')
    fig.tight_layout()
    plt.savefig(save_file_name,dpi=my_dpi)
    plt.show()

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='fcst initialisation time',default=None,type=str)
    parser.add_argument('--model', help='IFS or GAN',default='GAN',type=str)
    parser.add_argument('--day', help='lead time (in days)',default=1,type=int)
    parser.add_argument('--country', help='Country within which to plot',default='Ethiopia',type=str)
    parser.add_argument('--threshold', help='threshold within which to plot',default=40,type=int)
    parser.add_argument('--probability_bins', help='probability bins to use',
                        action='append',nargs='+',type=float,default=None)
    parser.add_argument('--clim', help='whether to do climatology as well',
                        action='store_true',default=False)
    
    args = parser.parse_args()

    date = args.date
    if date is None:
        date = np.array(['2025-04-10'],dtype='datetime64[D]')[0].astype(object).strftime("%Y%m%d")#datetime.now().strftime("%Y%M%d")
    
    probability_bins = args.probability_bins
    if probability_bins is not None:
        probability_bins = probability_bins[0]
       
    model = args.model
    day = args.day
    accumulation = 24
        
    country = args.country
    threshold = args.threshold
    clim = args.clim
    region_type = country_region_type_available[country]
    
    in_path = paths['OUT_PATH']+f'{accumulation}h_accumulations/{country}/{region_type}/'
    save_path = paths['OUT_PATH']+f'plots/{country}/{region_type}/'
    if os.path.exists(in_path+f'{model}_{date}_ELR_v{day}.nc'):
        ds = xr.open_dataset(in_path+f'{model}_{date}_ELR_v{day}.nc')
        plot_exceedance(ds, country, date, day, threshold, model, save_path, probability_bins=probability_bins, clim=clim)
    elif clim:
        plot_exceedance(None, country, date, day, threshold, model, save_path, probability_bins=probability_bins, clim=clim)
    else:
        print(f"No predictions could be found for {country} at date {date} and lead time {day} day(s)")


