from file_paths import paths


import shapefile
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union

from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cartopy.crs as ccrs

def get_geometry_idx(region_type,country):
    if country == 'Kenya':
        if region_type == 'subcounty':
            return 6
        elif region_type == 'county':
            return 0
    elif country == 'Ethiopia':
        if region_type == 'subcounty':
            return 2
        elif region_type == 'county':
            return 7
    elif country == 'Rwanda':
        if region_type == 'county':
            return 2

def get_geometry(Location, region_type='county',country='Kenya', return_all=False):

    if country=='Kenya' and region_type=='county':
        geometry_path = paths[f'{country}_shapes_county']
    else:
        geometry_path = paths[f'{country}_shapes']

    if return_all:
        shape_feature = ShapelyFeature(Reader(geometry_path).geometries(), ccrs.Robinson(), edgecolor='black')

        return shape_feature

    sf_region = shapefile.Reader(geometry_path)
    features = sf_region.shapeRecords()

    idx = get_geometry_idx(region_type, country)
    geometry_all = [Polygon(sf_region.shape(i).points) for i in range(len(features)) if\
                                          Location in features[i].record[idx].replace('/','-')]

    assert len(geometry_all)!=0

    if len(geometry_all)==1:
        return geometry_all
    else:
        return [unary_union(geometry_all)]
