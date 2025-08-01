from file_paths import paths


import shapefile
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union

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

def get_geometry(Location, region_type='county',country='Kenya'):

    if country == 'Kenya' and region_type == 'subcounty':
        geometry_path = paths['Kenya_subcounties'][Location]
    else:
        geometry_path = paths[f'{country}_shapes']

    sf_region = shapefile.Reader(geometry_path)
    features = sf_region.shapeRecords()

    idx = get_geometry_idx(region_type, country)

    if country=='Kenya': 
        if region_type=='county':
            Location = Location.upper()
        elif region_type=='subcounty':
            Location = Location.replace('-',' ')

    geometry_all = [Polygon(sf_region.shape(i).points) for i in range(len(features)) if\
                                          Location in features[i].record[idx]]

    assert len(geometry_all)!=0

    if len(geometry_all)==1:
        return geometry_all
    else:
        return [unary_union(geometry_all)]
