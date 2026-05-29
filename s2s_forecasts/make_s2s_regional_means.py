import geopandas as gpd
import xarray as xr
import regionmask
import numpy as np
import os

def load_shapefile(shapefile_path, subset=None):
    """
    Load a shapefile or GeoPackage and optionally filter it to a subset of
    region names.

    Parameters
    ----------
    shapefile_path : str or pathlib.Path
        Path to the shapefile/GeoPackage.
    subset : iterable of str, optional
        Region names to keep, matched against the ``shapeName`` column.

    Returns
    -------
    geopandas.GeoDataFrame
        Loaded geometries reprojected to EPSG:4326.

    Raises
    ------
    IOError
        If the file cannot be read.
    ValueError
        If ``subset`` is provided but no matching polygons are found.
    """
    #Load shapefile
    print("Reading shapefile from: ", shapefile_path)
    try:
        gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    except Exception as e:
        raise IOError(f"Error reading shapefile {shapefile_path}: {e}")
    
    #Crop to subset regions if provided
    if subset is not None:
        gdf = gdf[gdf["shapeName"].isin(subset)]
        
        if gdf.empty:
            raise ValueError(
                f"No polygons found for {subset} in {shapefile_path}. "
                "Check the exact spelling in the shapeName column."
            )
        
    return gdf


def mean_within_polygons(
    obj: xr.Dataset | xr.DataArray,
    gdf: gpd.GeoDataFrame,
    label_col: str = "shapeName",
    lat_name: str = "lat",
    lon_name: str = "lon",
    crop: bool = True,
):
    """
    Compute area-weighted means of gridded data within each polygon.

    The input object is masked by polygon boundaries and averaged over latitude
    and longitude using cosine(latitude) weights.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Gridded input data with latitude and longitude dimensions.
    gdf : geopandas.GeoDataFrame
        Polygon geometries defining regions of interest.
    label_col : str, optional
        Column in ``gdf`` used to label the output region dimension.
    lat_name : str, optional
        Name of the latitude dimension in ``obj``.
    lon_name : str, optional
        Name of the longitude dimension in ``obj``.
    crop : bool, optional
        If True, first subset the data to the polygon bounding box.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Input object averaged within each polygon, with a new ``region``
        dimension and a ``country`` coordinate.
    """
    #Tidy up gdf object
    gdf = gdf.to_crs("EPSG:4326").reset_index(drop=True)

    #Crop to bounding box of polygons to speed up masking if desired
    if crop:
        minx, miny, maxx, maxy = gdf.total_bounds

        obj = obj.sel(
            {
                lat_name: slice(miny, maxy),
                lon_name: slice(minx, maxx),
            }
        )
    #Calculate cosine latitude weights
    weights = xr.DataArray(
        np.cos(np.deg2rad(obj[lat_name])),
        coords={lat_name: obj[lat_name]},
        dims=(lat_name,),
        name="weights",
    )
    #Generate mask of which grid points fall within which polygon
    mask = regionmask.mask_geopandas(gdf, obj, wrap_lon=None)

    #Perform cropping
    out = []
    labels = gdf[label_col].astype(str).to_numpy()
    countries = gdf["country"].astype(str).to_numpy()

    for i, label in enumerate(labels):
        regional = (
            obj.where(mask == i)
            .weighted(weights)
            .mean(dim=(lat_name, lon_name))
            .expand_dims(region=[label])
        )

        regional['country'] = countries[i] #Add country as a coordinate for easier grouping later

        out.append(regional)

    return xr.concat(out, dim="region")


def make_regional_means(year, month, day, lead_times_weeks=[1,2,3], IN_FOLDER="./s2s_data", OUT_FOLDER="./s2s_data/regional_means", SHAPEFILE_FOLDER="./shapefiles", shapefile_name="admin1_merged_KeEtRwUg.gpkg", region_subset=None):
    """
    Generate and save regional-mean forecast NetCDF files for one or more lead
    times.

    For each requested lead time, this function loads the gridded forecast
    file, computes area-weighted polygon means using the provided shapefile,
    and writes the regional result to disk.

    Parameters
    ----------
    year, month, day : int
        Forecast initialization date.
    lead_times_weeks : list of int, optional
        Lead times to process.
    IN_FOLDER : str, optional
        Directory containing the input gridded forecast NetCDF files.
    OUT_FOLDER : str, optional
        Directory where regional-mean NetCDF files will be written.
    SHAPEFILE_FOLDER : str, optional
        Directory containing the shapefile or GeoPackage.
    shapefile_name : str, optional
        Name of the shapefile/GeoPackage used to define regions.
    region_subset : iterable of str, optional
        Optional subset of regions to keep from the shapefile.

    Returns
    -------
    None
    """
    #Load shapefile
    gdf = load_shapefile(os.path.join(SHAPEFILE_FOLDER, shapefile_name), subset=region_subset)

    #Iterate through lead times, opening file and calculating regional means for each
    for week in lead_times_weeks:
        fname = os.path.join(IN_FOLDER, f"{year}-{month:02d}-{day:02d}_tp_meanstd_{week}wklead.nc")
        try:
            print(f"Processing file: {fname}")
            ds = xr.open_dataset(fname)
        except Exception as e:
            print(f"Error opening file {fname}: {e}. Please check download was successful and file is not corrupted.")
            continue

        ds_regional = mean_within_polygons(
            ds,
            gdf,
        )

        #save regional means to netcdf
        os.makedirs(OUT_FOLDER, exist_ok=True)
        out_fname = f"{year}-{month:02d}-{day:02d}_tp_meanstd_{week}wklead_{shapefile_name.split('.')[0]}.nc"
        ds_regional.to_netcdf(os.path.join(OUT_FOLDER, out_fname))
        print(f"Saved regional means for lead time {week} weeks to: {out_fname}")
