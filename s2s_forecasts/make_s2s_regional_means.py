import xarray as xr
import numpy as np
import os
from pathlib import Path

def load_fraction_mask(mask_path: Path, subset=None):
    ds = xr.load_dataset(mask_path)
    
    if subset is not None:
        ds = ds.where(ds.region.isin(subset), drop=True)

    return ds

def regional_mean_from_fraction_mask(
        obj: xr.DataArray,
        region_mask: xr.DataArray,
        *,
        region_dim: str = "region",
        lat_name: str = "lat",
        lon_name: str = "lon",
        crop: bool = True,
        country_list: list = None,
    ):
        """
        Area-weighted regional mean using a fractional region mask.

        region_mask should have:
        - one dimension of regions, e.g. region
        - lat/lon dimensions
        - a coordinate called 'country' on the region dimension, if available
        """

        if region_dim not in region_mask.dims:
            raise ValueError(
                f"Expected region_mask to have a '{region_dim}' dimension, "
                f"but found dims={region_mask.dims}"
            )

        # Unique region labels
        if region_dim in region_mask.coords:
            region_labels = region_mask[region_dim].values
        else:
            region_labels = np.arange(region_mask.sizes[region_dim])


        #Make mask array
        region_mask = region_mask["region_mask"]

        if crop:
            spatial_valid = region_mask.fillna(0).sum(dim=region_dim) > 0
            obj = obj.where(spatial_valid, drop=True)
            region_mask = region_mask.where(spatial_valid, drop=True)

        area_weights = xr.DataArray(
            np.cos(np.deg2rad(obj[lat_name])),
            coords={lat_name: obj[lat_name]},
            dims=(lat_name,),
            name="area_weights",
        )

        out = []
        for i, reg_label in enumerate(region_labels):
            frac = region_mask.sel({region_dim: reg_label})

            # Mask the data outside the region, but do NOT replace with zero
            valid = frac.notnull() & (frac > 0)
            data_masked = obj.where(valid)

            # Use fraction mask only as weights
            weights2d = frac.where(valid, 0) * area_weights

            regional = (
                data_masked
                .weighted(weights2d)
                .mean(dim=(lat_name, lon_name))
                .expand_dims(region=[str(reg_label)])
            )

            if country_list is not None:
                regional = regional.assign_coords(
                    country=("region", [str(country_list[i])])
                )

            out.append(regional)

        return xr.concat(out, dim="region")


def make_regional_means(year, month, day, lead_times_weeks=[1,2,3], IN_FOLDER="./s2s_data", OUT_FOLDER="./s2s_data/regional_means", REGIONMASK_FOLDER="./shapefiles", regionmask_name="admin1_merged_KeEtRwUg.gpkg", region_subset=None):
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
    REGIONMASK_FOLDER : str, optional
        Directory containing the shapefile or GeoPackage.
    regionmask_name : str, optional
        Name of the shapefile/GeoPackage used to define regions.
    region_subset : iterable of str, optional
        Optional subset of regions to keep from the shapefile.

    Returns
    -------
    None
    """
    #Load shapefile
    mask = load_fraction_mask(os.path.join(REGIONMASK_FOLDER, regionmask_name), subset=region_subset)
    countries = mask.country.values
    #Iterate through lead times, opening file and calculating regional means for each
    for week in lead_times_weeks:
        fname = os.path.join(IN_FOLDER, str(year), f"{year}-{month:02d}-{day:02d}_tp_meanstd_{week}wklead.nc")
        try:
            print(f"Processing file: {fname}")
            ds = xr.open_dataset(fname)
        except Exception as e:
            print(f"Error opening file {fname}: {e}. Please check download was successful and file is not corrupted.")
            continue

        ds_regional = regional_mean_from_fraction_mask(
            ds,
            mask,
            country_list=countries,
        )

        #save regional means to netcdf
        os.makedirs(OUT_FOLDER, exist_ok=True)
        out_fname = f"{year}-{month:02d}-{day:02d}_tp_meanstd_{week}wklead_{regionmask_name.split('.')[0]}.nc"
        ds_regional.to_netcdf(os.path.join(OUT_FOLDER, str(year), out_fname))
        print(f"Saved regional means for lead time {week} weeks to: {out_fname}")
