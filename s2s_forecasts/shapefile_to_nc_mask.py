#Imports
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep
from pathlib import Path
import argparse

def vector_to_region_masks(
    vector_dir,
    vector_name,
    template_nc_path,
    out_dir,
    layer="auto",
    region_col=None,
    vector_crs=None,
    mask_crs="EPSG:4326",
    lat_name="lat",
    lon_name="lon",
    out_var="region_mask",
):
    """
    Create a 0/1 mask for each polygon feature in a vector file on the grid of a
    template NetCDF, then save the result as a NetCDF.

    Supports vector formats readable by geopandas, for example:
        .gpkg, .shp, .geojson, .json

    Output is saved as:
        out_dir / vector_name_with_nc_extension

    For example:
        vector_name = "regions.gpkg" -> "regions.nc"
        vector_name = "regions.shp"  -> "regions.nc"
    """
    vector_dir = Path(vector_dir)
    out_dir = Path(out_dir)
    vector_path = vector_dir / vector_name

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(vector_name).stem}.nc"

    ds = xr.open_dataset(template_nc_path)

    if lat_name not in ds.coords or lon_name not in ds.coords:
        raise ValueError(
            f"Could not find '{lat_name}' and '{lon_name}' in the NetCDF coordinates."
        )

    lat = ds[lat_name].values
    lon = ds[lon_name].values

    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("This version expects 1D lat/lon coordinates.")

    suffix = vector_path.suffix.lower()

    multilayer_suffixes = {
        ".gpkg",
        ".gdb",
        ".sqlite",
        ".db",
    }

    read_kwargs = {}

    if layer == "auto":
        if suffix in multilayer_suffixes:
            # Assumes the layer has the same name as the file stem.
            # Example: vector_name="regions.gpkg" -> layer="regions"
            read_kwargs["layer"] = vector_path.stem
    elif layer is not None:
        read_kwargs["layer"] = layer

    gdf = gpd.read_file(vector_path, **read_kwargs)

    if gdf.empty:
        raise ValueError(f"No features found in vector file: {vector_path}")

    if gdf.crs is None:
        if vector_crs is None:
            raise ValueError(
                "Input vector file has no CRS. Pass vector_crs, e.g. vector_crs='EPSG:4326'."
            )
        gdf = gdf.set_crs(vector_crs)

    if mask_crs is not None:
        gdf = gdf.to_crs(mask_crs)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    polygon_types = {"Polygon", "MultiPolygon"}
    gdf = gdf[gdf.geometry.geom_type.isin(polygon_types)].copy()

    if gdf.empty:
        raise ValueError("No valid Polygon or MultiPolygon geometries found.")

    if region_col is not None:
        if region_col not in gdf.columns:
            raise ValueError(f"Column '{region_col}' not found in vector file.")
        region_names = gdf[region_col].astype(str).tolist()
    else:
        region_names = [str(i) for i in gdf.index.tolist()]

    xx, yy = np.meshgrid(lon, lat)
    flat_points = [Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())]

    masks = []
    valid_region_names = []

    for name, geom in zip(region_names, gdf.geometry):
        pg = prep(geom)

        inside = np.array(
            [pg.contains(p) or pg.touches(p) for p in flat_points],
            dtype=bool,
        ).reshape(xx.shape)

        masks.append(inside.astype(np.uint8))
        valid_region_names.append(name)

    if not masks:
        raise ValueError("No valid geometries found in vector file.")

    mask_arr = np.stack(masks, axis=-1)

    coords = {
        lat_name: ds[lat_name],
        lon_name: ds[lon_name],
        "region": valid_region_names,
        }

    if "country" in gdf.columns:
        coords["country"] = (
            "region",
            gdf["country"].astype(str).to_numpy(),
        )

    da = xr.DataArray(
        mask_arr,
        dims=(lat_name, lon_name, "region"),
        coords=coords,
        name=out_var,
        attrs={
            "description": "0/1 masks for each region; 1 inside polygon, 0 outside",
            "source_vector": str(vector_path),
            "mask_crs": str(mask_crs),
        },
    )

    out_ds = da.to_dataset()
    out_ds.to_netcdf(out_path)

    return out_ds


#Main

def main():
    parser = argparse.ArgumentParser(
        description="Create region masks from a shapefile/vector file and save as NetCDF."
    )
    parser.add_argument("--vector-dir", required=True, help="Directory containing the vector file")
    parser.add_argument("--vector-name", required=True, help="Vector filename, e.g. regions.shp")
    parser.add_argument("--template-nc-path", required=True, help="Template NetCDF path")
    parser.add_argument("--out-dir", required=True, help="Output directory for the NetCDF")
    parser.add_argument("--layer", default="auto", help="Layer name for multi-layer files, or 'auto'")
    parser.add_argument("--region-col", default=None, help="Column to use for region names")
    parser.add_argument("--vector-crs", default=None, help="CRS to assign if missing, e.g. EPSG:4326")
    parser.add_argument("--mask-crs", default="EPSG:4326", help="CRS to reproject geometries to")
    parser.add_argument("--lat-name", default="lat", help="Latitude coordinate name")
    parser.add_argument("--lon-name", default="lon", help="Longitude coordinate name")
    parser.add_argument("--out-var", default="region_mask", help="Output variable name")

    args = parser.parse_args()

    vector_to_region_masks(
        vector_dir=args.vector_dir,
        vector_name=args.vector_name,
        template_nc_path=args.template_nc_path,
        out_dir=args.out_dir,
        layer=args.layer,
        region_col=args.region_col,
        vector_crs=args.vector_crs,
        mask_crs=args.mask_crs,
        lat_name=args.lat_name,
        lon_name=args.lon_name,
        out_var=args.out_var,
    )


if __name__ == "__main__":
    main()