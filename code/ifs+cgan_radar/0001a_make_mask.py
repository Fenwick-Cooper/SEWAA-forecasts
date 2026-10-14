from pathlib import Path
import numpy as np
import xarray as xr
import os
import argparse

mean_low_threshold = 1e-5
mean_high_threshold = 0.08
std_threshold = 2e-5

def find_any_nan_mask(
    folder,
    variable,
    x_dim="x",
    y_dim="y",
    save_mask=True
):
    """
    Find grid cells that are NaN in at least one NetCDF file.

    Searches for .nc files inside subdirectories named YYYY.

    Parameters
    ----------
    folder : str or Path
        Root directory containing YYYY subdirectories.

    variable : str
        Name of the variable to check.

    x_dim, y_dim : str
        Names of the spatial dimensions.

    save_mask : bool
        If True, save the mask to folder/mask/mask.nc.

    Returns
    -------
    xarray.DataArray
        Boolean mask if save_mask=False.
    """

    folder = Path(folder)

    # Find .nc files inside YYYY directories
    files = []

    for year_dir in folder.iterdir():
        if (
            year_dir.is_dir()
            and year_dir.name.isdigit()
            and len(year_dir.name) == 4
        ):
            files.extend(year_dir.glob("*.nc"))

    files = sorted(files)

    if not files:
        raise FileNotFoundError(
            f"No .nc files found inside YYYY folders in {folder}"
        )

    print(f"Found {len(files)} files.")

    # ---------------------------------------------------------
    # First file establishes the grid
    # ---------------------------------------------------------

    with xr.open_dataset(files[0]) as ds:

        if variable not in ds:
            raise KeyError(
                f"'{variable}' not found in {files[0]}"
            )

        da = ds[variable]
        nan_mask = da.isnull()
        n_count = 1

        other_dims = [
            d for d in da.dims
            if d not in (y_dim, x_dim)
        ]

        if other_dims:
            # If each file contains multiple rainfall samples,
            # these all need to contribute to the statistics.
            file_sum = da.sum(dim=other_dims, skipna=True)
            file_sum2 = (da ** 2).sum(dim=other_dims, skipna=True)
            nan_mask = nan_mask.any(dim=other_dims)
        else:
            file_sum = da.fillna(0)
            file_sum2 = (da ** 2).fillna(0)

        sum_x = file_sum
        sum_x2 = file_sum2

        nan_mask = nan_mask.load()

    # ---------------------------------------------------------
    # Loop through remaining files
    # ---------------------------------------------------------

    for i, file in enumerate(files[1:], start=2):

        print(f"[{i}/{len(files)}] {file}")

        with xr.open_dataset(file) as ds:

            if variable not in ds:
                raise KeyError(
                    f"'{variable}' not found in {file}"
                )

            da = ds[variable]
            file_mask = da.isnull()
            n_count += 1

            other_dims = [
                d for d in da.dims
                if d not in (y_dim, x_dim)
            ]

            if other_dims:
                # If each file contains multiple rainfall samples,
                # these all need to contribute to the statistics.
                file_sum = da.sum(dim=other_dims, skipna=True)
                file_sum2 = (da ** 2).sum(dim=other_dims, skipna=True)
                file_mask = file_mask.any(dim=other_dims)
            else:
                file_sum = da.fillna(0)
                file_sum2 = (da ** 2).fillna(0)

            sum_x += file_sum
            sum_x2 += file_sum2

        file_mask = file_mask.load()

        nan_mask = nan_mask | file_mask

    
    #Calculate mean and standard deviation of the variable across all files
    mean = sum_x / n_count
    print("mean min/mean/max", np.min(mean).values, np.mean(mean).values, np.max(mean).values)
    variance = sum_x2 / n_count - mean**2

    # Numerical roundoff can produce tiny negative values
    variance = xr.where(
        variance < 0,
        0,
        variance
    )

    std = np.sqrt(variance)
    print("std min/mean/max", np.min(std).values, np.mean(std).values, np.max(std).values)

    print("grid percentiles", "50:", np.quantile(mean,0.5), "75:", np.quantile(mean,0.75), "50:", np.quantile(mean,0.5), "90:", np.quantile(mean,0.9))

    #Mask where mean < threshold
    mean_mask_low = mean < mean_low_threshold

    mean_mask_high = mean > mean_high_threshold

    #Mask where std < threshold
    std_mask = std < std_threshold

    #combine masks
    nan_mask = (nan_mask | mean_mask_low | mean_mask_high | std_mask)
    
    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    n_bad = int(nan_mask.sum())
    n_total = nan_mask.size

    print("\nDone.")
    print(f"Grid cells:       {n_total:,}")
    print(f"Total masked:      {n_bad:,}")
    print(f"Fraction masked:  {n_bad / n_total:.2%}")

    if save_mask:
        out_dir = folder / "mask"
        out_dir.mkdir(exist_ok=True)

        out_path = out_dir / "mask.nc"

        nan_mask.to_netcdf(out_path)

        print(f"Mask saved to:    {out_path}")
    else:
        return nan_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find grid cells that are NaN in at least one NetCDF file."
    )

    parser.add_argument(
        "folder",
        type=Path,
        help="Root directory containing YYYY subdirectories."
    )

    parser.add_argument(
        "variable",
        type=str,
        help="Name of the variable to check."
    )

    args = parser.parse_args()

    find_any_nan_mask(
        folder=args.folder,
        variable=args.variable,
    )