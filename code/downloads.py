import os
import subprocess

import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator

TARGET_GRID_FILE = "./data/example_grids/eg_ifs_grid.nc"

def regrid_to_target(
    file_path,
    target_grid_file=TARGET_GRID_FILE,
):
    """
    Regrid a downloaded regular lat/lon IFS forecast onto the target
    projected y/x grid.

    The original file is replaced only after the regridded file has
    been written and validated successfully.
    """

    temp_file = file_path + ".regridded"

    try:
        with xr.open_dataset(file_path) as ds, \
             xr.open_dataset(target_grid_file) as target:

            # -----------------------------------------------------
            # Target y/x grid
            # -----------------------------------------------------

            target_x = target["x"].values
            target_y = target["y"].values
            target_lon = target["longitude"].values
            target_lat = target["latitude"].values

            if target_lon.shape != (len(target_y), len(target_x)):
                raise ValueError(
                    "Target longitude shape does not match target y/x grid."
                )

            if target_lat.shape != target_lon.shape:
                raise ValueError(
                    "Target latitude/longitude shapes do not match."
                )

            # -----------------------------------------------------
            # Is it already on the target y/x grid?
            # -----------------------------------------------------

            if (
                "x" in ds.dims
                and "y" in ds.dims
                and ds.sizes["x"] == len(target_x)
                and ds.sizes["y"] == len(target_y)
                and "longitude" in ds
                and "latitude" in ds
                and ds["longitude"].dims == ("y", "x")
                and ds["latitude"].dims == ("y", "x")
                and np.allclose(
                    ds["longitude"].values,
                    target_lon,
                    atol=1e-6,
                    rtol=0,
                )
                and np.allclose(
                    ds["latitude"].values,
                    target_lat,
                    atol=1e-6,
                    rtol=0,
                )
            ):
                print(f"{file_path} is already on the target grid.")
                return True

            # -----------------------------------------------------
            # Check source grid
            # -----------------------------------------------------

            if "latitude" not in ds.coords or "longitude" not in ds.coords:
                raise ValueError(
                    "Downloaded forecast does not have latitude/longitude "
                    "coordinates."
                )

            source_lat = ds["latitude"].values
            source_lon = ds["longitude"].values

            if source_lat.ndim != 1 or source_lon.ndim != 1:
                raise ValueError(
                    "Downloaded forecast is not on the expected regular "
                    "1-D latitude/longitude grid."
                )

            # RegularGridInterpolator requires monotonically
            # increasing coordinates.
            if source_lat[0] > source_lat[-1]:
                ds = ds.sortby("latitude")
                source_lat = ds["latitude"].values

            if source_lon[0] > source_lon[-1]:
                ds = ds.sortby("longitude")
                source_lon = ds["longitude"].values

            # -----------------------------------------------------
            # Make sure target lies inside source domain
            # -----------------------------------------------------

            if (
                target_lat.min() < source_lat.min()
                or target_lat.max() > source_lat.max()
                or target_lon.min() < source_lon.min()
                or target_lon.max() > source_lon.max()
            ):
                raise ValueError(
                    "Target grid extends outside downloaded IFS domain."
                )

            print(
                f"Regridding {file_path}: "
                f"{len(source_lat)}x{len(source_lon)} lat/lon "
                f"-> {len(target_y)}x{len(target_x)} y/x"
            )

            # Target points for scipy: (latitude, longitude)
            target_points = np.column_stack(
                (
                    target_lat.ravel(),
                    target_lon.ravel(),
                )
            )

            # -----------------------------------------------------
            # Build output dataset
            # -----------------------------------------------------

            output = xr.Dataset(attrs=ds.attrs)

            # Target spatial coordinates
            output = output.assign_coords(
                x=("x", target_x),
                y=("y", target_y),
            )

            # Copy non-spatial coordinates
            for coord_name, coord in ds.coords.items():
                if coord_name not in ("latitude", "longitude"):
                    output = output.assign_coords(
                        {coord_name: coord}
                    )

            # Target geographic coordinates
            output["longitude"] = (
                ("y", "x"),
                target_lon.astype(np.float32),
            )
            output["latitude"] = (
                ("y", "x"),
                target_lat.astype(np.float32),
            )

            # Copy useful grid metadata if present
            for name in ("central_lon", "central_lat"):
                if name in target:
                    output[name] = target[name]

            # -----------------------------------------------------
            # Regrid spatial variables
            # -----------------------------------------------------

            for variable_name, variable in ds.data_vars.items():

                if (
                    "latitude" not in variable.dims
                    or "longitude" not in variable.dims
                ):
                    # Non-spatial variable: copy unchanged.
                    output[variable_name] = variable
                    continue

                print(f"  Regridding {variable_name}")

                # Put spatial dimensions last.
                non_spatial_dims = [
                    dim
                    for dim in variable.dims
                    if dim not in ("latitude", "longitude")
                ]

                variable = variable.transpose(
                    *non_spatial_dims,
                    "latitude",
                    "longitude",
                )

                data = variable.values

                non_spatial_shape = data.shape[:-2]

                # Flatten e.g.
                #
                # time x valid_time x lat x lon
                #
                # into
                #
                # n_fields x lat x lon
                #
                data = data.reshape(
                    -1,
                    len(source_lat),
                    len(source_lon),
                )

                regridded = np.empty(
                    (
                        data.shape[0],
                        len(target_y),
                        len(target_x),
                    ),
                    dtype=np.float32,
                )

                # One RegularGridInterpolator per 2-D field.
                for field_idx in range(data.shape[0]):

                    interpolator = RegularGridInterpolator(
                        (source_lat, source_lon),
                        data[field_idx],
                        method="linear",
                        bounds_error=True,
                    )

                    regridded[field_idx] = interpolator(
                        target_points
                    ).reshape(
                        len(target_y),
                        len(target_x),
                    )

                # Restore non-spatial dimensions.
                regridded = regridded.reshape(
                    *non_spatial_shape,
                    len(target_y),
                    len(target_x),
                )

                output_dims = tuple(non_spatial_dims) + ("y", "x")

                output[variable_name] = xr.DataArray(
                    regridded,
                    dims=output_dims,
                    attrs=variable.attrs,
                )

            # -----------------------------------------------------
            # Write
            # -----------------------------------------------------

            output.to_netcdf(temp_file)

        if not valid_netcdf(temp_file):
            raise ValueError(
                "Regridded NetCDF failed validation."
            )

        os.replace(temp_file, file_path)

        print(
            f"Successfully regridded {file_path} "
            "onto target y/x grid."
        )

        return True

    except Exception as error:
        print(f"Unable to regrid {file_path}: {error}")
        return False

    finally:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass

def valid_netcdf(file_path):
    """
    Check whether a file is a readable NetCDF file.

    Returns
    -------
    bool
        True if the file appears valid, otherwise False.
    """

    if not os.path.isfile(file_path):
        return False

    if os.path.getsize(file_path) == 0:
        return False

    try:
        with xr.open_dataset(file_path) as ds:
            # Read a small amount of actual data, rather than just
            # checking that the NetCDF header can be opened.
            for variable in ds.data_vars:
                if ds[variable].size > 0:
                    ds[variable].isel(
                        {
                            dim: 0
                            for dim in ds[variable].dims
                        }
                    ).load()
                    break

        return True

    except Exception as error:
        print(f"Invalid/corrupt NetCDF file {file_path}: {error}")
        return False


def download_file(url, destination, timeout=3600):
    """
    Download a NetCDF file safely.

    The download is written to destination.part first. The temporary
    file is only moved to the final destination after it has downloaded
    successfully and passed NetCDF validation.

    Failed, timed-out, or corrupt downloads are deleted.

    Returns
    -------
    bool
        True if the file was successfully downloaded and validated.
    """

    temp_file = destination + ".part"

    max_subprocess_time = (timeout * 3) + 30

    # Remove an old partial download, if present.
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except OSError as error:
            print(f"Unable to remove old partial file {temp_file}: {error}")
            return False

    try:
        result = subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--show-error",
                "--connect-timeout", "20",
                "--max-time", str(timeout),
                "--retry", "2",
                "--retry-delay", "5",
                url,
                "-o", temp_file,
            ],
            timeout=max_subprocess_time,
        )

        if result.returncode != 0:
            print(
                f"Download failed for {url} "
                f"(curl return code {result.returncode})."
            )
            return False

        if not valid_netcdf(temp_file):
            print(f"Downloaded file is corrupt or invalid: {url}")
            return False

        # Atomic rename from .part to final filename.
        os.replace(temp_file, destination)

        print(f"Successfully downloaded and verified {destination}.")

        return True

    except subprocess.TimeoutExpired:
        print(f"Download timed out after {timeout} seconds: {url}")
        return False

    except Exception as error:
        print(f"Error downloading {url}: {error}")
        return False

    finally:
        # If anything went wrong, never leave the .part file behind.
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass


def ensure_netcdf(file_path, urls, timeout=300):
    """
    Ensure that a valid NetCDF file exists locally.

    If an existing file is corrupt, it is deleted.

    Each URL is tried in sequence until a valid file is successfully
    downloaded.

    Parameters
    ----------
    file_path : str
        Destination filename.

    urls : iterable
        Iterable of (source_name, url) tuples.

    timeout : int
        Maximum download time in seconds for each source.

    Returns
    -------
    bool
        True if a valid file exists at the end, otherwise False.
    """

    # ---------------------------------------------------------
    # Check existing file
    # ---------------------------------------------------------

    if os.path.isfile(file_path):

        print(f"{file_path} already exists. Checking file.")

        if valid_netcdf(file_path):
            print(f"{file_path} is valid.")

            return regrid_to_target(
                file_path,
                TARGET_GRID_FILE,
            )

        print(f"{file_path} is corrupt or invalid. Deleting it.")

        try:
            os.remove(file_path)
        except OSError as error:
            print(f"Unable to delete {file_path}: {error}")
            return False

    # ---------------------------------------------------------
    # Make sure destination directory exists
    # ---------------------------------------------------------

    directory = os.path.dirname(file_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    # ---------------------------------------------------------
    # Try each download source
    # ---------------------------------------------------------

    for source_name, url in urls:

        print(f"Trying {source_name}:")
        print(url)

        if download_file(
            url,
            file_path,
            timeout=timeout,
        ):
            if regrid_to_target(
                file_path,
                TARGET_GRID_FILE,
            ):
                return True

            print(
                f"Downloaded {file_path}, but regridding failed."
            )

            # Remove it before trying the next source.
            try:
                os.remove(file_path)
            except OSError:
                pass

            continue

        print(f"Unable to obtain a valid file from {source_name}.")

    print(f"Unable to obtain a valid copy of {file_path}.")

    return False