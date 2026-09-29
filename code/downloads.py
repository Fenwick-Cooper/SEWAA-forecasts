import os
import subprocess

import xarray as xr


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
            return True

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
            return True

        print(f"Unable to obtain a valid file from {source_name}.")

    print(f"Unable to obtain a valid copy of {file_path}.")

    return False