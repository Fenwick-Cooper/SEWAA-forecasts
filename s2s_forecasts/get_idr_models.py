from pathlib import Path
import subprocess
import zipfile
import shutil
import requests


def idr_models_available(OUT_FOLDER="./idr_models", regionmask_name=''):
    """
    Check whether the idr_models folder contains the expected model files.

    Parameters
    ----------
    OUT_FOLDER : str, optional
        Folder that should contain the model files.
    required_files : list[str] or None, optional
        Specific filenames that must be present. If None, the function checks
        whether the folder contains at least one .joblib file.

    Returns
    -------
    bool
        True if the required model files are present, otherwise False.
    """
    out_dir = Path(OUT_FOLDER)

    if not out_dir.exists():
        return False
    
    folders = [
        f"idr_models_{regionmask_name.split('.')[0]}_{i}wklead" for i in [1,2,3]
    ]

    return all((out_dir / folder).exists() for folder in folders)


def download_idr_models_oxford(OUT_FOLDER="./idr_models", regionmask_name=''):
    """
    Download the idr_models zip from the Oxford web server and unzip it into OUT_FOLDER
    if the model files are not already present.

    Parameters
    ----------
    OUT_FOLDER : str, optional
        Directory where the model files should live, by default "./idr_models".

    Returns
    -------
    None
    """
    server = "https://rain.physics.ox.ac.uk/ICPAC/operational/s2s_forecasts/zipped_files/idr_models.zip"

    out_dir = Path(OUT_FOLDER)
    out_dir.mkdir(parents=True, exist_ok=True)

    if idr_models_available(OUT_FOLDER=out_dir, regionmask_name=regionmask_name):
        print(f"Model files already found in {out_dir}. Nothing to download.")
        return

    zip_path = out_dir / "idr_models.zip"

    print(f"Downloading {server} ...")
    try:
        with requests.get(server, stream=True, timeout=120) as r:
            r.raise_for_status()

            content_type = r.headers.get("Content-Type", "")
            if "zip" not in content_type.lower() and not server.endswith(".zip"):
                print(f"Unexpected content type: {content_type}")
                return

            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    except requests.RequestException as e:
        print(f"Download failed: {e}")
        return

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            print("Archive contents:")
            print(zf.namelist())
            zf.extractall(out_dir)
    except zipfile.BadZipFile:
        print("Downloaded file is not a valid zip archive.")
        return
    finally:
        print(zip_path)
        zip_path.unlink(missing_ok=True)

    print("Download done.")

    if idr_models_available(OUT_FOLDER=out_dir, regionmask_name=regionmask_name):
        print(f"Model files downloaded and found in {out_dir}.")
        return
    else:
        print(f"Model files were not downloaded successfully. Try again later or check paths.")
        return