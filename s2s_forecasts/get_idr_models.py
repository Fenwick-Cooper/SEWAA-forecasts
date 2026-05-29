from pathlib import Path
import platform
import subprocess
import zipfile
import shutil


def idr_models_available(OUT_FOLDER="./idr_models", required_files=None):
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

    if required_files is not None:
        return all((out_dir / fname).exists() for fname in required_files)

    # Default check: any joblib file in the folder
    return any(out_dir.glob("*.joblib"))


def download_idr_models_oxford(OUT_FOLDER="./idr_models"):
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
    server = "https://rain.physics.ox.ac.uk/ICPAC/operational/s2s_forecasts/zipped_files"

    out_dir = Path(OUT_FOLDER)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Used with curl
    if platform.system() == "Windows":
        oblivion = "nul"
    else:
        oblivion = "/dev/null"

    # If the models are already there, do nothing
    if idr_models_available(OUT_FOLDER=OUT_FOLDER):
        print(f"Model files already found in {OUT_FOLDER}. Nothing to download.")
        return

    zip_name = "idr_models.zip"
    zip_path = out_dir / zip_name

    print("Checking University of Oxford for idr_models zip")
    return_value = subprocess.run(
        ["curl", "-Isw", "%{http_code}", server, "-o", oblivion],
        capture_output=True,
        text=True
    )

    if return_value.stdout == "200":
        print(f"Copying idr_models zip from University of Oxford to {zip_path}.")
        subprocess.run(["curl", "-L", server, "-o", str(zip_path)], check=True)

        print(f"Unzipping {zip_path} to {OUT_FOLDER}/.")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)

        # Remove the zip after extraction
        zip_path.unlink(missing_ok=True)

        print("Done.")
    else:
        print(f"Unable to copy idr_models zip from {server}. HTTP error {return_value.stdout}.")