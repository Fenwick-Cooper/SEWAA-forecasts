# Shouldn't be needed any more as IDR included; and geopandas dependency removed.

import importlib.util
import subprocess
import sys
from pathlib import Path
import zipfile
import shutil
import os

ZIP_PATH = Path("./s2s_forecasts/isodisreg-master.zip")
TARGET_DIR = Path("./isodisreg")

def pip_install(*args, env=None):
    """
    Install one or more Python packages using the current interpreter's pip.

    Parameters
    ----------
    *args : str
        Arguments passed directly to ``pip install``.
    env : dict, optional
        Environment variables to use for the subprocess call.

    Returns
    -------
    None

    Raises
    ------
    subprocess.CalledProcessError
        If the pip installation fails.
    """
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *args],
        env=env
    )

def ensure_package(pkg):
    """
    Ensure that a Python package is importable, installing it if needed.

    Parameters
    ----------
    pkg : str
        Name of the package to check and install if missing.

    Returns
    -------
    None
    """
    if importlib.util.find_spec(pkg) is None:
        print(f"Installing missing dependency: {pkg}")
        pip_install(pkg)

def ensure_packages(extra_packages=None):
    """
    Ensure required dependencies are available, installing them if necessary.

    If ``isodisreg`` is already installed, the function returns immediately.
    Otherwise it installs required dependencies such as ``geopandas``,
    optional extra packages, and build dependencies, then installs
    ``isodisreg`` from the local zip archive.

    Parameters
    ----------
    extra_packages : iterable of str, optional
        Additional packages to ensure are installed before building
        ``isodisreg``.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the expected zip archive is missing.
    subprocess.CalledProcessError
        If package installation fails.
    """
    print("Ensuring required packages")
    #Extra specified packages
    if extra_packages is not None:
        for pkg in extra_packages:
            ensure_package(pkg)

    def is_isodisreg_healthy():
        try:
            import isodisreg  # noqa: F401
            import isodisreg.modeling_evaluation  # noqa: F401
            return True
        except Exception:
            return False

    if is_isodisreg_healthy():
        print("isodisreg is already available")
        return

    # If the package folder exists but import fails, remove it and reinstall
    if TARGET_DIR.exists():
        print("isodisreg folder exists but import failed; deleting folder")
        shutil.rmtree(TARGET_DIR)
    # ISODISREG
    ensure_package("pybind11")
    ensure_package("wheel")
    ensure_package("setuptools")

    print("isodisreg not found, installing from zip...")

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(".")

    extracted_candidates = [
        p for p in Path(".").iterdir()
        if p.is_dir() and p.name.startswith("isodisreg")
    ]

    if not TARGET_DIR.exists():
        extracted_candidates[0].rename(TARGET_DIR)

    env = os.environ.copy()

    # Avoid broken conda linker
    env.pop("LD", None)
    env.pop("LDFLAGS", None)

    env["CC"] = "/usr/bin/gcc"
    env["CXX"] = "/usr/bin/g++"

    pip_install(
        "--no-build-isolation",
        "./isodisreg",
        env=env
    )

    print("isodisreg installed successfully")
