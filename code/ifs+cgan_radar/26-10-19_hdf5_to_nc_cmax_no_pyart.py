# conda env
# conda create -n radar_processing -c conda-forge python=3.11 arm_pyart xarray netcdf4 h5py numpy pandas
# conda activate radar_processing

from pathlib import Path
import time
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import xarray as xr
import pyart
from concurrent.futures import ProcessPoolExecutor
import os


BASE_DIR = Path('/Volumes/Rain/new_radar_hdf5_sample')
REFERENCE_NC = Path('../../data/example_grids/eg_radar_grid.nc')
OUTPUT_DIR = Path('/Users/matthewwright/Downloads/test_hdf_files_cmax')
HDF5_GLOB = '*.h5'
FOLDER_REGEX = r'\d{4}-\d{2}-\d{2}'
TIMESTAMP_REGEX = r'(?P<date>\d{4}-\d{2}-\d{2})_(?P<hms>\d{2}-\d{2}-\d{2})'
TEST_MODE = False
TEST_DATE = None   # '2025-04-30' or None
OVERWRITE = True
THRESHOLD = False
NUM_TIMESTEPS_6H = 0.65*72
MIN_HEIGHT_KM = 0.0
MAX_HEIGHT_KM = 30.0
RADAR_ALTITUDE_M = 1600.0   # override erroneous ODIM 0 m
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#Reflectivity --> precip coefficients
ZR_ALPHA = 200.0 #same as MDV files
ZR_BETA = 1.455 #same as MDV files

#Quality control parameters
MIN_DBZH = 10.0
HAIL_THRESHOLD_DBZH = 55.0

#Parallelisation
N_WORKERS = 2

def timestamp(path):
    m = re.search(TIMESTAMP_REGEX, path.stem)
    if not m:
        raise ValueError(f'No timestamp in {path.name}')
    return pd.Timestamp(m['date'] + ' ' + m['hms'].replace('-', ':'))

# ## Py-ART CAPPI conversion
# `grid_from_radars` uses the full 3-D volume and samples at one fixed altitude. `ROI_M` limits interpolation; unsupported cells remain missing. The output is reflectivity (dBZ), **not QPE**.

def cmax(path):

    radar = pyart.aux_io.read_odim_h5(
        str(path),
        file_field_names=False,
        include_fields=["reflectivity_horizontal"]
    )

    radar.altitude["data"] = np.array(
        [RADAR_ALTITUDE_M], dtype="float64"
    )

    field = (
        "reflectivity_horizontal"
        if "reflectivity_horizontal" in radar.fields
        else "DBZH"
    )

    if field not in radar.fields:
        raise KeyError(
            f"Reflectivity field not found: {list(radar.fields)}"
        )

    # Get radar gate coordinates in metres
    # relative to the radar
    gate_x = radar.gate_x["data"]
    gate_y = radar.gate_y["data"]
    gate_z = radar.gate_z["data"]

    dbzh = radar.fields[field]["data"]

    # Flatten arrays
    gx = gate_x.ravel()
    gy = gate_y.ravel()
    gz = gate_z.ravel()

    reflectivity = np.ma.filled(
        dbzh, np.nan
    ).ravel()

    # Filter by height and valid reflectivity
    valid = (
        np.isfinite(reflectivity)
        & np.isfinite(gx)
        & np.isfinite(gy)
        & np.isfinite(gz)
        & (gz >= MIN_HEIGHT_KM * 1000)
        & (gz <= MAX_HEIGHT_KM * 1000)
    )

    gx = gx[valid]
    gy = gy[valid]
    reflectivity = reflectivity[valid]

    # Convert gate coordinates to grid indices
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    ix = np.floor((gx - x[0]) / dx + 0.5).astype(np.int64)
    iy = np.floor((gy - y[0]) / dy + 0.5).astype(np.int64)

    # Keep gates within the reference grid
    inside = (
        (ix >= 0)
        & (ix < len(x))
        & (iy >= 0)
        & (iy < len(y))
    )

    ix = ix[inside]
    iy = iy[inside]
    reflectivity = reflectivity[inside]

    # Flatten 2D grid indices
    flat_index = iy * len(x) + ix

    # Calculate maximum reflectivity
    composite = np.full(
        len(y) * len(x),
        -np.inf,
        dtype=np.float32
    )

    np.maximum.at(composite, flat_index, reflectivity)

    # Cells without radar gates remain missing
    composite[composite == -np.inf] = np.nan

    return composite.reshape(
        len(y), len(x)
    )

def process_file(item):
    t, path = item

    dbzh = cmax(path)

    weak = np.isfinite(dbzh) & (dbzh < MIN_DBZH)

    dbzh = np.minimum(dbzh, HAIL_THRESHOLD_DBZH)

    Z = 10.0 ** (dbzh / 10.0)
    qpe = (Z / ZR_ALPHA) ** (1.0 / ZR_BETA)

    qpe[weak] = 0.0

    return t, qpe.astype(np.float32)

#Reference grid and input windows
with xr.open_dataset(REFERENCE_NC) as ref:
    x, y = ref.x.values.copy(), ref.y.values.copy()
    lat, lon = ref.lat.values.copy(), ref.lon.values.copy()
    center_lat = float(ref.central_lat.values)
    center_lon = float(ref.central_lon.values)
    ref_attrs = dict(ref.attrs)

assert np.allclose(np.diff(x), np.diff(x)[0]) and np.allclose(np.diff(y), np.diff(y)[0]), 'Grid must be regular'
assert np.diff(x)[0] > 0 and np.diff(y)[0] > 0, 'Expected ascending x/y'
print('Grid:', (len(y), len(x)), 'center:', center_lat, center_lon)
print('Reference attributes:', ref_attrs)
print('IMPORTANT: This notebook assumes reference x/y are metres in an azimuthal equidistant (AEQD) projection centered on central_lat/lon. Verify this before processing.')

def main():

    folders = sorted(p for p in BASE_DIR.iterdir() if p.is_dir() and re.fullmatch(FOLDER_REGEX, p.name))
    if TEST_DATE:
        folders = [p for p in folders if p.name == TEST_DATE]
    windows = defaultdict(list)
    for folder in folders:
        for path in sorted(folder.glob(HDF5_GLOB)):
            t = timestamp(path)
            windows[t.floor('6h')].append((t, path))
    print('Daily folders:', len(folders), '6-hour windows:', len(windows))
    for w, files in sorted(windows.items())[:8]:
        print(w, len(files))

    # Single-volume smoke test
    first = next((items[0][1] for _, items in sorted(windows.items()) if items), None)
    if first is None:
        raise RuntimeError('No input HDF5 files found')
    example = cmax(first)
    print(first.name, example.shape, 'finite pixels:', np.isfinite(example).sum())

    # ## Process and write NetCDF

    selected = sorted(windows.items())
    if TEST_MODE:
        selected = selected[:1]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:

        for start, records in selected:
            unique = dict((t, path) for t, path in sorted(records))

            if THRESHOLD and len(unique) < NUM_TIMESTEPS_6H:
                print('SKIP incomplete:', start, len(unique))
                continue

            out = OUTPUT_DIR / f'{start:%Y%m%d_%H}.nc'
            if out.exists() and not OVERWRITE:
                print('SKIP existing:', out)
                continue

            items = sorted(unique.items())
            if not items:
                print("SKIP empty window:", start)
                continue

            total = np.zeros((len(y), len(x)), dtype=np.float64)
            valid_count = np.zeros((len(y), len(x)), dtype=np.int32)

            n_processed = 0

            for t, qpe in executor.map(
                process_file,
                items,
                chunksize=2
            ):
                valid = np.isfinite(qpe)

                np.add(total, qpe, out=total, where=valid)
                valid_count += valid

                n_processed += 1

            mean_rate = np.full(
                (len(y), len(x)),
                np.nan,
                dtype=np.float32
            )

            np.divide(
                total,
                valid_count,
                out=mean_rate,
                where=valid_count > 0
            )

            ds = xr.Dataset(
                {
                    'rainfall_rate': (('y', 'x'), mean_rate)
                },
                coords={
                    'time': pd.Timestamp(start),
                    'y': y,
                    'x': x,
                    'lat': (('y', 'x'), lat),
                    'lon': (('y', 'x'), lon)
                },
                attrs={
                    'description': 'Mean radar rainfall rate over 6-hour window',
                    'aggregation': 'mean of available 5-minute rainfall rates',
                    'window_start': str(start),
                    'window_end': str(start + pd.Timedelta(hours=6)),
                    'n_timesteps': n_processed,
                    'radar_altitude_amsl_m': RADAR_ALTITUDE_M,
                    'zr_alpha': ZR_ALPHA,
                    'zr_beta': ZR_BETA,
                    'zr_equation': 'Z = alpha * R^beta',
                    'projection_assumption': 'AEQD centered on reference central_lat/central_lon',
                    'composite_method': 'maximum reflectivity from radar gates',
                    'gridding_method': 'nearest horizontal grid cell',
                    'minimum_height_arl_m': MIN_HEIGHT_KM * 1000,
                    'maximum_height_arl_m': MAX_HEIGHT_KM * 1000,
                }
            )

            ds['rainfall_rate'].attrs.update(
                units='mm h-1',
                long_name='Mean radar rainfall rate over 6 hours',
                cell_methods='time: mean',
                standard_name='rainfall_rate'
            )

            ds.to_netcdf(
                out,
                engine='netcdf4',
                encoding={
                    'rainfall_rate': {
                        'zlib': True,
                        'complevel': 4,
                        'dtype': 'float32',
                        '_FillValue': np.float32(-9999)
                    }
                }
            )

            print(
                'WROTE:', out,
                'shape:', ds.rainfall_rate.shape,
                'timesteps:', n_processed
            )

            ds.close()

if __name__ == "__main__":
    start_time = time.perf_counter()

    main()

    elapsed = time.perf_counter() - start_time

    print(f"\nTotal processing time: {elapsed / 60:.2f} minutes")
    print(f"Total processing time: {elapsed / 3600:.2f} hours")