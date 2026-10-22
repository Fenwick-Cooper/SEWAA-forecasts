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


BASE_DIR = Path('/Volumes/Rain/LaCie/projDir/data/mdv/radarPolar/ops/sur')
REFERENCE_NC = Path('../../data/example_grids/eg_radar_grid.nc')
OUTPUT_DIR = Path('/Volumes/Rain/rwanda_mdv_to_qpe')
MDV_GLOB = '*.mdv'
FOLDER_REGEX = r'\d{8}'
TEST_MODE = False
TEST_DATE = None   # '2025-04-30' or None
OVERWRITE = False
THRESHOLD = True
NUM_TIMESTEPS_6H = 0.65*72
CAPPI_HEIGHT_ARL_M = 3000.0  # change later; above radar
RADAR_ALTITUDE_M = 1616.0   # override erroneous ODIM 0 m
ROI_M = 2000#1200.0               # 3-D gate search radius; tune after inspection
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#Reflectivity --> precip coefficients
ZR_ALPHA = 200.0 #same as MDV files
ZR_BETA = 1.455 #same as MDV files

#Quality control parameters
MIN_DBZH = 10.0
HAIL_THRESHOLD_DBZH = 55.0

#Parallelisation
N_WORKERS = 4

def timestamp(path):
    date_str = path.parent.name
    time_str = path.stem

    return pd.to_datetime(
        date_str + time_str,
        format='%Y%m%d%H%M%S'
    ) - pd.Timedelta(minutes=5)

# ## Py-ART CAPPI conversion
# `grid_from_radars` uses the full 3-D volume and samples at one fixed altitude. `ROI_M` limits interpolation; unsupported cells remain missing. The output is reflectivity (dBZ), **not QPE**.

def cappi(path):
    radar = pyart.io.read_mdv(
        str(path),
        include_fields=['reflectivity']
    )

    field = 'reflectivity'

    grid = pyart.map.grid_from_radars(
        radar,
        grid_shape=(1, len(y), len(x)),
        grid_limits=(
            (CAPPI_HEIGHT_ARL_M, CAPPI_HEIGHT_ARL_M),
            (float(y[0]), float(y[-1])),
            (float(x[0]), float(x[-1]))
        ),
        grid_origin=(center_lat, center_lon),
        grid_origin_alt=RADAR_ALTITUDE_M,
        grid_projection={
            'proj': 'aeqd',
            'lat_0': center_lat,
            'lon_0': center_lon,
            'datum': 'WGS84'
        },
        fields=[field],
        gridding_algo='map_gates_to_grid',
        roi_func='constant',
        constant_roi=ROI_M,
        weighting_function='Barnes2',
    )

    result = np.ma.filled(
        grid.fields[field]['data'][0],
        np.nan
    ).astype('float32')

    return result

def process_file(item):
    t, path = item

    dbzh = cappi(path).astype(np.float32)

    # Identify weak echoes
    weak = np.isfinite(dbzh) & (dbzh < MIN_DBZH)

    # Cap high reflectivity to reduce hail-related overestimation
    dbzh = np.minimum(dbzh, HAIL_THRESHOLD_DBZH)

    # Convert dBZ to rainfall rate
    Z = 10.0 ** (dbzh / 10.0)
    qpe = (Z / ZR_ALPHA) ** (1.0 / ZR_BETA)

    # Weak echoes -> zero rainfall
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

    folders = sorted(
        p for p in BASE_DIR.iterdir()
        if p.is_dir() and re.fullmatch(FOLDER_REGEX, p.name) and p.name >= '20170101'

    )

    if TEST_DATE:
        folders = [p for p in folders if p.name == TEST_DATE]

    windows = defaultdict(list)

    for folder in folders:
        for path in sorted(folder.glob(MDV_GLOB)):
            t = timestamp(path)
            windows[t.floor('6h')].append((t, path))

    print('Daily folders:', len(folders))
    print('6-hour windows:', len(windows))

    for w, files in sorted(windows.items())[:8]:
        print(w, len(files))

    # Single-volume smoke test
    first = next((items[0][1] for _, items in sorted(windows.items()) if items), None)
    if first is None:
        raise RuntimeError('No input MDV files found')
    example = cappi(first)
    print(first.name, example.shape, 'finite pixels:', np.isfinite(example).sum())

    # ## Process and write NetCDF

    selected = sorted(windows.items())
    if TEST_MODE:
        selected = selected[1:2]

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

            results = list(executor.map(process_file, items))

            times = [t for t, qpe in results]
            arrays = np.stack([qpe for t, qpe in results])

            # Mean rainfall rate, ignoring missing observations
            valid_count = np.isfinite(arrays).sum(axis=0)
            total = np.nansum(arrays, axis=0)

            mean_rate = np.full(arrays.shape[1:], np.nan, dtype=np.float32)
            np.divide(total, valid_count, out=mean_rate, where=valid_count > 0)

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
                    'n_timesteps': len(times),
                    'cappi_height_arl_m': CAPPI_HEIGHT_ARL_M,
                    'cappi_altitude_amsl_m': RADAR_ALTITUDE_M + CAPPI_HEIGHT_ARL_M,
                    'radar_altitude_amsl_m': RADAR_ALTITUDE_M,
                    'interpolation_radius_m': ROI_M,
                    'zr_alpha': ZR_ALPHA,
                    'zr_beta': ZR_BETA,
                    'zr_equation': 'Z = alpha * R^beta',
                    'projection_assumption': 'AEQD centered on reference central_lat/central_lon'
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
                'timesteps:', len(times)
            )

            ds.close()

if __name__ == "__main__":
    start_time = time.perf_counter()

    main()

    elapsed = time.perf_counter() - start_time

    print(f"\nTotal processing time: {elapsed / 60:.2f} minutes")
    print(f"Total processing time: {elapsed / 3600:.2f} hours")