import os
import xarray as xr
import numpy as np

# Settings
root_dir = "/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/rwanda_radar_data/full_radar_data/6h"
output_txt = "zero_nan_files.txt"

zero_files = []
total_files = 0

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if not file.endswith(".nc"):
            continue
        if file.startswith("._"):
            continue

        total_files += 1
        filepath = os.path.join(root, file)

        try:
            with xr.open_dataset(filepath) as ds:
                numeric_vars = [
                    v for v in ds.data_vars
                    if np.issubdtype(ds[v].dtype, np.number)
                ]

                if numeric_vars and all(
                    np.all((ds[v].values == 0) | np.isnan(ds[v].values))
                    for v in numeric_vars
                ):
                    zero_files.append(filepath)
                    print(f"ALL ZERO/NAN: {filepath}")

        except Exception as e:
            print(f"ERROR: {filepath}: {e}")

# Save results
with open(output_txt, "w") as f:
    for filepath in zero_files:
        f.write(filepath + "\n")

# Summary
print("\n========== SUMMARY ==========")
print(f"Total NetCDF files: {total_files}")
print(f"All zero/NaN files: {len(zero_files)}")
print(f"Other files:        {total_files - len(zero_files)}")
print(f"Results saved to:   {output_txt}")
