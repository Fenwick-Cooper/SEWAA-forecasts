import json
import re
from pathlib import Path

import importlib.metadata as md
import numpy as np
import pandas as pd
import properscoring as ps
import xarray as xr
from isodisreg import idr
from local_idr import save_idr_model

#Global options
region_mask_name = 'EG' #Exclude .nc
imerg_path = "/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/processed_imerg_data/imerg_weekly_2001_2025_africa_1x1.nc"
lead_times = [1,2,3] #lead times in weeks
precip_file_str = "tprate_sfc"
test_years = np.arange(2005, 2025) #Years to test in LOYO validation, and save CRPS for
subset_regions = None  # e.g. ["Rwanda"]
idr_training_freq = 2  # use every nth timestep for training
force_regmean_calculation = True  # Whether to use existing regional means (if present), or recompute
do_seasonal = False #Whether to split into seasons and train for each season separately
seasons_to_use = {
            "DJF": [12, 1, 2],
            "MAM": [3, 4, 5],
            "JJA": [6, 7, 8],
            "SON": [9, 10, 11],
        } #If do_seasonal = True, these are the seasons that the data will be split into

#Folders to use
#S2S hindcasts -- processed into one file for each variable/lead time
s2s_precip_dir = Path("/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/processed_s2s_data")
#Where to store regional mean data
storage_dir = Path("/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/regional_data/admin1_merged")
#Where to save idr models to
idr_models_dir = Path("/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/idr_models")
#Where the region mask is stored
region_mask_dir = Path("/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/region_masks_for_web")
#Where to save CRPS to
crps_out_dir = Path(
    f"/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/CRPS_results/"
    f"{region_mask_name}_idr_trainevery{idr_training_freq}_annual_allyears"
)

#Helper functions

def package_version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None

def sanitise_filename(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "region"

def base_dates_for_md(month, day, train_years):
    out = []
    for y in train_years:
        try:
            out.append(pd.Timestamp(year=int(y), month=int(month), day=int(day)))
        except ValueError:
            continue
    return out

def crps_climatology_region(da_train, da_test, offsets=np.arange(-14, 15, 7)):
    """
    Compute CRPS using a climatological ensemble built from training years.
    Expects DataArrays with dims including time and region.
    """

    train_years = np.unique(da_train.time.dt.year.values)
    train_time_index = pd.DatetimeIndex(da_train.time.values)
    test_dates = pd.DatetimeIndex(da_test.time.values)
    unique_md = sorted({(d.month, d.day) for d in test_dates})

    ensemble_times_for_md = {}
    for (m, d) in unique_md:
        members = []
        base_list = base_dates_for_md(m, d, train_years)

        for base in base_list:
            for off in offsets:
                member_dt = base + pd.Timedelta(days=int(off))
                if member_dt in train_time_index:
                    members.append(member_dt)

        ensemble_times_for_md[(m, d)] = members

    ens_list = []
    for dt in test_dates:
        if (dt.month, dt.day) == (2, 29):
            continue

        member_times = ensemble_times_for_md.get((dt.month, dt.day), [])
        member_times = [t for t in member_times if t in train_time_index]
        if len(member_times) == 0:
            continue

        ens = da_train.sel(time=member_times)
        ens = ens.rename(time="member")
        ens = ens.assign_coords(member=np.arange(ens.sizes["member"]), time=dt)
        ens_list.append(ens)

    if not ens_list:
        raise ValueError("No matching climatology ensemble dates found.")

    ens_full = xr.concat(ens_list, dim="time")

    # Remove any test dates that did not get a matching climatological ensemble
    da_test = da_test.sel(time=da_test.time.isin(ens_full.time))

    crps = ps.crps_ensemble(
        da_test.values.transpose(1, 0),  # (time, region)
        ens_full.values,                 # (time, region, member)
        axis=-1,
    )

    return xr.DataArray(
        crps,
        coords={"time": da_test.time, "region": da_test.region},
        dims=["time", "region"],
        name="IMERG_CRPS",
    )

def load_fraction_mask(mask_path: Path):
    ds = xr.load_dataset(mask_path)

    if "region_mask" in ds.data_vars:
        mask = ds["region_mask"]
    elif len(ds.data_vars) == 1:
        mask = next(iter(ds.data_vars.values()))
    else:
        raise ValueError(
            f"Could not identify the fraction mask variable in {mask_path}. "
            f"Expected a variable named 'region_mask' or a single data variable."
        )

    return mask

def regional_mean_from_fraction_mask(
    obj: xr.DataArray,
    region_mask: xr.DataArray,
    *,
    region_dim: str = "region",
    lat_name: str = "lat",
    lon_name: str = "lon",
    crop: bool = True,
):
    """
    Area-weighted regional mean using a fractional region mask.

    region_mask should have:
    - one dimension of regions, e.g. region
    - lat/lon dimensions
    - a coordinate called 'country' on the region dimension, if available
    """

    if region_dim not in region_mask.dims:
        raise ValueError(
            f"Expected region_mask to have a '{region_dim}' dimension, "
            f"but found dims={region_mask.dims}"
        )

    # Unique region labels
    if region_dim in region_mask.coords:
        region_labels = region_mask[region_dim].values
    else:
        region_labels = np.arange(region_mask.sizes[region_dim])

    # Optional country labels attached to each region
    country_labels = None
    if "country" in region_mask.coords:
        country_labels = region_mask["country"].values

    if crop:
        spatial_valid = region_mask.fillna(0).sum(dim=region_dim) > 0
        obj = obj.where(spatial_valid, drop=True)
        region_mask = region_mask.where(spatial_valid, drop=True)

    area_weights = xr.DataArray(
        np.cos(np.deg2rad(obj[lat_name])),
        coords={lat_name: obj[lat_name]},
        dims=(lat_name,),
        name="area_weights",
    )

    out = []
    for i, reg_label in enumerate(region_labels):
        frac = region_mask.sel({region_dim: reg_label})

        # Mask the data outside the region, but do NOT replace with zero
        valid = frac.notnull() & (frac > 0)
        data_masked = obj.where(valid)

        # Use fraction mask only as weights
        weights2d = frac.where(valid, 0) * area_weights

        regional = (
            data_masked
            .weighted(weights2d)
            .mean(dim=(lat_name, lon_name))
            .expand_dims(region=[str(reg_label)])
        )

        if country_labels is not None:
            regional = regional.assign_coords(
                country=("region", [str(country_labels[i])])
            )

        out.append(regional)

    return xr.concat(out, dim="region")

def to_noleap(obj):
    """
    Remove Feb 29 and add a no-leap day-of-year coordinate (1–365).
    Works for DataArray or Dataset.
    """
    obj = obj.sel(time=~((obj.time.dt.month == 2) & (obj.time.dt.day == 29)))

    t = obj.time
    doy = t.dt.dayofyear
    doy_nl = xr.where(
        (t.dt.is_leap_year) & (doy > 59),
        doy - 1,
        doy,
    )

    return obj.assign_coords(dayofyear_nl=("time", doy_nl.data))

def fit_idr_xarray(obs: xr.DataArray, fcst: xr.Dataset):
    models = {}

    for region in obs.region.values:
        obs_r = obs.sel(region=region)
        fcst_r = fcst.sel(region=region)

        obs_r, fcst_r = xr.align(obs_r, fcst_r, join="inner")

        X = fcst_r.to_dataframe()
        df = X.copy()
        df["obs"] = obs_r.to_pandas()
        df = df.dropna()

        y = df["obs"]
        X = df.drop(columns=["obs"])
        X = X[["ens_mean", "ens_std"]]

        groups = {
            "ens_mean": 1,
            "ens_std": 2,
        }

        orders = {
            "1": "comp",  # mean monotone
            "2": "icx",    # spread convex
        }

        models[region] = idr(
            y,
            X,
            groups=groups,
            orders=orders,
            eps_rel=1e-3,
            eps_abs=1e-3,
            max_iter=5000,
        )

    return models

def predict_idr_xarray(models, fcst_new: xr.Dataset):
    predictions = {}

    for region in fcst_new.region.values:
        fcst_r = fcst_new.sel(region=region)
        X_new = fcst_r.to_dataframe()
        X_new = X_new[["ens_mean", "ens_std"]]
        predictions[region] = models[region].predict(X_new)

    return predictions

def crps_idr_xarray(predictions, obs: xr.DataArray):
    crps_list = []

    for region in obs.region.values:
        obs_r = obs.sel(region=region)
        preds = predictions[region]
        y = obs_r.to_pandas()

        crps_vals = preds.crps(y)

        da = xr.DataArray(
            crps_vals,
            coords={"time": obs_r.time},
            dims=["time"],
        ).expand_dims(region=[region])

        crps_list.append(da)

    return xr.concat(crps_list, dim="region")

def ensemble_to_features(ds: xr.Dataset, var="precipitation"):
    da = ds[var]
    return xr.Dataset({
        "ens_mean": da.mean(dim="number"),
        "ens_std": da.std(dim="number"),
    })

def save_idr_models(
    models,
    model_dir: Path,
    *,
    shape_name,
    lead,
    train_years,
    region_names,
    idr_training_freq,
):
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "artifact_version": 2,
        "shape_name": shape_name,
        "lead": int(lead),
        "train_years": [int(y) for y in np.unique(train_years)],
        "regions": [str(r) for r in region_names],
        "features": ["ens_mean", "ens_std"],
        "idr_training_freq": int(idr_training_freq),
        "versions": {
            "xarray": package_version("xarray"),
            "pandas": package_version("pandas"),
            "numpy": package_version("numpy"),
            "properscoring": package_version("properscoring"),
            "no_pickle_idr": package_version("no_pickle_idr"),
        },
        "models": {},
    }

    for region, model in models.items():
        safe_region = sanitise_filename(region)
        model_path = model_dir / safe_region
        save_idr_model(model, str(model_path))
        manifest["models"][str(region)] = str(model_path)
        print(f"Saved IDR model for region {region} to: {model_path}")

    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved manifest to: {model_dir / 'manifest.json'}")


def train_idr_models():
    print(f"Using region mask file stem: {region_mask_name}")

    crps_out_dir.mkdir(parents=True, exist_ok=True)
    storage_dir.mkdir(parents=True, exist_ok=True)
    idr_models_dir.mkdir(parents=True, exist_ok=True)
    region_mask_path = region_mask_dir / f"{region_mask_name}.nc"

    if do_seasonal:
        seasons = seasons_to_use
    else:
        seasons = {
            "annual": np.arange(1,13)
        }

    # Load fraction mask
    region_mask = load_fraction_mask(region_mask_path)

    # Optional subset of target regions
    if subset_regions is not None:
        region_dim = None
        for candidate in ("country", "region", "shapeName"):
            if candidate in region_mask.dims:
                region_dim = candidate
                break
        if region_dim is None:
            raise ValueError("Could not infer region dimension for subset selection.")
        region_mask = region_mask.sel({region_dim: subset_regions})

    # Regional means for IMERG
    imerg_regional_means_path = storage_dir / f"imerg_{region_mask_name}.nc"
    if force_regmean_calculation:
        print("Force polygon calculation is True, recalculating regional means for IMERG...")
        imerg_precip = to_noleap(xr.load_dataset(imerg_path))
        imerg_regional_means = regional_mean_from_fraction_mask(
            imerg_precip["precipitation"],
            region_mask,
        ).to_dataset(name="precipitation")
        imerg_regional_means.to_netcdf(imerg_regional_means_path)
    elif imerg_regional_means_path.exists():
        imerg_regional_means = to_noleap(xr.load_dataset(imerg_regional_means_path))
    else:
        imerg_precip = to_noleap(xr.load_dataset(imerg_path))
        imerg_regional_means = regional_mean_from_fraction_mask(
            imerg_precip["precipitation"],
            region_mask,
        ).to_dataset(name="precipitation")
        imerg_regional_means.to_netcdf(imerg_regional_means_path)

    # Regional means for S2S
    for lead in lead_times:
        precip_path = s2s_precip_dir / f"{precip_file_str}_{str(lead - 1)}wklead.nc" #Convention is different -- 1st week is called 0 in hindcast dir
        regional_means_path = storage_dir / f"s2s_{str(lead)}wklead_{region_mask_name}.nc"

        if force_regmean_calculation:
            print(
                f"Force polygon calculation is True, recalculating regional means for S2S lead time {str(lead)}..."
            )
            precip_ds = to_noleap(xr.load_dataset(precip_path))
            regional_means = regional_mean_from_fraction_mask(
                precip_ds["precipitation"],
                region_mask,
            ).to_dataset(name="precipitation")
            regional_means.to_netcdf(regional_means_path)
        elif regional_means_path.exists():
            print(f"Regional means for lead time {str(lead)} already exist, skipping.")
        else:
            print(f"Creating regional means for lead time {str(lead)} weeks...")
            precip_ds = to_noleap(xr.load_dataset(precip_path))
            regional_means = regional_mean_from_fraction_mask(
                precip_ds["precipitation"],
                region_mask,
            ).to_dataset(name="precipitation")
            regional_means.to_netcdf(regional_means_path)

    # CRPS calculations
    for lead in lead_times:
        for season, months in seasons.items():
            crps_raw_das = []
            crps_idr_das = []
            crps_clim_das = []

            print(f"Processing season {season} for lead time {str(lead)} weeks...")

            regional_means_path = storage_dir / f"s2s_{str(lead)}wklead_{region_mask_name}.nc"
            regional_means = to_noleap(xr.load_dataset(regional_means_path))
            regional_means = regional_means.sel(time=regional_means.time.dt.month.isin(months))
            imerg_season = imerg_regional_means.sel(
                time=imerg_regional_means.time.dt.month.isin(months)
            )

            s2s_for_idr, imerg_for_idr = xr.align(regional_means, imerg_season, join="inner")
            s2s_feat = ensemble_to_features(s2s_for_idr)

            idr_models_all = fit_idr_xarray(
                imerg_for_idr.isel(time=slice(0, None, idr_training_freq)).precipitation,
                s2s_feat.isel(time=slice(0, None, idr_training_freq)),
            )

            #Save IDR model trained on all the data
            save_idr_models(
                idr_models_all,
                model_dir=idr_models_dir / f"idr_models_{region_mask_name}_seasonal_{str(lead)}wklead_{season}",
                shape_name=region_mask_name,
                lead=lead,
                train_years=regional_means.time.dt.year.values,
                region_names=regional_means.region.values,
                idr_training_freq=idr_training_freq,
            )

            #Prepare season_year and regional_means
            regional_means_path = storage_dir / f"s2s_{str(lead)}wklead_{region_mask_name}.nc"
            regional_means = to_noleap(xr.load_dataset(regional_means_path))
            regional_means = regional_means.sel(time=regional_means.time.dt.month.isin(months))

            # Adjust season year for seasons spanning the year boundary
            if 12 in months and 1 in months:
                year_break = next(
                    i for i in range(len(months) - 1)
                    if months[i] > months[i + 1]
                )
                late_year_months = months[:year_break + 1]

                season_year = xr.where(
                    regional_means.time.dt.month.isin(late_year_months),
                    regional_means.time.dt.year + 1,
                    regional_means.time.dt.year,
                )
            else:
                season_year = regional_means.time.dt.year

            regional_means = regional_means.assign_coords(
                season_year=("time", season_year.data)
            )

            #Calculate CRPS using LOYO validation for all years in test years
            for year in test_years:
                print(f"Calculating CRPS for lead time {str(lead)} weeks, year {year}...")

                s2s_test_data = regional_means.sel(time=regional_means.season_year.isin([year]))
                imerg_test_data = imerg_regional_means.sel(
                    time=imerg_regional_means.time.isin(s2s_test_data.time)
                )

                s2s_train_data = xr.concat(
                    (
                        regional_means.where(regional_means.season_year <= year - 1, drop=True),
                        regional_means.where(regional_means.season_year >= year + 1, drop=True),
                    ),
                    dim="time",
                ).sortby("time")
                imerg_train_data = imerg_regional_means.sel(
                    time=imerg_regional_means.time.isin(s2s_train_data.time)
                )

                s2s_train_feat = ensemble_to_features(s2s_train_data)
                s2s_test_feat = ensemble_to_features(s2s_test_data)

                imerg_clim_crps = crps_climatology_region(
                    imerg_train_data.precipitation,
                    imerg_test_data.precipitation,
                )
                crps_clim_das.append(imerg_clim_crps)
                print(f"Calculated climatology CRPS for lead time {str(lead)} weeks, year {year}.")

                crps_raw = ps.crps_ensemble(
                    imerg_test_data.precipitation,
                    s2s_test_data.precipitation,
                    axis=-1,
                )
                crps_raw_da = xr.DataArray(
                    crps_raw.transpose(),
                    coords={"time": s2s_test_data.time, "region": s2s_test_data.region},
                    dims=["time", "region"],
                    name="CRPS_raw",
                )
                crps_raw_das.append(crps_raw_da)
                print(f"Calculated raw CRPS for lead time {str(lead)} weeks, year {year}.")

                print("Training IDR and calculating CRPS for postprocessed forecasts...")
                models = fit_idr_xarray(
                    imerg_train_data.precipitation.isel(time=slice(0, None, idr_training_freq)),
                    s2s_train_feat.isel(time=slice(0, None, idr_training_freq)),
                )
                preds = predict_idr_xarray(models, s2s_test_feat)
                crps_idr_da = crps_idr_xarray(preds, imerg_test_data.precipitation)
                crps_idr_das.append(crps_idr_da)
                print(f"Calculated IDR CRPS for lead time {str(lead)} weeks, year {year}.")

            crps_raw_full = xr.concat(crps_raw_das, dim="time").sortby("time")
            crps_idr_full = xr.concat(crps_idr_das, dim="time").sortby("time")
            crps_clim_full = xr.concat(crps_clim_das, dim="time").sortby("time")

            raw_path = crps_out_dir / f"crps_raw_{region_mask_name}_{str(lead)}wklead_{season}.nc"
            idr_path = crps_out_dir / f"crps_idr_{region_mask_name}_{str(lead)}wklead_{season}.nc"
            clim_path = crps_out_dir / f"crps_clim_{region_mask_name}_{str(lead)}wklead_{season}.nc"

            crps_raw_full.to_netcdf(raw_path)
            print(f"Saved raw CRPS for lead time {str(lead)} to {raw_path}")
            crps_idr_full.to_netcdf(idr_path)
            print(f"Saved IDR CRPS for lead time {str(lead)} to {idr_path}")
            crps_clim_full.to_netcdf(clim_path)
            print(f"Saved Climatology CRPS for lead time {str(lead)} to {clim_path}")

            for region in crps_idr_full.region:
                idr_print = crps_idr_full.sel(region=region).mean("time")
                raw_print = crps_raw_full.sel(region=region).mean("time")
                print(
                    f"{region.data}, IDR: {np.around(idr_print.values, 3)}, "
                    f"Raw: {np.around(raw_print.values, 3)}; "
                    f"IDR improvement: {np.around(raw_print.values - idr_print.values, 3)}"
                )


if __name__ == "__main__":
    train_idr_models()