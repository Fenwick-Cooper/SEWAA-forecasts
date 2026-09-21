import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import properscoring as ps
import xarray as xr
from isodisreg import idr


LEAD_TIMES = np.arange(0, 3)
PRECIP_FILE_STR = "tprate_sfc"
DEFAULT_S2S_PRECIP_DIR = Path("/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/processed_s2s_data")
DEFAULT_STORAGE_DIR = Path("/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/regional_data/all_countries")
DEFAULT_IMERG_PATH = Path("/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/processed_imerg_data/imerg_weekly_2001_2025_africa_1x1.nc")
DEFAULT_CRPS_OUT_DIR = Path(f"/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/CRPS_results")


def to_noleap(obj):
    """Remove Feb 29 and add a no-leap day-of-year coordinate (1-365)."""
    obj = obj.sel(time=~((obj.time.dt.month == 2) & (obj.time.dt.day == 29)))

    t = obj.time
    doy = t.dt.dayofyear
    doy_nl = xr.where((t.dt.is_leap_year) & (doy > 59), doy - 1, doy)
    return obj.assign_coords(dayofyear_nl=("time", doy_nl.data))


def ensemble_to_features(ds: xr.Dataset, var="precipitation"):
    da = ds[var]
    return xr.Dataset(
        {
            "ens_mean": da.mean(dim="number"),
            "ens_std": da.std(dim="number"),
        }
    )


def load_region_mask(region_mask_path: Path, mask_var: str = "region_mask") -> xr.Dataset:
    mask_ds = xr.load_dataset(region_mask_path)
    if mask_var not in mask_ds:
        raise ValueError(f"Could not find variable '{mask_var}' in {region_mask_path}")
    if "region" not in mask_ds[mask_var].dims:
        raise ValueError(f"Expected '{mask_var}' to have a 'region' dimension.")
    return mask_ds[mask_var]


def regional_mean_from_mask(
        obj: xr.DataArray,
        region_mask: xr.DataArray,
        lat_name: str = "lat",
        lon_name: str = "lon",
        crop: bool = True,
        region_dim: str = "region",
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


def base_dates_for_md(month, day, train_years):
    out = []
    for y in train_years:
        try:
            out.append(pd.Timestamp(year=int(y), month=int(month), day=int(day)))
        except ValueError:
            continue
    return out


def crps_climatology_region(da_train, da_test, offsets=np.arange(-14, 15, 7)):
    """Compute CRPS for region x time using a climatological ensemble from training years."""
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
        raise ValueError("Could not construct any climatological ensemble members.")

    ens_full = xr.concat(ens_list, dim="time")
    da_test = da_test.sel(time=da_test.time.isin(ens_full.time))

    crps = ps.crps_ensemble(
        da_test.transpose("time", "region").values,
        ens_full.values,
        axis=-1,
    )

    return xr.DataArray(
        crps,
        coords={"time": da_test.time, "region": da_test.region},
        dims=["time", "region"],
        name="IMERG_CRPS",
    )


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

        groups = {"ens_mean": 1, "ens_std": 2}
        orders = {"1": "comp", "2": "icx"}

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


def loyo_crps_from_regionmask(region_mask_name: str, test_years, s2s_precip_dir, imerg_path, storage_dir, crps_out_dir):
    print(f"Using region mask: {region_mask_name}")

    region_mask_path = Path(region_mask_name)
    if not region_mask_path.exists():
        raise FileNotFoundError(region_mask_path)

    region_mask_stem = region_mask_path.stem

    storage_dir.mkdir(parents=True, exist_ok=True)

    crps_out_dir = crps_out_dir / f"{region_mask_stem}_meanSTD_LOYO"
    crps_out_dir.mkdir(parents=True, exist_ok=True)

    region_mask_ds = load_region_mask(region_mask_path, mask_var="region_mask")

    imerg_regional_means_path = storage_dir / f"imerg_{region_mask_stem}.nc"
    if imerg_regional_means_path.exists():
        imerg_regional_means = to_noleap(xr.load_dataset(imerg_regional_means_path))
    else:
        imerg_precip = to_noleap(xr.load_dataset(imerg_path))
        imerg_regional_means = regional_mean_from_mask(
            imerg_precip["precipitation"], region_mask_ds
        ).to_dataset(name="precipitation")
        imerg_regional_means.to_netcdf(imerg_regional_means_path)

    for lead in LEAD_TIMES:
        crps_raw_das = []
        crps_idr_das = []
        crps_clim_das = []

        precip_path = s2s_precip_dir / f"{PRECIP_FILE_STR}_{lead}wklead.nc"
        regional_means_path = storage_dir / f"s2s_{lead}wklead_{region_mask_stem}.nc"

        if regional_means_path.exists():
            regional_means = to_noleap(xr.load_dataset(regional_means_path))
        else:
            precip_ds = to_noleap(xr.load_dataset(precip_path))
            regional_means = regional_mean_from_mask(
                precip_ds["precipitation"], region_mask_ds
            ).to_dataset(name="precipitation")
            regional_means.to_netcdf(regional_means_path)

        for year in test_years:
            print(f"Calculating CRPS for lead time {lead} weeks, year {year}...")

            s2s_test_data = regional_means.sel(time=regional_means.time.dt.year.isin([year]))
            imerg_test_data = imerg_regional_means.sel(
                time=imerg_regional_means.time.isin(s2s_test_data.time)
            )

            s2s_train_data = xr.concat(
                (
                    regional_means.sel(time=slice(None, str(int(year - 1)))),
                    regional_means.sel(time=slice(str(int(year + 1)), None)),
                ),
                dim="time",
            )
            imerg_train_data = imerg_regional_means.sel(
                time=imerg_regional_means.time.isin(s2s_train_data.time)
            )

            s2s_train_feat = ensemble_to_features(s2s_train_data)
            s2s_test_feat = ensemble_to_features(s2s_test_data)

            imerg_clim_crps = crps_climatology_region(
                imerg_train_data.precipitation, imerg_test_data.precipitation
            )
            crps_clim_das.append(imerg_clim_crps)

            crps_raw = ps.crps_ensemble(
                imerg_test_data.precipitation, s2s_test_data.precipitation, axis=-1
            )
            crps_raw_da = xr.DataArray(
                crps_raw.transpose(),
                coords={"time": s2s_test_data.time, "region": s2s_test_data.region},
                dims=["time", "region"],
                name="CRPS_raw",
            )
            crps_raw_das.append(crps_raw_da)

            print("Training IDR and calculating CRPS for postprocessed forecasts...")
            models = fit_idr_xarray(
                imerg_train_data.precipitation,
                s2s_train_feat,
            )
            preds = predict_idr_xarray(models, s2s_test_feat)
            crps_idr_da = crps_idr_xarray(preds, imerg_test_data.precipitation)
            crps_idr_das.append(crps_idr_da)

        crps_raw_full = xr.concat(crps_raw_das, dim="time")
        crps_idr_full = xr.concat(crps_idr_das, dim="time")
        crps_clim_full = xr.concat(crps_clim_das, dim="time")

        crps_raw_full.to_netcdf(crps_out_dir / f"crps_raw_{region_mask_stem}_{lead}wklead.nc")
        crps_idr_full.to_netcdf(crps_out_dir / f"crps_idr_{region_mask_stem}_{lead}wklead.nc")
        crps_clim_full.to_netcdf(crps_out_dir / f"crps_clim_{region_mask_stem}_{lead}wklead.nc")

        for region in crps_idr_full.region:
            idr_print = crps_idr_full.sel(region=region).mean("time")
            raw_print = crps_raw_full.sel(region=region).mean("time")
            print(
                f"{region.data}, IDR: {np.around(idr_print.values, 3)}, Raw: {np.around(raw_print.values, 3)}; "
                f"IDR improvement: {np.around(raw_print.values - idr_print.values, 3)}"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region-mask-path",
        required=True,
        help="Path to the NetCDF region mask file",
    )
    parser.add_argument(
        "--test-years",
        required=True,
        nargs="+",
        type=int,
        help="One or more test years for LOYO, e.g. 2021 2022 2023 2024",
    )
    args = parser.parse_args()
    loyo_crps_from_regionmask(
        args.region_mask_path,
        np.array(args.test_years),
        s2s_precip_dir=DEFAULT_S2S_PRECIP_DIR,
        imerg_path=DEFAULT_IMERG_PATH,
        storage_dir=DEFAULT_STORAGE_DIR,
        crps_out_dir=DEFAULT_CRPS_OUT_DIR)
