import argparse
import pickle
from pathlib import Path
import numpy as np
import xarray as xr
from isodisreg import idr


LEAD_TIMES = np.arange(0, 3)
PRECIP_FILE_STR = "tprate_sfc"
DEFAULT_S2S_PRECIP_DIR = Path("/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/processed_s2s_data")
DEFAULT_STORAGE_DIR = Path("/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/regional_data/all_countries")
DEFAULT_IMERG_PATH = Path("/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/processed_imerg_data/imerg_weekly_2001_2025_africa_1x1.nc")
DEFAULT_MODEL_OUT_DIR = Path("/network/group/aopp/predict/AWH029_WRIGHT_S2SPREC/IDR_models")


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


def train_idr_models_from_regionmask(
    region_mask_name: str,
    s2s_precip_dir,
    imerg_path,
    storage_dir,
    model_out_dir,
    ):
    print(f"Using region mask: {region_mask_name}")

    region_mask_path = Path(region_mask_name)
    if not region_mask_path.exists():
        raise FileNotFoundError(region_mask_path)

    region_mask_stem = region_mask_path.stem
    storage_dir.mkdir(parents=True, exist_ok=True)
    model_out_dir.mkdir(parents=True, exist_ok=True)

    region_mask_ds = load_region_mask(region_mask_path, mask_var="region_mask")

    imerg_regional_means_path = storage_dir / f"imerg_{region_mask_stem}.nc"
    if imerg_regional_means_path.exists():
        imerg_regional_means = to_noleap(xr.load_dataset(imerg_regional_means_path))
    else:
        imerg_precip = to_noleap(xr.load_dataset(DEFAULT_IMERG_PATH))
        imerg_regional_means = regional_mean_from_mask(
            imerg_precip["precipitation"], region_mask_ds
        ).to_dataset(name="precipitation")
        imerg_regional_means.to_netcdf(imerg_regional_means_path)

    for lead in LEAD_TIMES:
        print(f"Training IDR on all data for lead time {lead} weeks...")

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

        s2s_feat = ensemble_to_features(regional_means)
        imerg_train_data = imerg_regional_means.sel(time=imerg_regional_means.time.isin(regional_means.time))

        models = fit_idr_xarray(
            imerg_train_data["precipitation"],
            s2s_feat,
        )

        model_path = model_out_dir / f"idr_models_{region_mask_stem}_{lead}wklead.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(models, f)

        print(f"Saved IDR models to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region-mask-path",
        required=True,
        help="Path to the NetCDF region mask file",
    )
    args = parser.parse_args()
    main(
        args.region_mask_path,
        s2s_precip_dir=DEFAULT_S2S_PRECIP_DIR,
        imerg_path=DEFAULT_IMERG_PATH,
        storage_dir=DEFAULT_STORAGE_DIR,
        model_out_dir=DEFAULT_MODEL_OUT_DIR,
        )
