#Import required packages
import os
from pathlib import Path
import xarray as xr
import joblib
import pandas as pd
import numpy as np
from get_idr_models import download_idr_models_oxford

def load_idr_models(path: Path):
    """
    Load a serialized IDR model artifact from disk.

    Parameters
    ----------
    path : pathlib.Path
        Path to a joblib file containing a dictionary-like artifact with at
        least a ``"models"`` key, and typically metadata such as ``"lead"``
        and ``"regions"``.

    Returns
    -------
    models : dict
        Mapping from region name to fitted IDR model.
    artifact : dict
        Full loaded artifact, including models and metadata.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or cannot be loaded.
    """
    #Make Path instance if needed
    if not isinstance(path, Path):
        path = Path(path)
        
    download_idr_models_oxford(OUT_FOLDER=path.parent)

    artifact = joblib.load(path)

    models = artifact["models"]

    print(f"Loaded IDR model artifact from: {path}")
    print(
        "Loaded metadata: "
        f"lead={artifact['lead']}, "
        f"regions={artifact['regions']}"
    )

    return models, artifact

def load_ifs_onelead_regional_mean(year, month, day, lead_times_weeks=1, IN_FOLDER="./s2s_data/regional_means", regionmask_name="admin1_merged_KeEtRwUg.gpkg"):
    """
    Load precomputed IFS regional-mean forecast data for a given initialization
    date and lead time.

    Parameters
    ----------
    year, month, day : int
        Forecast initialization date.
    lead_times_weeks : int, optional
        Lead time in weeks. Default is 1.
    IN_FOLDER : str, optional
        Directory containing the regional-mean NetCDF files.
    regionmask_name : str, optional
        Shapefile name used when creating the regional-mean file. The base name
        is used to construct the NetCDF filename.

    Returns
    -------
    xarray.Dataset
        Dataset containing regional mean forecast variables.

    Raises
    ------
    FileNotFoundError
        If the expected NetCDF file is missing.
    """
    fname = os.path.join(IN_FOLDER, f"{year}-{month:02d}-{day:02d}_tp_meanstd_{lead_times_weeks}wklead_{regionmask_name.split('.')[0]}.nc")

    try:
        ds = xr.open_dataset(fname)
        print(f"Loaded IFS regional mean data from: {fname}")
        return ds
    except FileNotFoundError:
        raise FileNotFoundError(f"Regional mean file not found: {fname}. Make sure to run make_s2s_regional_means.py first to create this file.")

def predict_idr_xarray(models, fcst_new: xr.Dataset):
    """
    Generate IDR predictions for each region in a forecast dataset. Xarray wrapping of IDR package.

    Parameters
    ----------
    models : dict
        Mapping from region name to fitted IDR model. Each model must expose a
        ``predict`` method that accepts a pandas DataFrame with columns
        ``"ens_mean"`` and ``"ens_std"``.
    fcst_new : xarray.Dataset
        Forecast dataset with a ``region`` dimension and variables
        ``tp_mean`` and ``tp_std``.

    Returns
    -------
    dict
        Mapping from region name to model predictions.

    Raises
    ------
    KeyError
        If a region in ``fcst_new`` does not have a corresponding model.
    """
    predictions = {}

    for region in fcst_new.region.values:

        if region not in models:
            raise KeyError(f"No saved IDR model found for region {region}")

        fcst_r = fcst_new.sel(region=region)
        X_new = pd.DataFrame({
            "ens_mean": [fcst_r["tp_mean"].item()],
            "ens_std":  [fcst_r["tp_std"].item()],
        })

        preds = models[region].predict(X_new)
        predictions[region] = preds

    return predictions

def histogram(pred, bins=np.linspace(0,20,11)):
    """
    Histogram of IDR predictions.

    Parameters
    ----------
    preds : IDR prediction object
    bins : int or sequence, optional
        Number of bins or explicit bin edges. Same meaning as numpy.histogram.

    Returns
    -------
    counts : np.ndarray
        Array of length (n_bins)
    """
    cdf_counts = np.append([0],pred.cdf(bins))
    counts = np.diff(cdf_counts)

    # Probabilities: sum of bins equals 1
    total = counts.sum()
    if total > 0:
        counts = counts / total

    return counts

def histogram_regions(preds_by_region, bins=np.linspace(0,20,11)):
    """
    Apply existing histogram() method across regions and return
    a single DataArray with dims ('region', 'bin').

    Parameters
    ----------
    preds_by_region : dict
        Mapping {region_name: IDR_class}
    bins : int or sequence
    range : tuple

    Returns
    -------
    xr.DataArray
    """

    regions = []
    histograms = []
    bins = bins

    for region, preds in preds_by_region.items():

        hist = histogram(
            preds, #Prediction for one region
            bins=bins, #Bins array is UPPER boundary -- so first bin 0 to 0th label, second bin 0th to 1st label
        )

        regions.append(region)
        histograms.append(hist)

    histograms = np.stack(histograms, axis=0)

    da = xr.DataArray(
        histograms,
        dims=("region", "bins"),
        coords={
            "region": regions,
            "bins": bins,
        },
        name="histogram",
    )
    return da.to_dataset(name='counts')

def quantiles(self, qs=np.linspace(0,1,50)):
    """
    Quantiles of IDR predictions.

    Parameters
    ----------
    qs : array-like, optional
        Percentiles to compute, e.g. 1, 3, ..., 99.

    Returns
    -------
    quantiles : np.ndarray
        Array of shape (n_predictions, n_qs)
    qs : np.ndarray
        The percentile levels used.
    """
    predictions = self.predictions
    qs = np.asarray(qs)

    def q0(data):
        x = np.asarray(data.points)
        # If idr.quantile expects probabilities in [0, 1], use qs / 100.
        q = data.quantile(qs)
        return np.asarray(q)

    results = list(map(q0, predictions))
    quantiles = np.vstack(results)

    return quantiles.squeeze(), qs


def quantiles_regions(preds_by_region, qs=np.linspace(0,1,51)):
    """
    Apply quantile() across regions and return a single DataArray with dims
    ('region', 'quantile').

    Parameters
    ----------
    preds_by_region : dict
        Mapping {region_name: IDR_class}
    qs : array-like
        Percentile levels, e.g. 1, 3, ..., 99.

    Returns
    -------
    xr.Dataset
    """
    regions = []
    quantile_vals = []

    for region, preds in preds_by_region.items():
        q = preds.qpred(qs)
        regions.append(region)
        quantile_vals.append(q)

    quantile_vals = np.stack(quantile_vals, axis=0)

    da = xr.DataArray(
        quantile_vals,
        dims=("region", "q"),
        coords={
            "region": regions,
            "q": qs,
        },
    )
    return da.to_dataset(name="tp_value")

def split_and_replicate_regions(ds, dim="region", sep=","):
    """
    Split compound region labels into multiple rows and replicate the
    corresponding data along the region dimension.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Input object with a region-like dimension containing labels that may
        include multiple region names separated by ``sep``.
    dim : str, optional
        Name of the dimension containing region labels. Default is ``"region"``.
    sep : str, optional
        Separator used to split compound labels. Default is ",".

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Object with one row per split region label and updated coordinates.
    """
    labels = ds[dim].values

    new_labels = []
    old_pos = []

    for i, lab in enumerate(labels):
        parts = [p.strip() for p in str(lab).split(sep)]
        new_labels.extend(parts)
        old_pos.extend([i] * len(parts))

    # repeat data along the region dimension
    ds_new = ds.isel({dim: xr.DataArray(old_pos, dims=dim)})
    ds_new = ds_new.assign_coords({dim: new_labels})

    return ds_new

#Main function
def produce_s2s_idr_forecasts(
    year,
    month,
    day,
    lead_times_weeks=[1, 2, 3],
    bins="default",
    IDR_MODEL_FOLDER="./idr_models",
    regionmask_name="admin1_merged_KeEtRwUg.gpkg",
    REGIONAL_MEAN_FOLDER="./s2s_data/regional_means",
    OUT_FOLDER="../interface/view_forecasts/data/counts_s2s",
):
    """
    Produce IDR-based S2S forecast products for one or more lead times.

    For each lead time, this function loads the fitted regional IDR models and
    corresponding regional-mean forecast data, generates predictions, computes
    histogram and quantile products, and saves them to NetCDF files.

    Parameters
    ----------
    year, month, day : int
        Forecast initialization date.
    lead_times_weeks : list of int, optional
        Lead times to process.
    bins : str or sequence, optional
        Histogram bin specification. If ``"default"``, use the built-in weekly
        bin set derived from Fenwick's 6-hourly bins.
    IDR_MODEL_FOLDER : str, optional
        Directory containing serialized IDR model artifacts.
    regionmask_name : str, optional
        Shapefile name used when building the region labels.
    REGIONAL_MEAN_FOLDER : str, optional
        Directory containing regional-mean forecast NetCDF files.
    OUT_FOLDER : str, optional
        Directory where forecast outputs will be written.

    Returns
    -------
    None
    """

    print(f"Producing S2S IDR forecasts for {year}-{month:02d}-{day:02d} with lead times {lead_times_weeks} weeks...")
    print("Using shapefile:", regionmask_name)
    
    #Iterate through lead times, loading models and data, generating predictions, and saving outputs
    for lead in lead_times_weeks:
        print(f"Processing lead time {lead} weeks...")
        idr_models_name = f'idr_models_{regionmask_name.split(".")[0]}_{lead}wklead.joblib'

        #Load IDR model and regional mean data for this date and lead time
        print(f"Processing forecast for {year}-{month:02d}-{day:02d} with lead time {lead} weeks...")
        models, _ = load_idr_models(os.path.join(IDR_MODEL_FOLDER, idr_models_name))
        print(f"Loaded IDR models for lead time {lead} weeks.")
        data = load_ifs_onelead_regional_mean(year, month, day, lead_times_weeks=lead, IN_FOLDER=REGIONAL_MEAN_FOLDER, regionmask_name=regionmask_name)
        print(f"Loaded IFS regional mean data for lead time {lead} weeks.")
        
        #Run prediction
        preds = predict_idr_xarray(models, data)

        if bins == 'default':
            bins = [1, 2.5, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 85, 100, 120, 150, 200, 250, 300, 500, 1000, 5000]
        
        #Produce histogram of predictions for this lead time and save to netcdf
        hist = histogram_regions(
            preds,
            bins=bins
        )
        print(f"Produced histogram of IDR predictions for lead time {lead} weeks.")

        hist = hist.assign_coords(
            time=data.time,
            valid_time=data.valid_time,
        )
        hist = hist.expand_dims(("time", "valid_time")).squeeze()
        hist['country'] = ("region", data.country.values)

        hist = split_and_replicate_regions(hist, dim="region", sep=",")

        #Add metadata
        hist = hist.assign_attrs({
            "version":          "1.0 (2026-06-01)",
            "description":      "Probability ('counts') of weekly mean rainfall in each bin, in each region."
                                "Weekly means are calculated from IFS subseasonal forecast data, calibrated using isotonic distributional regression. "
                                "Weekly means represent the accumulation from time valid_time to valid_time+7 days"
                                "Regions are taken from the regionmask defined in regionmask_name. "
                                "The 'bins' coordinate refers to the upper boundary of the bin: the first bin is rainfall from 0mm to bins[0]; second bin is rainfall from bins[0] to bins[1] etc. "
                                "Probabilities sum to 1. ",
            "regionmask_name":  f"{regionmask_name}",
            "units":            "mm/week",
            })

        #save to netcdf
        os.makedirs(OUT_FOLDER, exist_ok=True)
        hist.to_netcdf(os.path.join(OUT_FOLDER, f"{year}-{month:02d}-{day:02d}_histogram_{regionmask_name.split('.')[0]}_{lead}wklead.nc"))
        print(f"Saved histogram for lead time {lead} weeks to: {OUT_FOLDER}/{year}-{month:02d}-{day:02d}_histogram_{regionmask_name.split('.')[0]}_{lead}wklead.nc")

        #QUANTILES CURRENTLY NOT NEEDED
        #Produce quantiles of predictions for this lead time and save to netcdf
        # pred_quantiles = quantiles_regions(
        #     preds,
        #     qs=np.linspace(0, 1, 51)
        # )
        # print(f"Produced quantiles of IDR predictions for lead time {lead} weeks.")

        # pred_quantiles = pred_quantiles.assign_coords(
        #     time=data.time,
        #     valid_time=data.valid_time,
        # )
        # pred_quantiles = pred_quantiles.expand_dims(("time", "valid_time")).squeeze()
        # pred_quantiles['country'] = ("region", data.country.values)

        # pred_quantiles = split_and_replicate_regions(pred_quantiles, dim="region", sep=",")

        #save to netcdf
        # os.makedirs(OUT_FOLDER, exist_ok=True)
        # pred_quantiles.to_netcdf(os.path.join(OUT_FOLDER, f"{year}-{month:02d}-{day:02d}_quantiles_{regionmask_name.split('.')[0]}_{lead}wklead.nc"))
        # print(f"Saved quantiles for lead time {lead} weeks to: {OUT_FOLDER}/{year}-{month:02d}-{day:02d}_quantiles_{regionmask_name.split('.')[0]}_{lead}wklead.nc")
