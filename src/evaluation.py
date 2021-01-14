import xarray as xr
import numpy as np
import pandas as pd
import xskillscore as xs
from dask.diagnostics import ProgressBar


""" Evaluation functions and classes (??) 
# Let's ignore ensemble dimension for a start. 
# TODO for later: Add ensemble option

General workflow structure: 
1. Load in the evaluation data. Do we want to use the dataloader class? Yes! 
    Here, we need the tigge data also for different lead times as well as the different
    ensembles (ignor for now). 
    Leadtimes are already included as option in the dataloader, 
    also the minmax and tp-log scaling can be switched off per options. 

2. Compute baseline or baselines, if several are considered. 

3. Compute metrics. 
    a) Deterministic: FSS, RMSE, F1 
    b) Ensemble: CRPS, Rank histograms
    c) Realism: precip amount histogram/spectra, cell size distributions
"""

#------------ FSS-functions ( copied and adapted from L. Scheck.) ----------------
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . S C O R E _ F S S
#  compute Fractions (skill) score and related quantities (copied from L. Scheck)
#
#  Almost completely adapted from
#  Faggian, Roux, Steinle, Ebert (2015) "Fast calculation of the fractions skill score"
#  MAUSAM, 66, 3, 457-466
#
#"""
#.. module:: score_fss
#:platform: Unix
#:synopsis: Compute the fraction skill score (2D).
#.. moduleauthor:: Nathan Faggian <n.faggian@bom.gov.au>
#"""

def _compute_integral_table(field) :
    return field.cumsum(1).cumsum(0)


def _integral_filter(field, n, table=None) :
    """
    Fast summed area table version of the sliding accumulator.
    :param field: nd-array of binary hits/misses.
    :param n: window size.
    """
    w = n // 2
    if w < 1. :
        return field
    if table is None:
        table = _compute_integral_table(field)

    r, c = np.mgrid[ 0:field.shape[0], 0:field.shape[1] ]
    r = r.astype(np.int)
    c = c.astype(np.int)
    w = np.int(w)
    r0, c0 = (np.clip(r - w, 0, field.shape[0] - 1), np.clip(c - w, 0, field.shape[1] - 1))
    r1, c1 = (np.clip(r + w, 0, field.shape[0] - 1), np.clip(c + w, 0, field.shape[1] - 1))
    integral_table = np.zeros(field.shape).astype(np.int64)
    integral_table += np.take(table, np.ravel_multi_index((r1, c1), field.shape))
    integral_table += np.take(table, np.ravel_multi_index((r0, c0), field.shape))
    integral_table -= np.take(table, np.ravel_multi_index((r0, c1), field.shape))
    integral_table -= np.take(table, np.ravel_multi_index((r1, c0), field.shape))
    return integral_table


def _fss(fcst, obs, threshold, window, fcst_cache=None, obs_cache=None):
    """
    Compute the fraction skill score using summed area tables .
    :param fcst: nd-array, forecast field.
    :param obs: nd-array, observation field.
    :param window: integer, window size.
    :return: tuple of FSS numerator, denominator and score.
    """
    fhat = _integral_filter( fcst > threshold, window, fcst_cache )
    ohat = _integral_filter( obs  > threshold, window, obs_cache  )

    num = np.nanmean(np.power(fhat - ohat, 2))
    denom = np.nanmean(np.power(fhat, 2) + np.power(ohat, 2))
    return num, denom, 1.-num/denom

def _fss_frame(fcst, obs, windows, levels):
    """
    Compute the fraction skill score data-frame.
    :param fcst: nd-array, forecast field.
    :param obs: nd-array, observation field.
    :param window: list, window sizes.
    :param levels: list, threshold levels.
    :return: list, dataframes of the FSS: numerator,denominator and score.
    """
    num_data, den_data, fss_data = [], [], []
    #print(fcst.shape)
    #print(obs.shape)
    for level in levels:
        ftable = _compute_integral_table( fcst > level )
        otable = _compute_integral_table( obs  > level )
        _data = [_fss(fcst, obs, level, w, ftable, otable) for w in windows]
        num_data.append([x[0] for x in _data])
        den_data.append([x[1] for x in _data])
        fss_data.append([x[2] for x in _data])
    return np.array(fss_data) #pd.DataFrame(fss_data, index=levels, columns=windows)
# ------------------- Done with FSS functions -----------------------




def compare_fields(X, y_G = None, y_b = None, y=None, 
                   levels = np.arange(0,10,0.1), cmap='viridis', 
                   eval_mask = None ): 
    """ Compare precipitation fields by making example plots
    TODO: make map plots
    TODO: include eval_mask to shade invalid data
    X: TIGGE field 
    y_G: downscaled field from Generator/CNN, ... 
    y_b: baseline 
    y: radar precip
    """
    
    fig, axs = plt.subplots(1, 4, figsize=[20,4], sharey=True, squeeze=True)
    settings = dict(levels=levels, cmap=cmap)

    X.plot(ax=axs[0], **settings)
    axs[0].set_title('Tigge original')
    
    if y_G:
        y_G.plot(ax=axs[3], **settings)
    axs[1].set_title('downscaled')
    
    y_b.plot(ax=axs[2], **settings)
    axs[2].set_title('baseline')
    
    y.plot(ax=axs[3], **settings)
    axs[3].set_title('MRMS')
    
    return fig, axs

def get_eval_mask(criterion='radarquality', rq_threshold = -1, 
                rq_fn = '/datadrive/mrms/4km/RadarQuality.nc', ds = None): 
    """ Returns a lon-lat mask which area we evaluate on. 
        The radar quality mask is used to determine this. 

        criterion: criterion to apply. ('radarquality') 
        rq_threshold: threshold for 'radarquality'-criterion.   
                    -1 covers everything with radar availability. 
                    Larger thresholds require higher quality. 
        ds: If patchareas criterion, the tiggemrmsm object is required 
                    to get the radarmask for the patches.

        Returns: boolean xr-dataarray, with same lon-lat dimensions 
            as the radar data. 
    """
    if criterion =='radarquality': # use rq>rq-threshold as criterion 
        rq = xr.open_dataarray(rq_fn)
        eval_mask = rq>rq_threshold


    return eval_mask


def get_baseline(X, y, kind = 'interpol', 
                 X_lon=None, X_lat=None, y_lon = None, y_lat =None ): 
    """ Function computes baseline, i.e. interpolates X onto the grid of y. 
    
    This is probably overkill for now, but might be handy later on when we have different baselines

    1. If X and y are given as numpy arrays, transform to xarray 
    2. Apply interpolation
    
    X: Tigge dataset, or sample. Can be xarray format or numpy
    y: Target radar dataset or radar sample corresponding to X. Can be xarray format or numpy 
    kind: kind of baseline to use. So for only linear interpolation is used
    X_lon,X_lat: arrays of longitudes and latitudes for X. Need to be specified only if X is numpy array.
    y_lon,y_lat: arrays of longitudes and latitudes for y. Need to be specified only if y is numpy array.
    
    Returns: baseline downscaled X to the grid of y
    """
    # make sure we have xarrays as inputs, makes interpolation easier
    if type(X) != xr.DataArray:
        X = xr.DataArray(data=X, dims=["lat", "lon"], 
                         coords=dict(lon=("lon", X_lon), lat=("lat", X_lat)))
        # iterative function call:
        return get_baseline(X,y,kind=kind,**kws) 
    if type(y) != xr.DataArray:
        y = xr.DataArray(data=y, dims=["lat", "lon"], 
                         coords=dict(lon=("lon", y_lon), lat=("lat", y_lat)))
        # iterative function call:
        return get_baseline(X,y,kind = kind, **kws)
    
    assert type(X) == xr.DataArray, 'X is not an xarray.'
    assert type(y) == xr.DataArray, 'y is not an xarray.'
    #assert X.dims == y.dims, 'Dimensions of X and y do not match.'
    
        
    # Do the interpolation 
    y_baseline = X.interp_like(y, kwargs = dict(fill_value='extrapolate')) 
    # fill_value: for scipy interoplate, extrapolates values at the boundaries, so no Nans appear! 
    
    assert y_baseline.shape == y.shape, 'y_baseline and y do not have the same size!'
    return y_baseline
    
def compute_eval_metrics(fcst, obs, eval_mask = None, metrics = ['RMSE', 'FSS'],
                     fss_scales=[41,61], fss_thresholds = [1., 5.] ): 
    """ Function to compute evaluation metrics to compare a forecast with observations
    fcst: xr-array of the forecast, e.g. the interpolation baseline or the downscaled forecast
    obs: xr-array fo observations, i.e. radar data 
    eval_mask: 2d boolean mask to apply the evaluation on, i.e. radar quality mask
    metrics: list of metrics to consider
    
    fss_scales: list of spatial scales to use for the FSS calculation
    fss_thresholds: list of precip thresholds to use for the FSS calculation
    
    Returns: xr-dataset with different metrics as different variables   
        Note that this function utilizes dask and returns xarrays with dask arrays. Use
    """
    
    
    if (not 'time' in fcst.dims):
        # fcst and obs have different time dimension names --> need to be the same
        fcst = fcst.rename({'valid_time':'time'})
        
    # rechunking necessary for performance 
    fcst = fcst.chunk({'time':1})
    obs = obs.chunk({'time':1}) 
    
    # apply eval mask: 
    if eval_mask is not None:
        fcst = fcst.where(eval_mask)
        obs = obs.where(eval_mask)
        
    metrics_ds = xr.Dataset()
    # RMSE 
    if 'RMSE' in metrics: 
        rmse = xs.rmse(fcst, 
            obs, dim=['lon', 'lat'], skipna=True)
        metrics_ds['RMSE'] = rmse
        
    if 'FSS' in metrics: # maybe there is a way to implement this faster?
        from dask.diagnostics import ProgressBar

        fss_da = xr.apply_ufunc(_fss_frame, fcst, obs, input_core_dims=[[ 'lat', 'lon'], ['lat', 'lon']],
                       output_core_dims=[['thresholds','scales']], 
                       output_dtypes=[fcst.dtype],
                       dask_gufunc_kwargs = dict(output_sizes= {'scales':len(fss_scales), 'thresholds': len(fss_thresholds)},),
                       vectorize =True, dask='parallelized',
                       kwargs = dict(windows=fss_scales, levels = fss_thresholds))
        fss_da['thresholds'] = np.array(fss_thresholds)
        fss_da['scales'] = np.array(fss_scales)
        #with ProgressBar(minimum=1): 
            #fss_da = fss_da.compute()
        metrics_ds['FSS'] = fss_da
    
    return metrics_ds

def _main(lead_time = 12): 
    
    # 1. Load in data: 
    ds = TiggeMRMSDataset(
    tigge_dir='/datadrive/tigge/32km/',
    tigge_vars=['total_precipitation'],
    mrms_dir='/datadrive/mrms/4km/RadarOnly_QPE_06H/',
    rq_fn='/datadrive/mrms/4km/RadarQuality.nc',
    val_days=7,
    split='valid',
    tp_log=0, scale=False,
    lead_time=12, # Sofar, this can not yet handle arrays of lead time. 
    ) 


    # 2. Compute baseline 
    tigge = ds.tigge.isel(variable=0)
    mrms = ds.mrms
    baseline = get_baseline(tigge, mrms)


    # 3. evaluation mask
    eval_mask = get_eval_mask()
    eval_mask['lat']=mrms.lat # somehow rq has weird lon lat values! 
    eval_mask['lon']=mrms.lon

    # 4. compute metrics: 
    metrics = compute_eval_metrics(baseline, mrms, eval_mask )
    with ProgressBar():
        metrics.load() # execute fss computation    


if __name__ == '__main__':
    Fire(_main)
