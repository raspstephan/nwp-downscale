import xarray as xr
import numpy as np
import pandas as pd
import xskillscore as xs
from dask.diagnostics import ProgressBar
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from datetime import date
import subprocess
import torch
import torch.nn as nn
from src.dataloader import log_retrans
import tqdm.notebook as tqdm
from pytictoc import TicToc
from multiprocess import Pool
import multiprocessing as mp
from tqdm import tqdm
from src.regrid import regrid
import torch.nn.functional as F

"""
Eval Functions
"""

def tigge_interp_patch_eval(dl_test, ds_min, ds_max, tp_log, device):
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """
    
    t = TicToc()
    crps = []
    rmse = []
    max_pool_crps = []
    avg_pool_crps = []
    rhist = []
    rels_1 = []
    rels_4 = []
    pred_means = []
    pred_hists = []
    truth_means = []
    truth_hists = []
    preds_fss = []
    preds_brier_1 = []
    preds_brier_5 = []
    preds_brier_10 = []
    
    t.tic()
    num_workers = mp.cpu_count()
    print("num_workers:", num_workers)
    pool = Pool(processes=num_workers)
    t.toc('Setting up the pool took')
    
    print(f"Total batches: {len(dl_test)}")
    def log_result(result):
        for res in result:
            crps.append(res[0])
            max_pool_crps.append(res[1])
            avg_pool_crps.append(res[2])
            rmse.append(res[3])
            rhist.append(res[4])
            rels_1.append(res[5])
            rels_4.append(res[6])
            preds_brier_1.append(res[7])
            preds_brier_5.append(res[8])
            preds_brier_10.append(res[9])
            
        print("batch complete")
        print(f"current len of crps {len(crps)}")
            
    for batch_idx, (x,y) in enumerate(dl_test):
        t.tic()
        x = x.to(device)
        print(x.shape)
        preds = F.interpolate(x, size = (128, 128), mode='bilinear').detach().to('cpu').numpy().squeeze()
        print(preds.shape)
        preds = preds.transpose(1,0,2,3)
        truth = y.numpy().squeeze(1)
        
        print("preds.shape", preds.shape)
        
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        preds = xr.DataArray(
                preds,
                dims=['member', 'sample', 'lat', 'lon'],
                name='tp'
            )

        truth = truth * (ds_max - ds_min) + ds_min

        preds = preds * (ds_max - ds_min) + ds_min

        if tp_log:
            truth = log_retrans(truth, tp_log)
            preds = log_retrans(preds, tp_log)
        
        mean_fss = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 4, window=25, device=device)
        
        preds_fss.append(mean_fss)
        
        eps = 1e-6
        bin_edges = [-eps] + np.linspace(eps, log_retrans(ds_max, tp_log)+eps, 51).tolist()
        pred_means.append(np.mean(preds.sel(member=0)))
        pred_hists.append(np.histogram(preds.sel(member=0), bins = bin_edges, density=False)[0])
        truth_means.append(np.mean(truth))
        truth_hists.append(np.histogram(truth, bins = bin_edges, density=False)[0])
        
        truth_pert = truth + np.random.normal(scale=1e-6, size=truth.shape)
        preds_pert = preds + np.random.normal(scale=1e-6, size=preds.shape) 

        pool.starmap_async(compute_metrics, [(truth, preds, truth_pert, preds_pert, i) for i in range(x.shape[0])], callback=log_result).wait()
        t.toc('batch_took')
        
        

    rels_1 = xr.concat(rels_1, dim = "patch")
    weights_1 = rels_1.samples / rels_1.samples.sum(dim="patch")
    weighted_relative_freq_1 = (weights_1*rels_1.relative_freq).sum(dim="patch")
    samples_1 = rels_1.samples.sum(dim="patch")
    forecast_probs_1 = rels_1.forecast_probability
    
    rels_4 = xr.concat(rels_4, dim = "patch")
    weights_4 = rels_4.samples / rels_4.samples.sum(dim="patch")
    weighted_relative_freq_4 = (weights_4*rels_4.relative_freq).sum(dim="patch")
    samples_4 = rels_4.samples.sum(dim="patch")
    forecast_probs_4 = rels_4.forecast_probability
    
    rhist = [sum([h[i] for h in rhist]) for i in range(11)]
    
    pred_hists = (np.sum(np.array(pred_hists), axis=0), bin_edges)
    truth_hists = (np.sum(np.array(truth_hists), axis=0), bin_edges)
    
    print(f"total in pres hist {np.sum(pred_hists[0])}, total in true hist {np.sum(truth_hists[0])}")
    
    metrics = {"crps": np.mean(crps), 
               "max_pool_crps": np.mean(max_pool_crps), 
               "avg_pool_crps": np.mean(avg_pool_crps),
               "rankhist": rhist, 
               "reliability_1": (weighted_relative_freq_1, forecast_probs_1, samples_1), 
               "reliability_4": (weighted_relative_freq_4, forecast_probs_4, samples_4), 
               "rmse": np.mean(rmse), 
               "true_mean": np.mean(truth_means),
               "preds_mean": np.mean(pred_means), 
               "true_hist": truth_hists,
               "preds_hist": pred_hists, 
               "fss": np.mean(preds_fss),
                "preds_brier_1" : np.mean(preds_brier_1),
                "preds_brier_5" : np.mean(preds_brier_5),
                "preds_brier_10": np.mean(preds_brier_10)
              }
    
    
    return metrics


def compute_metrics(truth, preds, truth_pert, preds_pert, sample):
    sample_crps = xs.crps_ensemble(truth.sel(sample=sample), preds.sel(sample=sample)).values
    truth_course = truth.coarsen(lat=4, lon=4)
    preds_course = preds.coarsen(lat=4, lon=4)
    sample_max_pool_crps = xs.crps_ensemble(truth_course.max().sel(sample=sample), preds_course.max().sel(sample=sample)).values
    sample_avg_pool_crps = xs.crps_ensemble(truth_course.mean().sel(sample=sample), preds_course.mean().sel(sample=sample)).values
    sample_rmse = xs.rmse(preds.sel(sample=sample).mean('member'), truth.sel(sample=sample), dim=['lat', 'lon']).values
    rhist = xs.rank_histogram(truth_pert.sel(sample=sample), preds_pert.sel(sample=sample)).values
    
    rel1 = xs.reliability(truth.sel(sample=sample)>1,(preds.sel(sample=sample)>1).mean('member'))
    rel1 = xr.where(np.isnan(rel1), 0, rel1)
    rel1['relative_freq'] = rel1
    
    rel4 = xs.reliability(truth.sel(sample=sample)>4,(preds.sel(sample=sample)>4).mean('member'))
    rel4 = xr.where(np.isnan(rel4), 0, rel4)
    rel4['relative_freq'] = rel4

    
    sample_brier_1 = xs.brier_score(truth.sel(sample=sample) > 1.0, (preds.sel(sample=sample) > 1.0).mean('member'), dim=['lat', 'lon'])
    
    sample_brier_5 = xs.brier_score(truth.sel(sample=sample) > 5.0, (preds.sel(sample=sample) > 5.0).mean('member'), dim=['lat', 'lon'])
        
    sample_brier_10 = xs.brier_score(truth.sel(sample=sample) > 10.0, (preds.sel(sample=sample) > 10.0).mean('member'), dim=['lat', 'lon'])
    
    return (sample_crps, sample_max_pool_crps, sample_avg_pool_crps, sample_rmse, rhist, rel1, rel4, sample_brier_1, sample_brier_5, sample_brier_10)
    

    
def fss(x,y,threshold, window, device):
    x_mask = x>=threshold
    y_mask = y>=threshold
    window_size=window**2
    mse = []
    yin = torch.from_numpy(y_mask.values.astype(np.float32)).unsqueeze(1).to(device)
    y_out = F.avg_pool2d(yin, window, stride=1, padding=0)
    for member in range(x_mask.shape[1]):
        xin = torch.from_numpy(x_mask[:,member:member+1,:,:].values.astype(np.float32)).to(device)
        x_out = F.avg_pool2d(xin, window, stride=1, padding=0)
        mseij = torch.mean(torch.square(x_out - y_out))    
        mse_ref = torch.mean(torch.square(x_out)) +  torch.mean(torch.square(y_out))
        if mse_ref == 0:
            continue
        fss_ij = 1 - (mseij / mse_ref)
        mse.append(fss_ij.detach().cpu().numpy())
             
    return np.mean(mse)
       
    #for sample in range(x_mask.shape[0])for sample in range(x_mask.shape[0]):
#        yin = torch.from_numpy(y_mask[sample:sample+1,:,:].values.astype(np.float32)).unsqueeze(1).to(device)
#        y_out = conv(yin)/window_size
#        for member in range(x_mask.shape[1]):
#            xin = torch.from_numpy(x_mask[sample:sample+1,member:member+1,:,:].values.astype(np.float32)).transpose(0,1).to(device)  
#            x_out = conv(xin)/window_size
#            mseij = torch.mean(torch.square(x_out - y_out))    
#            mse_ref = torch.mean(torch.square(x_out)) +  torch.mean(torch.square(y_out))
#            if mse_ref == 0:
#                continue
#            fss_ij = 1 - (mseij / mse_ref)
#            mse.append(fss_ij.detach().cpu().numpy())
            
#    return np.mean(mse)

def par_gen_full_field_eval(gen, ds_test, nens, ds_min, ds_max, tp_log, device):
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """  
    
    timer = TicToc()
    crps = []
    rmse = []
    max_pool_crps = []
    avg_pool_crps = []
    rhist = []
    rels_1 = []
    rels_4 = []
    pred_means = []
    pred_hists = []
    truth_means = []
    truth_hists = []
    
    timer.tic()
    num_workers = mp.cpu_count()
    print("num_workers:", num_workers)
    pool = Pool(processes=num_workers)
    timer.toc('Setting up the pool took')
    
    print(f"Total batches: {len(ds_test.tigge.valid_time)}")
    def log_result(result):
        for res in result:
            crps.append(res[0])
            max_pool_crps.append(res[1])
            avg_pool_crps.append(res[2])
            rmse.append(res[3])
            rhist.append(res[4])
            rels_1.append(res[5])
            rels_4.append(res[6])
            
        print("batch complete")
        print(f"current len of crps {len(crps)}")
            
    full_preds = []
    full_truth = []
    for idx, t in enumerate(tqdm(range(len(ds_test.tigge.valid_time)))):
        x, y = ds_test.return_full_array(t)
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x).to(device)
        preds = []
        for i in range(nens):
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3]).to(device)
            pred = gen(x, noise).detach().to('cpu').numpy().squeeze()
            preds.append(pred)

        full_preds.append(preds)
        truth = y.squeeze(0)
        full_truth.append(truth)
        if idx>30:
            break

    timer.tic()
    
    preds = np.array(full_preds)
    truth = np.array(full_truth)
    

    preds = xr.DataArray(
                preds,
                dims=['sample', 'member', 'lat', 'lon'],
                name='tp'
            )
        
    truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )

    truth = truth * (ds_max - ds_min) + ds_min

    preds = preds * (ds_max - ds_min) + ds_min

    if tp_log:
        truth = log_retrans(truth, tp_log)
        preds = log_retrans(preds, tp_log)
        
    timer.toc("xarray, scaling done in", restart=True)
    # get mask   
    ds = xr.open_dataset('/home/jupyter/data/hrrr/raw/total_precipitation/20180215_00.nc')
    ds_regridded = regrid(ds, 4, lons=(235, 290), lats=(50, 20))
    hrrr_mask = np.isfinite(ds_regridded).tp.isel(init_time=0, lead_time=0)
    
    timer.toc("hrrr mask loaded in", restart=True)
    rq = xr.open_dataarray(f'/home/jupyter/data/mrms/4km/RadarQuality.nc')
    mrms_mask = rq>-1
    mrms_mask = mrms_mask.assign_coords({
        'lat': hrrr_mask.lat,
        'lon': hrrr_mask.lon
    })
    total_mask = mrms_mask * hrrr_mask
    total_mask = total_mask.isel(lat=slice(0, -6))
    total_mask = total_mask.assign_coords({'lat': truth.lat.values, 'lon': truth.lon.values})
    
    timer.toc("total mask computed in", restart=True)
#     apply mask
    truth = truth.where(total_mask)
    preds = preds.where(total_mask)
    
    timer.toc("mask applied in", restart=True)
    # compute fss
    mean_fss = fss(preds,truth, 4, 25, device)
    
    eps = 1e-6
    bin_edges = [-eps] + np.linspace(eps, log_retrans(ds_max, tp_log)+eps, 51).tolist()
    pred_means.append(np.mean(preds.sel(member=0)))
    pred_hists.append(np.histogram(preds.sel(member=0), bins = bin_edges, density=False)[0])
    truth_means.append(np.mean(truth))
    truth_hists.append(np.histogram(truth, bins = bin_edges, density=False)[0])

    truth_pert = truth + np.random.normal(scale=1e-6, size=truth.shape)
    preds_pert = preds + np.random.normal(scale=1e-6, size=preds.shape) 

    pool.starmap_async(compute_metrics, [(truth, preds, truth_pert, preds_pert, i) for i in range(preds.shape[0])], callback=log_result).wait()

        
        
    timer.toc("compute metrics completed in", restart=True)
    
    rels_1 = xr.concat(rels_1, dim = "time")
    print(rels_1)
    weights_1 = rels_1.samples / rels_1.samples.sum(dim="time")
    weighted_relative_freq_1 = (weights_1*rels_1.relative_freq).sum(dim="time")
    samples_1 = rels_1.samples.sum(dim="time")
    forecast_probs_1 = rels_1.forecast_probability
    
    rels_4 = xr.concat(rels_4, dim = "time")
    weights_4 = rels_4.samples / rels_4.samples.sum(dim="time")
    weighted_relative_freq_4 = (weights_4*rels_4.relative_freq).sum(dim="time")
    samples_4 = rels_4.samples.sum(dim="time")
    forecast_probs_4 = rels_4.forecast_probability
    
    rhist = [sum([h[i] for h in rhist]) for i in range(nens+1)]
    
    pred_hists = (np.sum(np.array(pred_hists), axis=0), bin_edges)
    truth_hists = (np.sum(np.array(truth_hists), axis=0), bin_edges)
    
    print(f"total in pres hist {np.sum(pred_hists[0])}, total in true hist {np.sum(truth_hists[0])}")
    
    metrics = {"crps": np.mean(crps), 
               "max_pool_crps": np.mean(max_pool_crps), 
               "avg_pool_crps": np.mean(avg_pool_crps),
               "rankhist": rhist, 
               "reliability_1": (weighted_relative_freq_1, forecast_probs_1, samples_1), 
               "reliability_4": (weighted_relative_freq_4, forecast_probs_4, samples_4), 
               "rmse": np.mean(rmse), 
               "true_mean": np.mean(truth_means),
               "preds_mean": np.mean(pred_means), 
               "true_hist": truth_hists,
               "preds_hist": pred_hists, 
               "fss": mean_fss
              }
    
    
    return metrics

def par_SR_gen_patch_eval(gen, dl_test, nens, ds_min, ds_max, tp_log, device):
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """
            
    
    t = TicToc()
    crps = []
    rmse = []
    max_pool_crps = []
    avg_pool_crps = []
    rhist = []
    rels_1 = []
    rels_4 = []
    pred_means = []
    pred_hists = []
    truth_means = []
    truth_hists = []
    preds_fss = []
    preds_brier_1 = []
    preds_brier_5 = []
    preds_brier_10 = []
    
    t.tic()
    num_workers = mp.cpu_count()
    print("num_workers:", num_workers)
    pool = Pool(processes=num_workers)
    t.toc('Setting up the pool took')
    
    print(f"Total batches: {len(dl_test)}")
    def log_result(result):
        for res in result:
            crps.append(res[0])
            max_pool_crps.append(res[1])
            avg_pool_crps.append(res[2])
            rmse.append(res[3])
            rhist.append(res[4])
            rels_1.append(res[5])
            rels_4.append(res[6])
            preds_brier_1.append(res[7])
            preds_brier_5.append(res[8])
            preds_brier_10.append(res[9])
            
        print("batch complete")
        print(f"current len of crps {len(crps)}")
            
    for batch_idx, (x,y) in enumerate(dl_test):
        t.tic()
        x = x.to(device)
        preds = []
        for i in range(x.shape[1]):
            noise = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3]).to(device)
            pred = gen(x[:,i:i+1,:,:], noise).detach().to('cpu').numpy().squeeze()
            preds.append(pred)
        preds = np.array(preds)
        truth = y.numpy().squeeze(1)
        
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        preds = xr.DataArray(
                preds,
                dims=['member', 'sample', 'lat', 'lon'],
                name='tp'
            )

        truth = truth * (ds_max - ds_min) + ds_min

        preds = preds * (ds_max - ds_min) + ds_min

        if tp_log:
            truth = log_retrans(truth, tp_log)
            preds = log_retrans(preds, tp_log)
        
        mean_fss = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 4, window=25, device=device)
        
        preds_fss.append(mean_fss)
        
        eps = 1e-6
        bin_edges = [-eps] + np.linspace(eps, log_retrans(ds_max, tp_log)+eps, 51).tolist()
        pred_means.append(np.mean(preds.sel(member=0)))
        pred_hists.append(np.histogram(preds.sel(member=0), bins = bin_edges, density=False)[0])
        truth_means.append(np.mean(truth))
        truth_hists.append(np.histogram(truth, bins = bin_edges, density=False)[0])
        
        truth_pert = truth + np.random.normal(scale=1e-6, size=truth.shape)
        preds_pert = preds + np.random.normal(scale=1e-6, size=preds.shape) 

        pool.starmap_async(compute_metrics, [(truth, preds, truth_pert, preds_pert, i) for i in range(x.shape[0])], callback=log_result).wait()
        t.toc('batch_took')
        
        

    rels_1 = xr.concat(rels_1, dim = "patch")
    weights_1 = rels_1.samples / rels_1.samples.sum(dim="patch")
    weighted_relative_freq_1 = (weights_1*rels_1.relative_freq).sum(dim="patch")
    samples_1 = rels_1.samples.sum(dim="patch")
    forecast_probs_1 = rels_1.forecast_probability
    
    rels_4 = xr.concat(rels_4, dim = "patch")
    weights_4 = rels_4.samples / rels_4.samples.sum(dim="patch")
    weighted_relative_freq_4 = (weights_4*rels_4.relative_freq).sum(dim="patch")
    samples_4 = rels_4.samples.sum(dim="patch")
    forecast_probs_4 = rels_4.forecast_probability
    
    rhist = [sum([h[i] for h in rhist]) for i in range(nens+1)]
    
    pred_hists = (np.sum(np.array(pred_hists), axis=0), bin_edges)
    truth_hists = (np.sum(np.array(truth_hists), axis=0), bin_edges)
    
    print(f"total in pres hist {np.sum(pred_hists[0])}, total in true hist {np.sum(truth_hists[0])}")
    
    metrics = {"crps": np.mean(crps), 
               "max_pool_crps": np.mean(max_pool_crps), 
               "avg_pool_crps": np.mean(avg_pool_crps),
               "rankhist": rhist, 
               "reliability_1": (weighted_relative_freq_1, forecast_probs_1, samples_1), 
               "reliability_4": (weighted_relative_freq_4, forecast_probs_4, samples_4), 
               "rmse": np.mean(rmse), 
               "true_mean": np.mean(truth_means),
               "preds_mean": np.mean(pred_means), 
               "true_hist": truth_hists,
               "preds_hist": pred_hists, 
               "fss": np.mean(preds_fss),
                "preds_brier_1" : np.mean(preds_brier_1),
                "preds_brier_5" : np.mean(preds_brier_5),
                "preds_brier_10": np.mean(preds_brier_10)
              }
    
    
    return metrics
    
def par_gen_patch_eval(gen, dl_test, nens, ds_min, ds_max, tp_log, device):
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """
            
    
    t = TicToc()
    crps = []
    rmse = []
    max_pool_crps = []
    avg_pool_crps = []
    rhist = []
    rels_1 = []
    rels_4 = []
    pred_means = []
    pred_hists = []
    truth_means = []
    truth_hists = []
    preds_fss = []
    preds_brier_1 = []
    preds_brier_5 = []
    preds_brier_10 = []
    
    t.tic()
    num_workers = mp.cpu_count()
    print("num_workers:", num_workers)
    pool = Pool(processes=num_workers)
    t.toc('Setting up the pool took')
    
    print(f"Total batches: {len(dl_test)}")
    def log_result(result):
        for res in result:
            crps.append(res[0])
            max_pool_crps.append(res[1])
            avg_pool_crps.append(res[2])
            rmse.append(res[3])
            rhist.append(res[4])
            rels_1.append(res[5])
            rels_4.append(res[6])
            preds_brier_1.append(res[7])
            preds_brier_5.append(res[8])
            preds_brier_10.append(res[9])
            
        print("batch complete")
        print(f"current len of crps {len(crps)}")
            
    for batch_idx, (x,y) in enumerate(dl_test):
        t.tic()
        x = x.to(device)
        preds = []
        for i in range(nens):
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3]).to(device)
            try: 
                pred, _ = gen(x, noise)
            except:
                pred = gen(x, noise)  
            preds.append(pred.detach().to('cpu').numpy().squeeze())
        preds = np.array(preds)
        truth = y.numpy().squeeze(1)
        
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        preds = xr.DataArray(
                preds,
                dims=['member', 'sample', 'lat', 'lon'],
                name='tp'
            )

        truth = truth * (ds_max - ds_min) + ds_min

        preds = preds * (ds_max - ds_min) + ds_min

        if tp_log:
            truth = log_retrans(truth, tp_log)
            preds = log_retrans(preds, tp_log)
        
        mean_fss = fss(preds.transpose('sample', 'member', 'lat', 'lon'),truth, threshold = 4, window=25, device=device)
        
        preds_fss.append(mean_fss)
        
        eps = 1e-6
        bin_edges = [-eps] + np.linspace(eps, log_retrans(ds_max, tp_log)+eps, 51).tolist()
        pred_means.append(np.mean(preds.sel(member=0)))
        pred_hists.append(np.histogram(preds.sel(member=0), bins = bin_edges, density=False)[0])
        truth_means.append(np.mean(truth))
        truth_hists.append(np.histogram(truth, bins = bin_edges, density=False)[0])
        
        truth_pert = truth + np.random.normal(scale=1e-6, size=truth.shape)
        preds_pert = preds + np.random.normal(scale=1e-6, size=preds.shape) 

        pool.starmap_async(compute_metrics, [(truth, preds, truth_pert, preds_pert, i) for i in range(x.shape[0])], callback=log_result).wait()
        t.toc('batch_took')
        
        

    rels_1 = xr.concat(rels_1, dim = "patch")
    weights_1 = rels_1.samples / rels_1.samples.sum(dim="patch")
    weighted_relative_freq_1 = (weights_1*rels_1.relative_freq).sum(dim="patch")
    samples_1 = rels_1.samples.sum(dim="patch")
    forecast_probs_1 = rels_1.forecast_probability
    
    rels_4 = xr.concat(rels_4, dim = "patch")
    weights_4 = rels_4.samples / rels_4.samples.sum(dim="patch")
    weighted_relative_freq_4 = (weights_4*rels_4.relative_freq).sum(dim="patch")
    samples_4 = rels_4.samples.sum(dim="patch")
    forecast_probs_4 = rels_4.forecast_probability
    
    rhist = [sum([h[i] for h in rhist]) for i in range(nens+1)]
    
    pred_hists = (np.sum(np.array(pred_hists), axis=0), bin_edges)
    truth_hists = (np.sum(np.array(truth_hists), axis=0), bin_edges)
    
    print(f"total in pres hist {np.sum(pred_hists[0])}, total in true hist {np.sum(truth_hists[0])}")
    
    metrics = {"crps": np.mean(crps), 
               "max_pool_crps": np.mean(max_pool_crps), 
               "avg_pool_crps": np.mean(avg_pool_crps),
               "rankhist": rhist, 
               "reliability_1": (weighted_relative_freq_1, forecast_probs_1, samples_1), 
               "reliability_4": (weighted_relative_freq_4, forecast_probs_4, samples_4), 
               "rmse": np.mean(rmse), 
               "true_mean": np.mean(truth_means),
               "preds_mean": np.mean(pred_means), 
               "true_hist": truth_hists,
               "preds_hist": pred_hists, 
               "fss": np.mean(preds_fss),
                "preds_brier_1" : np.mean(preds_brier_1),
                "preds_brier_5" : np.mean(preds_brier_5),
                "preds_brier_10": np.mean(preds_brier_10)
              }
    
    
    return metrics




def gen_patch_eval(gen, dl_test, nens, ds_min, ds_max, tp_log, device):
    """
    gen: generator, which takes (forecast, noise) as arguments
    dl_test: dataloader
    ds_min and ds_max: the min and max values for unscaling
    tp_log: for undoing the log scaling
    """
    t = TicToc()
    crps = []
    rmse = []
    max_pool_crps = []
    avg_pool_crps = []
    rhist = xr.DataArray(data = np.zeros(nens+1), dims = "rank")
    rels = []
    for batch_idx, (x,y) in enumerate(dl_test):
        print(f"batch {batch_idx} out of {len(dl_test)}")
        x = x.to(device)
        preds = []
        for i in range(nens):
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3]).to(device)
            pred = gen(x, noise).detach().to('cpu').numpy().squeeze()
            preds.append(pred)
        preds = np.array(preds)
        truth = y.numpy().squeeze(1)
        truth = xr.DataArray(
                truth,
                dims=['sample','lat', 'lon'],
                name='tp'
            )
        preds = xr.DataArray(
                preds,
                dims=['member', 'sample', 'lat', 'lon'],
                name='tp'
            )

        truth = truth * (ds_max - ds_min) + ds_min

        preds = preds * (ds_max - ds_min) + ds_min
    
        if tp_log:
            truth = log_retrans(truth, tp_log)
            preds = log_retrans(preds, tp_log)
        
        truth_pert = truth + np.random.normal(scale=1e-6, size=truth.shape)
        preds_pert = preds + np.random.normal(scale=1e-6, size=preds.shape)

        
        t.tic()
        for sample in range(x.shape[0]):

            sample_crps = xs.crps_ensemble(truth.sel(sample=sample), preds.sel(sample=sample)).values
            truth_course = truth.coarsen(lat=4, lon=4)
            preds_course = preds.coarsen(lat=4, lon=4)
            sample_max_pool_crps = xs.crps_ensemble(truth_course.max().sel(sample=sample), preds_course.max().sel(sample=sample)).values
            sample_avg_pool_crps = xs.crps_ensemble(truth_course.mean().sel(sample=sample), preds_course.mean().sel(sample=sample)).values
            crps.append(sample_crps)
            max_pool_crps.append(sample_max_pool_crps)
            avg_pool_crps.append(sample_avg_pool_crps)
            
#             t.toc('crps took', restart=True)
            
            sample_rmse = xs.rmse(preds.sel(sample=sample).mean('member'), truth.sel(sample=sample), dim=['lat', 'lon']).values
            rmse.append(sample_rmse)
            
#             t.tic()
            rhist += xs.rank_histogram(truth_pert.sel(sample=sample), preds_pert.sel(sample=sample)).values
#             t.toc('rank histogram took', restart=True)
            
#             t.tic()
            rel = xs.reliability(truth.sel(sample=sample)>1,(preds.sel(sample=sample)>1).mean('member'))
            rel = xr.where(np.isnan(rel), 0, rel)
            rel['relative_freq'] = rel
            rels.append(rel)
#             t.toc('reliability took', restart=True)
            
        t.toc('metrics took', restart=True)

        
    rels = xr.concat(rels, dim = "patch")
    weights = rels.samples / rels.samples.sum(dim="patch")
    weighted_relative_freq = (weights*rels.relative_freq).sum(dim="patch")
    samples = rels.samples.sum(dim="patch")
    forecast_probs = rels.forecast_probability
    
    return np.mean(crps), np.mean(max_pool_crps), np.mean(avg_pool_crps), rhist, (weighted_relative_freq, forecast_probs, samples), np.mean(rmse)

def single_full_test_prediction(gen, ds_test, device):
    # Get predictions for full field
    
    preds = []
    for t in tqdm.tqdm(range(len(ds_test.tigge.valid_time))):
        x, y = ds_test.return_full_array(t)
        x = torch.FloatTensor(x).unsqueeze(0).to(device)
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3]).to(device)
        pred = gen(x, noise).to('cpu').detach().numpy().squeeze()
        preds.append(pred)
    preds = np.array(preds)
    
    # Unscale
    preds = preds * (ds_test.maxs.tp.values - ds_test.mins.tp.values) + ds_test.mins.tp.values
    
    # Un-log
    if ds_test.tp_log:
        preds = log_retrans(preds, ds_test.tp_log)
    
    # Convert to xarray
    preds = xr.DataArray(
        preds,
        dims=['valid_time', 'lat', 'lon'],
        coords={
            'valid_time': ds_test.tigge.valid_time,
            'lat': ds_test.mrms.lat.isel(
                lat=slice(ds_test.pad_mrms, ds_test.pad_mrms+preds.shape[1])
            ),
            'lon': ds_test.mrms.lon.isel(
                lon=slice(ds_test.pad_mrms, ds_test.pad_mrms+preds.shape[2])
            )
        },
        name='tp'
    )
    return preds
    
    
def ensemble_full_test_predictions(gen, ds_test, nens, device):
    """Wrapper to create ensemble"""
    preds = [single_full_test_prediction(gen, ds_test, device) for _ in range(nens)]
    return xr.concat(preds, 'member')   

def make_eval_mask():
    # mask ground truth data
    rq = xr.open_dataarray('/datadrive_ssd/mrms/4km/RadarQuality.nc')
    eval_mask = rq>-1
    fn = "/datadrive_ssd/mrms/4km/RadarOnly_QPE_06H/RadarOnly_QPE_06H_00.00_20180101-000000.nc"
    ds = xr.open_dataset(fn)
    assert eval_mask.lat.shape ==ds.lat.shape
    eval_mask['lat'] = ds.lat 
    assert eval_mask.lon.shape ==ds.lon.shape
    eval_mask['lon'] = ds.lon
    return eval_mask

def get_full_masked_mrms(gen, ds_test, device):
    #get pred to align lat and lon
    preds = single_full_test_prediction(gen, ds_test,  device);

    eval_mask = make_eval_mask()
    
    mrms = ds_test.mrms.sel(lat=preds.lat.values, 
                            lon=preds.lon.values).rename(
                            {'time': 'valid_time'}) * ds_test.maxs.tp.values
    if ds_test.tp_log:
        mrms = log_retrans(mrms, ds_test.tp_log)
    
    mrms = mrms.where(eval_mask)
    
    return mrms

def gen_full_eval(gen, ds_test, mrms, nens, device):
    preds = ensemble_full_test_predictions(gen, ds_test, nens, device)  
#     print(preds)
    crps = xs.crps_ensemble(mrms, preds).values
    rmse = xs.rmse(preds.mean('member'), mrms, dim=['lat', 'lon', 'valid_time'], skipna=True).values
    
    return preds, crps, rmse


def interpolation_full_baseline(ds_test, mrms):
    tigge = ds_test.tigge.isel(variable=0) * ds_test.maxs.tp.values
    tigge = log_retrans(tigge, ds_test.tp_log)
    #interpolate
    interp = tigge.interp_like(mrms, method='linear')   
    #calculate error
    rmse = xs.rmse(interp, mrms, dim=['lat', 'lon', 'valid_time'], skipna=True).values
    
    return tigge, interp, rmse

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

def _my_f1_score(obs,fcst, thresholds, **kws):
    """ wraps scikit-learn f1-score computation to fit the needs of the apply-ufunc seting.
    obs: 2d np.array of observation data 
    fcst: 2d np.array of forecast data 
    thresholds: list of precipitation thresholds to apply. Computes f1-score for each threshold. 

    Returns np.array of f1-scores, with each value belonging to a different threshold
    
    """ 
    assert obs.shape==fcst.shape,'Shapes of obs and fcst do not match.'
    f1_scores = [f1_score((obs>threshold).ravel(), (fcst>threshold).ravel(),**kws) for threshold in thresholds]
    
    return np.array(f1_scores)



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
    
    #if y_G:
    y_G.plot(ax=axs[1], **settings)
    axs[1].set_title('downscaled')
    
    y_b.plot(ax=axs[2], **settings)
    axs[2].set_title('baseline')
    
    y.plot(ax=axs[3], **settings)
    axs[3].set_title('MRMS')
    
    return fig, axs

def get_hrrr_mask(year = '2020', fdir = '/datadrive/hrrr/4km/'):
    """ Function to get the hrrr mask from interpolation. 
    Not very elegantly coded: we compute the yearly maximum
    at each grid point. Assuming that every valid grid point rains, 
    the mask is defined to exclude all grid points with zero precip 
    in the whole year. 
    year: year as str to be used 
    fdir: directory where to save the resulting data
    """
    try: 
        fn_mask = fdir+year+'_hrrr_interpolation_mask.nc'
        return xr.open_dataarray(fn_mask) 
    except:
        fn = '/datadrive/hrrr/4km/total_precipitation/'+year+'*.nc'
        hrrr_ds = xr.open_mfdataset(fn)
        hrrr_ds= hrrr_ds.tp.diff('lead_time').sel(lead_time =np.timedelta64(12, 'h'))
        hrrr_ds['valid_time'] = hrrr_ds.init_time + hrrr_ds.lead_time
        hrrr_ds= hrrr_ds.swap_dims({'init_time': 'valid_time'})
        mask = (hrrr_ds.max('valid_time')>0)
        
        process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
        git_head_hash = process.communicate()[0].strip()
        mask.attrs['git_hash_at_creation'] = str(git_head_hash)
        mask.attrs['year']=year
        mask.attrs['date_of_computation'] = date.today().strftime("%d/%m/%Y")

        # I don't have writing permission
        #mask.to_netcdf(fdir+year+'_hrrr_interpolation_mask.nc')
        return mask.compute() 

def get_eval_mask(criterion='radarquality', rq_threshold = -1, 
                rq_fn = '/datadrive/mrms/4km/RadarQuality.nc', ds = None): 
    """ Returns a lon-lat mask which area we evaluate on. 
        The radar quality mask is used to determine this. 

        criterion: criterion to apply. ('radarquality', 'hrr+radar) 
        rq_threshold: threshold for 'radarquality'-criterion.   
                    -1 covers everything with radar availability. 
                    Larger thresholds require higher quality. 
        ds: If patchareas criterion, the tiggemrmsm object is required 
                    to get the radarmask for the patches.

        Returns: boolean xr-dataarray, with same lon-lat dimensions 
            as the radar data. 
    """
    # TODO: consider time dependece of radarmask! (do we need to inlcude that?)
    if criterion in ['radarquality', 'hrr+radar']: # use rq>rq-threshold as criterion 
        rq = xr.open_dataarray(rq_fn)
        eval_mask = rq>rq_threshold

        # hardcode: get proper lon-lat values for rq-mask. Somehow weird!
        fn = "/datadrive/mrms_old/4km/RadarOnly_QPE_06H/MRMS_RadarOnly_QPE_06H_00.00_20201001-000000.nc"
        ds = xr.open_dataset(fn)
        assert eval_mask.lat.shape ==ds.lat.shape
        eval_mask['lat'] = ds.lat 
        assert eval_mask.lon.shape ==ds.lon.shape
        eval_mask['lon'] = ds.lon
    
    if criterion =='hrr+radar': 
        hrrr_mask = get_hrrr_mask(year = '2020', fdir = '/datadrive/')
        # somehow lon lats are not absolutely identical! 
        assert hrrr_mask.lat.shape ==ds.lat.shape
        hrrr_mask['lat'] = ds.lat 
        assert hrrr_mask.lon.shape ==ds.lon.shape
        hrrr_mask['lon'] = ds.lon

        eval_mask = eval_mask * hrrr_mask 

    return eval_mask


def get_baseline(X, y, kind = 'interpol', 
                 X_lon=None, X_lat=None, y_lon = None, y_lat =None, 
                 HRRR_fdir = '/datadrive/hrrr/4km/total_precipitation/' ): 
    """ Function computes baseline, i.e. interpolates X onto the grid of y. 
    
    This is probably overkill for now, but might be handy later on when we have different baselines

    1. If X and y are given as numpy arrays, transform to xarray 
    2. Apply interpolation
    
    X: Tigge dataset, or sample. Can be xarray format or numpy
    y: Target radar dataset or radar sample corresponding to X. Can be xarray format or numpy 
    kind: kind of baseline to use: ['interpol', 'HRRR']
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
    if kind =='interpol': 
        y_baseline = X.interp_like(y, kwargs = dict(fill_value='extrapolate')) 
        # fill_value: for scipy interoplate, extrapolates values at the boundaries, so no Nans appear! 
    elif kind =='HRRR': # load HRRR data 
        hrrr = xr.open_mfdataset(HRRR_fdir+ '*')
        hrrr= hrrr.tp.diff('lead_time').sel(lead_time = X.lead_time)
        hrrr['valid_time'] = hrrr.init_time + hrrr.lead_time
        hrrr= hrrr.swap_dims({'init_time': 'valid_time'})
        y_baseline = hrrr # This is not tested and probably not yet finished!
        
    assert y_baseline.shape == y.shape, 'y_baseline and y do not have the same size!'
    return y_baseline
    
def compute_eval_metrics(fcst, obs, eval_mask = None, metrics = ['RMSE', 'FSS', 'F1'],
                     fss_scales=[41,61], fss_thresholds = [1., 5.],
                     f1_thresholds = [0.1,1., 5.], f1_kws=dict() ): 
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
    
    
        
    # rechunking necessary for performance 
    fcst = fcst.chunk({'valid_time':1})
    obs = obs.chunk({'valid_time':1}) 
    
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
                       output_core_dims=[['fss_thresholds','fss_scales']], 
                       output_dtypes=[fcst.dtype],
                       dask_gufunc_kwargs = dict(output_sizes= {'fss_scales':len(fss_scales), 'fss_thresholds': len(fss_thresholds)},),
                       vectorize =True, dask='parallelized',
                       kwargs = dict(windows=fss_scales, levels = fss_thresholds))
        fss_da['fss_thresholds'] = np.array(fss_thresholds)
        fss_da['fss_scales'] = np.array(fss_scales)
        #with ProgressBar(minimum=1): 
            #fss_da = fss_da.compute()
        metrics_ds['FSS'] = fss_da

    if 'F1' in metrics: 
        f1_da = xr.apply_ufunc(_my_f1_score, (obs.where(eval_mask)), (fcst.where(eval_mask)), 
                    input_core_dims=[['lat', 'lon'], ['lat', 'lon']], output_dtypes=[fcst.dtype],
                    output_core_dims=[['f1_thresholds']], vectorize=True, dask='parallelized',
                    dask_gufunc_kwargs = dict(output_sizes= {'f1_thresholds': len(f1_thresholds)},),
                    kwargs = dict(thresholds=f1_thresholds, **f1_kws))
        f1_da['f1_thresholds'] = f1_thresholds
        f1_da.name = 'F1-Score'
        metrics_ds['F1-Score'] = f1_da
    
    return metrics_ds


def evaluate_downscaled_fcst(coarse_fcst, downscaled_fcst, obs, baselines = ['interpol', 'HRRR'],  save_to=None, **kws):
    """ xarrays as input 
    Parameter: 
    coarse_fcst (e.g. tigge data): xr.dataarray precipitation
    downscaled_fcst (e.g. gan-generated): xr.dataarray precipitation
    obs (e.g. radar): xr.dataarray precipiatation
    save_to: string, if given, saves eval-metrics to file as specified by string
    """

    # Step 1: data preparation: matching time dimensions, matching lon-lat dimensions
    if type(downscaled_fcst) is not xr.DataArray: # select variable "tp"
        try: downscaled_fcst=downscaled_fcst.tp 
        except: "downscaled_fcst input must be a xr.dataarray." 
    if 'variable' in coarse_fcst.coords: # select tp from coord "variable"
        try: coarse_fcst=coarse_fcst.sel(variable='tp')
        except: "coars_fcst input must be precipitation only." 
        
    obs = obs.sel(lat=downscaled_fcst.lat, lon=downscaled_fcst.lon)
    obs = obs.sel(time = downscaled_fcst.valid_time.values)    
    obs = obs.rename({'time':'valid_time'})  # we need consistent time naming
        
    
        
    # Step 2: compute baselines
    bl_dict = dict()
    for bl in baselines: 
        bl_dict[bl] = get_baseline(coarse_fcst, obs, kind=bl)
    #assert baseline_0.shape == downscaled_fcst.shape, 'baseline and downscaled_fcst have different shapes!'
        

    
    # Step 3: evaluation mask
    eval_mask = get_eval_mask()
    eval_mask = eval_mask.sel(lat=downscaled_fcst.lat, lon=downscaled_fcst.lon)

    # Step 4: compute different metrics
    metrics_list = []
    for bl, fcst in bl_dict.items():
        metrics = compute_eval_metrics(fcst, obs, eval_mask )       
        metrics['fcst_type']  = bl + '_baseline'
        metrics_list.append(metrics)
    
    metrics_dfcst = compute_eval_metrics(downscaled_fcst, obs, eval_mask, **kws)
    metrics_dfcst['fcst_type'] = 'Generator'
    metrics_list.append(metrics_dfcst)
    metrics = xr.concat(metrics_list,dim = "fcst_type")
    metrics
    
    print("Compute metrics:")
    with ProgressBar():
        metrics.load() # execute fss computation  
    
    # Step 5: save metrics to file 
    process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip()
    git_head_hash
    metrics.attrs['git_hash_at_creation'] = git_head_hash
    metrics.attrs['date_of_computation'] = date.today().strftime("%d/%m/%Y")
    if save_to: 
        metrics.to_netcdf(save_to)

    return metrics
    



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
