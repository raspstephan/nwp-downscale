import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import xarray as xr
import shutil
import gzip
from zipfile import ZipFile
from fire import Fire
import urllib.request
from urllib.error import HTTPError

variables = [
    'MultiSensor_QPE_{}H_Pass1',
    'MultiSensor_QPE_{}H_Pass2',
    'RadarOnly_QPE_{}H',
    'RadarQualityIndex',
]
        
        
def download_and_extract(year, month, day, hour, tmp_path, save_path, dt, delete=True, zip_fn=None, delete_grib=True):
    zip_fn = zip_fn or download_nrms_from_cache(year, month, day, hour, tmp_path)
    unzip_file(zip_fn, tmp_path)
    for v in variables:
        v = v.format(str(dt).zfill(2))
        try:
            grib_fn = move_relevant_file(year, month, day, hour, v, tmp_path, save_path)
            ds = xr.open_dataset(grib_fn, engine='cfgrib')
            ds = ds.rename({
                'paramId_0': 'tp',
                'latitude': 'lat',
                'longitude': 'lon',
            }).drop(['valid_time', 'heightAboveSea', 'step']).expand_dims(dim='time')
            nc_fn = grib_fn.rstrip('.grib2') + '.nc'
            ds.to_netcdf(nc_fn)
            ds.close()
            if delete_grib:
                os.remove(grib_fn)
        except FileNotFoundError:
            print(f'File not found: {v}')
    if delete:
        month = str(month).zfill(2)
        day = str(day).zfill(2)
        hour = str(hour).zfill(2)
        os.remove(zip_fn)
        shutil.rmtree(f'{tmp_path}/{year}{month}{day}{hour}')
        

def move_relevant_file(year, month, day, hour, variable, tmp_path, save_path):
    month = str(month).zfill(2)
    day = str(day).zfill(2)
    hour = str(hour).zfill(2)
    fn = f'MRMS_{variable}_00.00_{year}{month}{day}-{hour}0000.grib2'
    gz_fn = f'{tmp_path}/{year}{month}{day}{hour}/CONUS/{variable}/{fn}.gz'
    os.makedirs(f'{save_path}/{variable}', exist_ok=True)
    save_fn  = f'{save_path}/{variable}/{fn}'
    gunzip_file(gz_fn, save_fn)
    return save_fn

    
def gunzip_file(gz_fn, out_fn=None):
    if not out_fn: out_fn = gz_fn[:-3]
    with gzip.open(gz_fn, 'rb') as f_in:
        with open(out_fn, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
                 

def unzip_file(zip_fn, tmp_path):
    zf = ZipFile(zip_fn, 'r')
    zf.extractall(tmp_path)
    zf.close()
    

def download_nrms_from_cache(year, month, day, hour, path):
    month = str(month).zfill(2)
    day = str(day).zfill(2)
    hour = str(hour).zfill(2)
    fn = f'{year}{month}{day}{hour}.zip'
    url = f"https://mrms.agron.iastate.edu/{year}/{month}/{day}/{fn}"
    zip_fn = f'{path}/{fn}'
    urllib.request.urlretrieve(url, zip_fn)
    return zip_fn


def download_loop(start_date, stop_date, dt, tmp_path, save_path, delete=True, delete_grib=True):
    dates = pd.DatetimeIndex(np.arange(start_date, stop_date, dt, dtype='datetime64[h]'))
    for d in tqdm(dates):
        print(d)
        try:
            download_and_extract(
                d.year, d.month, d.day, d.hour, tmp_path, save_path, dt, delete=delete, delete_grib=delete_grib)
        except HTTPError:
            print('Missing')

if __name__ == '__main__':
    Fire(download_loop)