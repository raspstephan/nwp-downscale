import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import shutil
import gzip
from zipfile import ZipFile
from fire import Fire
import urllib.request

default_variables = [
    'MultiSensor_QPE_01H_Pass1',
    'MultiSensor_QPE_03H_Pass1',
    'MultiSensor_QPE_06H_Pass1',
    'MultiSensor_QPE_01H_Pass2',
    'MultiSensor_QPE_03H_Pass2',
    'MultiSensor_QPE_06H_Pass2',
    'RadarOnly_QPE_01H',
    'RadarOnly_QPE_03H',
    'RadarOnly_QPE_06H',
]

def download_loop(start_date, stop_date, tmp_path, save_path, variables=None):
    dates = pd.DatetimeIndex(np.arange(start_date, stop_date, dtype='datetime64[h]'))
    for d in tqdm(dates):
        download_and_extract(d.year, d.month, d.day, d.hour, tmp_path, save_path, variables=variables)
        
        
def download_and_extract(year, month, day, hour, tmp_path, save_path, delete=True, zip_fn=None,
                         variables=None):
    zip_fn = zip_fn or download_nrms_from_cache(year, month, day, hour, tmp_path)
    unzip_file(zip_fn, tmp_path)
    variables = variables or default_variables
    for v in variables:
        try:
            move_relevant_file(year, month, day, hour, v, tmp_path, save_path)
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


if __name__ == '__main__':
    Fire(download_loop)