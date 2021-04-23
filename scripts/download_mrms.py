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
    
def gunzip_file(gz_fn, out_fn=None):
    if not out_fn: out_fn = gz_fn[:-3]
    with gzip.open(gz_fn, 'rb') as f_in:
        with open(out_fn, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return out_fn
                 

def download_mrms_from_server(year, month, day, hour, path):
    fn = f'RadarOnly_QPE_01H_00.00_{year}{month}{day}-{hour}0000.grib2.gz'
    url = f"https://mtarchive.geol.iastate.edu/{year}/{month}/{day}/mrms/ncep/RadarOnly_QPE_01H/" + fn
    gz_fn = f'{path}/{fn}'
    urllib.request.urlretrieve(url, gz_fn)
    return gz_fn


def download_loop(start_date, stop_date, tmp_path, save_path, delete_grib=True, aggregate=True,
                  check_exists=True
):
    dates = pd.DatetimeIndex(np.arange(start_date, stop_date, dtype='datetime64[h]'))
    for d in tqdm(dates):
        if aggregate:
            # Check if 6 hour file exists
            next_6 = d.hour % 6
            if next_6 == 0: next_6 = 6
            next_6 = d + np.timedelta64(6 - (d.hour % 6), 'h')
            agg_fn = save_path + '/RadarOnly_QPE_06H/' + f'RadarOnly_QPE_06H_00.00_{next_6.year}{str(next_6.month).zfill(2)}{str(next_6.day).zfill(2)}-{str(next_6.hour).zfill(2)}0000.nc'
            print(agg_fn)
            if os.path.exists(agg_fn):
                print('Exists:', agg_fn)
                continue

        month = str(d.month).zfill(2)
        day = str(d.day).zfill(2)
        hour = str(d.hour).zfill(2)
        nc_fn = save_path + '/RadarOnly_QPE_01H/' + f'RadarOnly_QPE_01H_00.00_{d.year}{month}{day}-{hour}0000.nc'
        if not (check_exists and os.path.exists(nc_fn)):
            print(d)
            try:
                gz_fn = download_mrms_from_server(d.year, month, day, hour, tmp_path)
                grib_fn = gunzip_file(gz_fn)
                ds = xr.open_dataset(grib_fn, engine='cfgrib')
                ds = ds.rename({
                    'paramId_0': 'tp',
                    'latitude': 'lat',
                    'longitude': 'lon',
                }).drop(['valid_time', 'heightAboveSea', 'step']).expand_dims(dim='time')
                ds.to_netcdf(nc_fn)
                ds.close()
                if delete_grib:
                    os.remove(gz_fn)
                    os.remove(grib_fn)
            except (HTTPError, gzip.BadGzipFile):
                print('Missing')
        else:
            print('Exists:', nc_fn)
        
        if aggregate:
            if d.hour % 6 == 0:
                agg_fn = save_path + '/RadarOnly_QPE_06H/' + f'RadarOnly_QPE_06H_00.00_{d.year}{month}{day}-{hour}0000.nc'
                print(agg_fn)
                if not (check_exists and os.path.exists(agg_fn)):
                    try:
                        nc_fns = []
                        for h in range(-5, 1):
                            dd = d + np.timedelta64(h, 'h')
                            month = str(dd.month).zfill(2)
                            day = str(dd.day).zfill(2)
                            hour = str(dd.hour).zfill(2)
                            nc_fns.append(save_path + '/RadarOnly_QPE_01H/' + f'RadarOnly_QPE_01H_00.00_{dd.year}{month}{day}-{hour}0000.nc')
                        print(nc_fns)
                        ds = xr.open_mfdataset(nc_fns).load()
                        ds = ds.rolling(time=6).sum().dropna('time')
                        ds.to_netcdf(agg_fn)
                        ds.close()
                    except FileNotFoundError:
                        print('Not all 1H files available', agg_fn)
                    for nc_fn in nc_fns:
                        if os.path.exists(nc_fn): 
                            print('Delete:', nc_fn)
                            os.remove(nc_fn)
                else:
                    print('Exists:', agg_fn)

if __name__ == '__main__':
    Fire(download_loop)