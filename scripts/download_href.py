import tqdm
import os
import urllib.request
from urllib.error import HTTPError
import xarray as xr
import numpy as np
import pandas as pd
from fire import Fire
from gribapi.errors import PrematureEndOfFileError
import pdb
import signal

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    print("URL Timeout")
    raise TimeoutException("URL Timeout")

def main(start_date, stop_date, tmp_path, save_path, delete_grib=True, check_exists=True,
         models=['hiresw_conusarw', 'hiresw_conusnmmb', 'hiresw_conusnssl', 'nam_conusnest'], version=''):
    """
    The output format is cumulative in 6h increments!
    """

    init_dates = pd.to_datetime(np.arange(
        start_date, stop_date, np.timedelta64(12, 'h'), dtype='datetime64[h]'
        ))
    final_lead_times = np.arange(6, 48+6, 6, dtype='timedelta64[h]')
    for model in models:
        print(model)
        if model == 'hrrr':  # definitely cumulative
            lead_times = np.arange(6, 48+6, 6)
            cumulative = True
        else:   # 1h accumulations or not sure
            lead_times = np.arange(1, 48+1, 1)
            cumulative = False
        
        for date in tqdm.tqdm(init_dates):
            date_str = f'{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}'
            nc_path = f'{save_path}/{model}/raw{version}/total_precipitation'
            os.makedirs(nc_path, exist_ok=True)
            nc_fn = f'{nc_path}/{date_str}_{str(date.hour).zfill(2)}.nc'
#             pdb.set_trace()
            if not (check_exists and os.path.exists(nc_fn)):
                das = []
                tmp_fns = []
                actual_lead_times = []

                for l in lead_times:
                    print(l)
                    try:
                        # Download
                        path = f'https://data.nssl.noaa.gov/thredds/fileServer/FRDD/HREF/{date.year}/{date_str}/{model}_{date_str}{str(date.hour).zfill(2)}f{str(l).zfill(3)}.grib2'
                        tmp_fn = f'{tmp_path}/{date_str}_{str(date.hour).zfill(2)}_{str(l).zfill(2)}.grib2'
                        downloaded = False
                        while not downloaded:
                            signal.signal(signal.SIGALRM, handler)
                            signal.alarm(10)
                            try:
                                urllib.request.urlretrieve(path, tmp_fn)
                                downloaded = True
                            except TimeoutException as exc:
                                print(exc)
                        signal.alarm(0)
                                
                        tmp_fns.append(tmp_fn)
                        
                        # Open dataset
                        da = xr.open_dataset(
                            tmp_fn, 
                            engine='cfgrib', 
                            filter_by_keys={'stepType': 'accum', 'typeOfLevel': 'surface'}
                        ).tp.rename({'latitude': 'lat', 'longitude': 'lon'})
                        das.append(da)
                        actual_lead_times.append(l)
                    except KeyError:
                        print('File corrupted')
                    except HTTPError:
                        print('Missing:', path)

                # Concat and save
                if len(das) > 0:
                    try:
                        da = xr.concat(das, xr.DataArray(
                            np.array(actual_lead_times).astype('timedelta64[h]'),
                            dims='lead_time', 
                            name='lead_time'
                        ))
                        da = da.expand_dims('init_time').assign_coords(init_time=[date])
                        
                        if not cumulative:  # Convert to cumulative
                            if (da.diff('lead_time') < -0.05).sum() == 0:  # then it is cumulative
                                print('Is actually cumulative')
                            else:
                                if not len(das) == len(lead_times):
                                    raise ValueError('Need every hour for 1h accumulation models')
                                da = da.cumsum('lead_time')
                            da = da.sel(lead_time=final_lead_times)

                        # Convert from kg/m^2 to mm
                        da = da / 997 * 1000
                        
                        if model == 'nam_conusnest':  # I think 2h accumulations!?
                            da /= 2.
                        
                        da.to_netcdf(nc_fn)
                        da.close()
                        [da.close() for da in das]
                    except ValueError:
                        print('Not every hour file available')

                if delete_grib:
                        for tmp_fn in tmp_fns: 
                            os.remove(tmp_fn)
                            os.remove(tmp_fn + '.923a8.idx')
            else:
                print('Exists:', nc_fn)

if __name__ == '__main__':
    Fire(main)