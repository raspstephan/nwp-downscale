from fire import Fire
import xarray as xr 
import numpy as np
import pandas as pd
from datetime import timedelta
import os
from tqdm import tqdm

def add_zero_lead_time(da, check_exists=True):
    if da.lead_time[0] == np.timedelta64(0):
        return da
    else:
        da = xr.concat([
            (da.isel(lead_time=0)  * 0).assign_coords({
                    'lead_time': da.isel(lead_time=0).lead_time - np.timedelta64(6, 'h'),
                }).assign_coords({
                    'valid_time': da.isel(lead_time=0).valid_time.values - np.timedelta64(6, 'h'),
                }).drop('step'),
            da.drop('step')
        ], dim='lead_time')
        return da


def main(start_date, stop_date, path, check_exists=True):
    
    save_path = f'{path}/href/4km/total_precipitation/'
    os.makedirs(save_path, exist_ok=True)
    
    init_dates = pd.to_datetime(np.arange(
        start_date, stop_date, np.timedelta64(12, 'h'), dtype='datetime64[h]'
    ))
    
    models = ['hiresw_conusarw', 'hiresw_conusnmmb', 'hiresw_conusnssl', 'nam_conusnest', 'hrrr']
    
    for date in tqdm(init_dates):
        
        members = []
        names = []
        lag = 12
        previous_date = date - timedelta(hours=lag)
        current_date_str = f'{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}'
        previous_date_str = f'{previous_date.year}{str(previous_date.month).zfill(2)}{str(previous_date.day).zfill(2)}'
        
        save_fn = f'{save_path}{current_date_str}_{str(date.hour).zfill(2)}.nc'
        if not (check_exists and os.path.exists(save_fn)):
            try:
                for model in models:
                    current = add_zero_lead_time(
                        xr.open_dataarray(
                            f'{path}/{model}/4km/total_precipitation/{current_date_str}_{str(date.hour).zfill(2)}.nc'
                        )
                    )
                    previous = add_zero_lead_time(
                        xr.open_dataarray(
                            f'{path}/{model}/4km/total_precipitation/{previous_date_str}_{str(previous_date.hour).zfill(2)}.nc'
                        )
                    )
                    previous = (previous - previous.sel(lead_time=np.timedelta64(lag, 'h'))).assign_coords({
                        'init_time': previous.init_time + np.timedelta64(lag, 'h')
                    })
                    members.extend([current, previous])
                    names.extend([model, f'{model}-{lag}h'])

                href_ds = xr.concat(
                    members, 
                    dim=xr.DataArray(names, dims='member', name='member'), 
                    coords='minimal',
                    compat='override'
                )

                print('Saving', save_fn)
                href_ds.to_netcdf(save_fn)
            except FileNotFoundError:
                print('Not all files exist')
        else:
            print('Exists:', save_fn)
    

if __name__ == '__main__':
    Fire(main)