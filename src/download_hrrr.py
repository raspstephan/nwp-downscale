import tqdm
import os
import urllib.request
from urllib.error import HTTPError
import xarray as xr
import numpy as np
import pandas as pd
from fire import Fire

def main(start_date, stop_date, tmp_path, save_path, delete_grib=True, check_exists=True):

    init_dates = pd.to_datetime(np.arange(
        start_date, stop_date, np.timedelta64(12, 'h'), dtype='datetime64[h]'
        ))
    lead_times = np.arange(0, 36+6, 6)
    for date in tqdm.tqdm(init_dates):
        date_str = f'{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}'
        nc_fn = f'{save_path}/{date_str}_{str(date.hour).zfill(2)}.nc'
        if not (check_exists and os.path.exists(nc_fn)):
            das = []
            tmp_fns = []
            try:
                for l in lead_times:
                        # Download
                        aws_path = f'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date_str}/conus/hrrr.t{str(date.hour).zfill(2)}z.wrfsfcf{str(l).zfill(2)}.grib2'
                        tmp_fn = f'{tmp_path}/{date_str}_{str(date.hour).zfill(2)}_{str(l).zfill(2)}.grib2'
                        urllib.request.urlretrieve(aws_path, tmp_fn)
                        tmp_fns.append(tmp_fn)

                        # Convert
                        var_name = 'APCP_P8_L1_GLC0_acc' + (f'{l}h' if l > 0 else '')
                        da = xr.open_dataset(tmp_fn, engine='pynio')[var_name].rename('tp').rename(
                            {'gridlat_0': 'lat', 'gridlon_0': 'lon', 'ygrid_0': 'y', 'xgrid_0': 'x'})
                        das.append(da)
                
                # Concat and save
                da = xr.concat(das, xr.DataArray(lead_times.astype('timedelta64[h]'), dims='lead_time', name='lead_time'))
                da = da.expand_dims('init_time').assign_coords(init_time=[date])
                # Convert from kg/m^2 to mm
                da = da / 997 * 1000
                da.to_netcdf(nc_fn)
                
            except HTTPError:
                    print('Missing:', aws_path)

            if delete_grib:
                    for tmp_fn in tmp_fns: 
                        os.remove(tmp_fn)
        else:
            print('Exists:', nc_fn)

if __name__ == '__main__':
    Fire(main)