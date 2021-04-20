from ecmwfapi import ECMWFDataServer
from ecmwfapi.api import APIException
from fire import Fire
import numpy as np
import calendar
import pandas as pd
import os
import xarray as xr

var_dict = {
    'total_precipitation': '228228',
    'total_column_water': '136',
    '2m_temperature': '167',
    'convective_available_potential_energy': '59',
    'convective_inhibition': '228001',
    'u_component_of_wind': '131',
    'v_component_of_wind': '132'
}

cf2pynio = {
    'total_precipitation': 'tp_P11_L1_GGA0_acc',
    'total_column_water': 'tcw_P1_L1_GGA0',
    '2m_temperature': '2t_P1_L103_GGA0',
    'convective_available_potential_energy': 'cape_P1_L1_GGA0',
    'convective_inhibition': 'ci_P1_L1_GGA0',
    'u_component_of_wind': 'u_P1_L100_GGA0',
    'v_component_of_wind': 'v_P1_L100_GGA0'
}

long2short = {
    'total_precipitation': 'tp',
    'total_column_water': 'tcw',
    '2m_temperature': 't2m',
    'convective_available_potential_energy': 'cape',
    'convective_inhibition': 'cin',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v'
}

areas = {
    'CONUS': [[50, 20], [235, 290]]
}



def crop_to_nc(fn, lats, lons, var):
    ds = xr.open_mfdataset(fn, engine='pynio')[cf2pynio[var]].rename(long2short[var])
    ds = ds.rename({
        'initial_time0_hours': 'init_time',
        'forecast_time0': 'lead_time',
        'lat_0': 'lat',
        'lon_0': 'lon',
    })
    if 'ensemble0' in ds.coords:
        ds = ds.rename({'ensemble0': 'member'})
    ds = ds.sel(lat=slice(*lats), lon=slice(*lons))
    fn_nc = fn.rstrip('.grib') + '.nc'
    ds.to_netcdf(fn_nc)
    print('Saved to nc:', fn_nc)


def main(var, start_month, stop_month, dir, ensemble=False, members=50, lead_time=48, dt=6, level=None, check_exists=True,
         lats=[90, -90], lons=[0, 360], area='CONUS', delete_grib=True):
    """
    Downloads TIGGE files in monthly batches.
    """

    server = ECMWFDataServer()
    if area:
        lats, lons = areas[area]

    months = np.arange(start_month, stop_month, dtype='datetime64[M]')
    dir = f'{dir}/{var}{level if level else ""}{f"_ens{members}" if ensemble else ""}'
    os.makedirs(dir, exist_ok=True)

    for month in months:
        try:
            days = calendar.monthrange(pd.to_datetime(month).year, pd.to_datetime(month).month)[1]
            fn = f'{dir}/{month}.grib'
            fn_nc = fn.rstrip('.grib') + '.nc'
            print(fn)
            if check_exists and os.path.exists(fn_nc):
                print(fn_nc, 'exists')
                continue

            request = {
                "class": "ti",
                "dataset": "tigge",
                "date": f"{month}-01/to/{month}-{days}",
                "expver": "prod",
                "levtype": "sfc",
                "origin": "ecmf",
                "param": var_dict[var],
                "step": '/'.join(np.arange(0, lead_time+dt, dt).astype(str)),
                "time": "00:00:00/12:00:00",
                "type": "cf",
                "target": fn,
                # "format": "netcdf"
            }

            if level:
                request['levtype'] = 'pl'
                request['levelist'] = str(level)
            if ensemble:
                request['number'] = '/'.join(np.arange(1, members+1).astype(str))
                request['type'] = 'pf'

            server.retrieve(request)

            crop_to_nc(fn, lats, lons, var)
            if delete_grib:
                os.remove(fn)

        except APIException:
            print(f'Damaged files {month}-01/to/{month}-{days}')
    
if __name__ == '__main__':
    Fire(main)