from ecmwfapi import ECMWFDataServer
from ecmwfapi.api import APIException
from fire import Fire
import xarray as xr
import os
from download_tigge import areas

def main(dir, lats=[90, -90], lons=[0, 360], area='CONUS', delete_grib=True):
    """Download orography and land-sea mask from TIGGE."""
    server = ECMWFDataServer()
    os.makedirs(dir, exist_ok=True)

    server = ECMWFDataServer()
    server.retrieve({
        "class": "ti",
        "dataset": "tigge",
        "date": "2020-12-01/to/2020-12-01",
        "expver": "prod",
        "levtype": "sfc",
        "origin": "ecmf",
        "param": "172/228002",
        "step": "0",
        "time": "00:00:00",
        "type": "cf",
        "target": f"{dir}/constants.grib",
    })

    if area:
        lats, lons = areas[area]

    ds = xr.open_dataset(f"{dir}/constants.grib", engine='pynio')
    ds = ds.rename({
        'lat_0': 'lat',
        'lon_0': 'lon',
        'orog_P1_L1_GGA0': 'orog',
        'lsm_P1_L1_GGA0': 'lsm'
    })
    ds = ds.sel(lat=slice(*lats), lon=slice(*lons))
    ds.to_netcdf(f"{dir}/constants.nc")
    print('Saved to nc:', f"{dir}/constants.nc")

    if delete_grib:
        os.remove(f"{dir}/constants.grib")



if __name__ == '__main__':
    Fire(main)