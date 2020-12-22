from fire import Fire
import xarray as xr 
import xesmf as xe
from glob import glob
from tqdm import tqdm
import numpy as np
import os


def regrid(
        ds_in,
        km,
        method='bilinear',
        # reuse_weights=True
):

    ddeg_out = km/100.
    # Rename to ESMF compatible coordinates
    if 'latitude' in ds_in.coords:
        ds_in = ds_in.rename({'latitude': 'lat', 'longitude': 'lon'})

    # Create output grid. To make sure grids stay consistent irrespective of the range, start with a global grid
    global_grid = xr.Dataset(
        {
            'lat': (['lat'], np.arange(90, -90, -ddeg_out)),
            'lon': (['lon'], np.arange(0, 360, ddeg_out)),
        }
    )

    # Crop grid
    lats = slice(np.round(ds_in.lat.max()).values, np.round(ds_in.lat.min()).values)
    lons = slice(np.round(ds_in.lon.min()).values, np.round(ds_in.lon.max()).values)
    grid_out = global_grid.sel(lat=lats, lon=lons)

    # Create regridder
    regridder = xe.Regridder(
        ds_in, grid_out, method, periodic=False
    )

    # # Hack to speed up regridding of large files
    # ds_list = []
    # chunk_size = 500
    # n_chunks = len(ds_in.time) // chunk_size + 1
    # for i in range(n_chunks):
    #     ds_small = ds_in.isel(time=slice(i*chunk_size, (i+1)*chunk_size))
    #     ds_list.append(regridder(ds_small).astype('float32'))
    # ds_out = xr.concat(ds_list, dim='time')

    # # Set attributes since they get lost during regridding
    # for var in ds_out:
    #     ds_out[var].attrs =  ds_in[var].attrs
    # ds_out.attrs.update(ds_in.attrs)

    # # Regrid dataset
    ds_out = regridder(ds_in)
    return ds_out.astype('float32')


def main(var, path, km, check_exists=True):
    path_in = f'{path}/raw/{var}/'
    files = [p.split('/')[-1] for p in glob(f'{path_in}/*.nc')]
    path_out = f'{path}/{km}km/{var}/'
    os.makedirs(path_out, exist_ok=True)

    for f in tqdm(files):
        if check_exists and os.path.exists(path_out + f):
            print(path_out + f, 'exists')
        else:
            ds_in = xr.open_dataset(path_in + f)
            ds_out = regrid(ds_in, km)
            print('Saving file:', path_out + f)
            ds_out.to_netcdf(path_out + f)
            ds_in.close(); ds_out.close()



if __name__ == '__main__':
    Fire(main)