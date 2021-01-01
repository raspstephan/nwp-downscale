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
        lats=None,
        lons=None
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
    lats = lats or (np.round(ds_in.lat.max()).values, np.round(ds_in.lat.min()).values)
    lons = lons or (np.round(ds_in.lon.min()).values, np.round(ds_in.lon.max()).values)
    grid_out = global_grid.sel(lat=slice(*lats), lon=slice(*lons))

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


def regrid_mrms(ds_in, km, lats=None, lons=None):
    """WARNING: This function contains a lot of hard-coding!"""
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
    lats = lats or (np.round(ds_in.lat.max()).values, np.round(ds_in.lat.min()).values)
    lons = lons or (np.round(ds_in.lon.min()).values, np.round(ds_in.lon.max()).values)
    grid_out = global_grid.sel(lat=slice(*lats), lon=slice(*lons))

    # Coarse grain array. Offsets of 2 make sure array is compatible with 4km grid spacing
    coarse_ds = ds_in.isel(lat=slice(2, None), lon=slice(2, None)).coarsen(lat=km, lon=km, boundary='trim').mean()

    # Round to nearest 2 decimals
    coarse_ds = coarse_ds.assign_coords(
        lat=np.around(coarse_ds.lat.values, 2),
        lon=np.around(coarse_ds.lon.values, 2)
    )

    coarse_ds = coarse_ds.sel(lat=grid_out.lat, lon=grid_out.lon, method='nearest')
    return coarse_ds



def main(var, path, km, check_exists=True, lats=None, lons=None, mrms=False):
    path_in = f'{path}/raw/{var}/'
    files = [p.split('/')[-1] for p in sorted(glob(f'{path_in}/*.nc'))]
    path_out = f'{path}/{km}km/{var}/'
    os.makedirs(path_out, exist_ok=True)

    for f in tqdm(files):
        if check_exists and os.path.exists(path_out + f):
            print(path_out + f, 'exists')
        else:
            ds_in = xr.open_dataset(path_in + f)
            if mrms: 
                ds_out = regrid_mrms(ds_in, km, lats=lats, lons=lons)
            else:
                ds_out = regrid(ds_in, km, lats=lats, lons=lons)
            print('Saving file:', path_out + f)
            ds_out.to_netcdf(path_out + f)
            ds_in.close(); ds_out.close()



if __name__ == '__main__':
    Fire(main)