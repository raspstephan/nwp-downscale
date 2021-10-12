import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import pandas as pd
from .utils import tqdm, device, to_categorical
from skimage.measure import block_reduce
from skimage.transform import resize
import os

def log_trans(x, eps):
    """Log transform with given epsilon. Preserves zeros."""
    return np.log(x + eps) - np.log(eps)

def log_retrans(x, eps):
    """Inverse log transform"""
    return np.exp(x + np.log(eps)) - eps


class TiggeMRMSDataset(Dataset):
    """ PyTorch Dataset for TIGGE MRMS pairing.
    
    Returns TiggeMRMSDataset object. 
    
    self.idxs: numpy array with three columns: 
        [first, second, third] column corresponds to
        [time,  lat,    lon  ] index of the patches. 
        The time-idx value corresponds to the time given by the overlap_times array.

            
    """
    def __init__(self, tigge_dir, tigge_vars, mrms_dir, lead_time=12, patch_size=512, rq_fn=None, 
                 const_fn=None, const_vars=None, scale=True, data_period=None, first_days=None,
                 val_days=None, split=None, mins=None, maxs=None, pad_tigge=0, pad_tigge_channel=False, tp_log=None,
                 cat_bins=None, pure_sr_ratio=None, dropna=True, ensemble_mode=None, idx_stride=1,
                 rq_threshold=0, rq_coverage=1
                ):
        """
        tigge_dir: Path to TIGGE data without variable name
        tigge_vars: List of TIGGE variables
        mrms_dir: Path to MRMS data including variable name
        lead_time: Lead time of TIGGE data. 12h means 6-12h accumulation
        patch_size: Size of patch in km
        rq_fn: Path to radar quality file
        const_fn: Path to constants file
        cons_vars: Variables to use from constants file
        data_period: Tuple indicating which time period to load
        val_days: Use first N days of each month for validation
        first_days: Use first X days from each month. Subsampling.
        split: Either "train" or "valid"
        scale: Do min-max scaling
        mins: Dataset of mins for TIGGE vars. Computed if not given.
        maxs: Same for maxs
        pad_tigge: Padding to add to TIGGE patches on each side.
        tp_log: whether to scale the total precipitation logarithmically. 
        """
        self.lead_time = lead_time
        self.patch_size = patch_size
        self.first_days = first_days
        self.val_days = val_days
        self.split= split
        self.cat_bins = cat_bins
        self.pure_sr_ratio = pure_sr_ratio
        self.tp_log = tp_log
        self.ensemble_mode = ensemble_mode
        self.tigge_vars = tigge_vars
        self.idx_stride = idx_stride
        self.rq_threshold = rq_threshold
        self.rq_coverage = rq_coverage
        # Open datasets
        self.tigge = xr.merge([
            xr.open_mfdataset(f'{tigge_dir}/{v}/*.nc') for v in tigge_vars
        ])  # Merge all TIGGE variables
        
               
        if 'convective_inhibition' in tigge_vars:
            print("setting nans in convective_inhibition to 0")
            self.tigge['cin'] = self.tigge['cin'].fillna(0)
        
        self.tigge['tp'] = self.tigge.tp.diff('lead_time')   # Need to take diff to get 6h accumulation
        self.tigge = self.tigge.sel(lead_time=np.timedelta64(lead_time, 'h'))
        self.mrms = xr.open_mfdataset(f'{mrms_dir}/*.nc').tp   # NOTE: Takes around 30s
        # Make sure there are no negative values
        self.tigge['tp'] = self.tigge['tp'].where(self.tigge['tp'] >= 0, 0)  
        self.mrms = self.mrms.where(self.mrms >= 0, 0)
        if data_period:   # NOTE: This will not speed up the open_mfdataset step
            self.tigge = self.tigge.sel(init_time=slice(*data_period))
            self.mrms = self.mrms.sel(time=slice(*data_period))
#         import pdb; pdb.set_trace()
        if dropna:
            self.tigge.load()
            self.tigge = self.tigge.dropna('init_time')
        self._crop_times()   # Only take times that overlap and (potentially) do train/val split
        print('Loading data')

        self.tigge.load(); self.mrms.load()   # Load datasets into RAM
        if tp_log:
            self.tigge['tp'] = log_trans(self.tigge['tp'], tp_log)
            if self.cat_bins is None:   # No log transform for categorical output
                self.mrms = log_trans(self.mrms, tp_log)
        
        if scale:   # Apply min-max scaling
            self._scale(mins, maxs, scale_mrms=True if self.cat_bins is None else False)
        
 
        # Doing this here saves time
        self.tigge = self.tigge.to_array()
        
        self.tigge_km = 32 # ds.tigge.lon.diff('lon').max()*100  # Currently hard-coded 
        self.mrms_km = 4
        self.ratio = self.tigge_km // self.mrms_km
        self.pad_tigge = pad_tigge
        self.pad_tigge_channel = pad_tigge_channel
        self.pad_mrms = self.pad_tigge * self.ratio
        self.patch_tigge = self.patch_size // self.tigge_km
        self.patch_mrms = self.patch_size // self.mrms_km
        
        if rq_fn:  # Create mask of regions with radar coverage
            self._create_rqmask(rq_fn)
        self._setup_indices()   # Set up sample indices (time, lat, lon)
        
        if const_fn:  # Open and scale constants file
            self.const = xr.open_dataset(const_fn).load()
            self.const_vars = const_vars
            if scale:
                const_mins = self.const.min()
                const_maxs = self.const.max()
                self.const = (self.const - const_mins) / (const_maxs - const_mins)
            self.const = self.const[self.const_vars].to_array()
            
        self.var_names = {'total_precipitation': 'tp', 'total_precipitation_ens10':'tp', 'total_column_water':'tcw', 'total_column_water_ens10':'tcw', '2m_temperature':'t2m', 'convective_available_potential_energy':'cape', 'convective_inhibition':'cin'}
    
    def _scale(self, mins, maxs, scale_mrms=True):
        """Apply min-max scaling. Use same scaling for tp in TIGGE and MRMS."""
        
        self.mins = mins or self.tigge.min()   # Use min/max if provided, otherwise compute
        self.maxs = maxs or self.tigge.max()
        
        if self.cat_bins is None:
            self.tigge = (self.tigge - self.mins) / (self.maxs - self.mins)
        if scale_mrms:
            self.mrms = (self.mrms - self.mins.tp) / (self.maxs.tp - self.mins.tp)
                             
        
    def _crop_times(self):
        """Crop TIGGE and MRMS arrays to where they overlap"""
        # Make TIGGE file have valid_time as dimension
        valid_time = self.tigge.init_time + self.tigge.lead_time
        self.tigge.coords['valid_time'] = valid_time
        self.tigge = self.tigge.swap_dims({'init_time': 'valid_time'})

        # Compute intersect
        self.overlap_times = np.intersect1d(self.mrms.time, self.tigge.valid_time)

        if self.first_days: # Only select first X days
            dt = pd.to_datetime(self.overlap_times)
            self.overlap_times = self.overlap_times[dt.day <= self.first_days]
        if self.val_days: # Split into train and valid based on day of month
            dt = pd.to_datetime(self.overlap_times)
            self.overlap_times = self.overlap_times[
                dt.day <= self.val_days if self.split == 'valid' else dt.day > self.val_days
            ]
        
        # Apply selection
        self.mrms = self.mrms.sel(time=self.overlap_times)
        self.tigge = self.tigge.sel(valid_time=self.overlap_times)
        
    def _setup_indices(self):
        """Create a list of indices containing (time, lat_idx, lon_idx). _idx is the TIGGE index, bottom left."""
        # I do not account for differences in the extent of MRMS and TIGGE for now. 
        # Fine for current settings, but could be problematic later.
        # idxs are TIGGE idxs
        idxs = np.mgrid[self.pad_tigge:len(self.tigge.lat) - self.patch_tigge - self.pad_tigge:self.idx_stride, 
                        self.pad_tigge:len(self.tigge.lon) - self.patch_tigge - self.pad_tigge:self.idx_stride]
        idxs = np.array([g.flatten() for g in idxs]).T
        if hasattr(self, 'rqmask'):   # Only take indices where radar coverage is available
            idxs = np.array([r for r in idxs if self.rqmask.isel(lat=r[0]*self.ratio, lon=r[1]*self.ratio)])
        
#         # Account for padding on each side
#         nlat = (len(self.tigge.lat) - 2*self.pad_tigge)  // self.patch_tigge
#         nlon = (len(self.tigge.lon) - 2*self.pad_tigge) // self.patch_tigge
#         # This creates indices with (lat_idx, lon_idx)
#         idxs = np.array([g.flatten() for g in np.mgrid[:nlat, :nlon]]).T
#         if hasattr(self, 'rqmask'):   # Only take indices where radar coverage is available
#             idxs = np.array([r for r in idxs if self.rqmask.isel(lat=r[0], lon=r[1])])

        # Now add time indices
        self.ntime = len(self.overlap_times)
        self.idxs = np.concatenate([
            np.concatenate(
                [np.ones((len(idxs), 1), dtype=int)*i, idxs], 1
            ) for i in range(self.ntime)
        ])
    
    def _create_rqmask(self, rq_fn):
        """Coarsen radar mask to patch and check for full coverage"""
        rq = xr.open_dataarray(rq_fn).load()
        # Account for padding
#         rq = rq.isel(lat=slice(self.pad_mrms, -self.pad_mrms or None), lon=slice(self.pad_mrms, -self.pad_mrms or None))
#         self.rqmask = rq.coarsen(lat=self.patch_mrms, lon=self.patch_mrms, boundary='trim').min() >= 0
        # RQ mask checks for validity of patch indexed by lower left coordinate
        # Note: lat is oriented reversele, so in "real" coords it's the upper left corner
        rq = rq > self.rq_threshold
        self.rqmask = (rq[::-1, ::-1].rolling(
            {'lat': self.patch_mrms}, min_periods=1
        ).mean().rolling(
            {'lon': self.patch_mrms}, min_periods=1
        ).mean() >= self.rq_coverage)[::-1, ::-1]
        
    def __len__(self):
        return len(self.idxs)

    @property
    def input_vars(self):
        v = len(self.tigge.variable)
        if hasattr(self, 'const'):
            v += len(self.const.variable)
        return v
    
    def __getitem__(self, idx, time_idx=None, full_array=False, no_cat=False,  member_idx=None):
        """Return individual sample. idx is the sample id, i.e. the index of self.idxs.
        X: TIGGE sample
        y: corresponding MRMS (radar) sample
        
        **Attention:**
        The self.tigge latitude variable is from ~50-20 degrees, i.e. not from small to large!
        Be careful when transforming indices to actual latitude values! 
        
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()
        time_idx_tmp, lat_idx, lon_idx = self.idxs[idx]
        time_idx = time_idx or time_idx_tmp

        # Get features for given time and patch
        if full_array:  # Return full lats, lons
            lat_slice = slice(0, None)
            lon_slice = slice(0, None)
        else:
#             lat_slice = slice(lat_idx * self.patch_tigge, (lat_idx+1) * self.patch_tigge + self.pad_tigge*2)
#             lon_slice = slice(lon_idx * self.patch_tigge, (lon_idx+1) * self.patch_tigge + self.pad_tigge*2)
            lat_slice = slice(lat_idx-self.pad_tigge, lat_idx + self.patch_tigge + self.pad_tigge)
            lon_slice = slice(lon_idx-self.pad_tigge, lon_idx + self.patch_tigge + self.pad_tigge)
        X = self.tigge.isel(
            valid_time=time_idx,
            lat=lat_slice,
            lon=lon_slice
        )
        
        self.var_stack_idxs = {}
        ind_count = 0
            
        if self.ensemble_mode == 'stack':
            X = X.rename({'variable': 'raw_variable'}).stack(variable = ['raw_variable', 'member']).transpose(
                'variable', 'lat', 'lon')
        if self.ensemble_mode == 'random':
            if member_idx is None:
                member_idx = np.random.choice(self.tigge.member)
            X = X.sel(member=member_idx)
        if self.ensemble_mode == 'stack_by_variable':
            X = xr.concat([X.rename({'variable': 'raw_variable'}).sel(raw_variable=self.var_names[i]).stack(variable=['member']).transpose(
                'variable', 'lat', 'lon').drop('raw_variable') for i in self.tigge_vars if 'ens10' in i] + 
           [X.sel(variable=[self.var_names[i] for i in self.tigge_vars if 'ens10' not in i], member=0).transpose(
                'variable', 'lat', 'lon')], 
          'variable')
            
            for i, var in enumerate(self.tigge_vars):
                if 'ens10' in var:
                    self.var_stack_idxs[self.var_names[var]] = ind_count + np.arange(10)
                    ind_count+=10
            for i, var in enumerate(self.tigge_vars):
                if 'ens10' not in var:
                    self.var_stack_idxs[self.var_names[var]] = ind_count + np.arange(1)
                    ind_count+=1
            
        X = X.values
        if hasattr(self, 'const'):  # Add constants
            X = self._add_const(X, lat_slice, lon_slice)
        

        # Get targets
        if full_array:   # Return corresponding MRMS slice; not used currently
            lat_slice = slice(0, len(self.tigge.lat) * self.ratio)
            lon_slice = slice(0, len(self.tigge.lon) * self.ratio)
        else:
#             lat_slice = slice(
#                 lat_idx * self.patch_mrms + self.pad_mrms, 
#                 (lat_idx+1) * self.patch_mrms + self.pad_mrms
#             )
#             lon_slice = slice(
#                 lon_idx * self.patch_mrms + self.pad_mrms, 
#                 (lon_idx+1) * self.patch_mrms + self.pad_mrms
#             )
            lat_slice = slice(lat_idx * self.ratio, lat_idx * self.ratio + self.patch_mrms)
            lon_slice = slice(lon_idx * self.ratio, lon_idx * self.ratio + self.patch_mrms)
        y = self.mrms.isel(
            time=time_idx,
            lat=lat_slice,
            lon=lon_slice
        ).values[None]  # Add dimension for channel
        if self.pure_sr_ratio:
            X = self._make_sr_X(y)
        if self.cat_bins is not None and not no_cat:
            y = self._categorize(y)
            
        if self.pad_tigge_channel:
            X_crop = X[:,self.pad_tigge:self.pad_tigge + self.patch_tigge, self.pad_tigge:self.pad_tigge + self.patch_tigge]
            X_downsample = resize(X[0:1,:,:], (1, self.patch_tigge, self.patch_tigge))
            X = np.concatenate((X_crop, X_downsample), axis=0)
            self.var_stack_idxs['pad_tigge_channel'] = ind_count + np.arange(1)
            ind_count+=1
            
        return X.astype(np.float32), y.astype(np.float32)   # [vars, patch, patch]
    
    def _add_const(self, X, lat_slice, lon_slice):
        """Add constants to X"""
        Xs = [X]
        Xs.append(self.const.isel(
            lat=lat_slice,
            lon=lon_slice
        ))
        return np.concatenate(Xs)
    
    def _make_sr_X(self, y):
        X = block_reduce(y, (1, self.pure_sr_ratio, self.pure_sr_ratio), np.mean)
        return X


    def _categorize(self, y):
        """Converts continuous output to one-hot-encoded categories"""
        y_shape = y.shape
        y = pd.cut(y.reshape(-1), self.cat_bins, labels=False, include_lowest=True).reshape(y_shape)
        # y = to_categorical(y.squeeze(), num_classes=len(self.cat_bins))
        # y = np.rollaxis(y, 2)
        return y.squeeze().astype('int')
    
    def return_full_array(self, time_idx, member_idx=None):
        """Shortcut to return a full scaled array for a single time index"""
        return self.__getitem__(0, time_idx, full_array=True, member_idx=member_idx)


    def compute_weights(self, min_weight=0.02, max_weight=0.4, threshold=0.025, exp=4, 
                        compute_on_X=False):
        """
        Compute sampling weights for each sample. WEight is simply the mean precip
        value of the target, clipped.
        This can then be used in torch.utils.data.WeightedRandomSampler, for example.
        """
        # Get the mean precipitation from MRMS for each sample
        # mean_precip = []
        # for idx in range(len(self.idxs)):
        #     X, y = self.__getitem__(idx, no_cat=True)
        #     mean_precip.append(y.mean())
        # weights = np.clip(mean_precip, 0.01, 0.1)
        if self.tp_log: 
            threshold = log_trans(threshold, self.tp_log)
            if self.cat_bins is None:
                threshold = threshold / self.maxs.tp
        assert compute_on_X == False, 'Not implemented.'
        
        coverage = (self.mrms > threshold)[:, ::-1, ::-1].rolling(
            lat=self.patch_mrms
        ).mean().rolling(
            lon=self.patch_mrms
        ).mean()[:, ::-1, ::-1]
        
        mrms_idxs = np.copy(self.idxs)
        mrms_idxs[:, 1:] *= self.ratio
        coverage = coverage.values[tuple(mrms_idxs.T)]

#         coverage = []
#         for idx in range(len(self.idxs)):
#             X, y = self.__getitem__(idx, no_cat=True)
#             if compute_on_X: y = X
#             y = y > threshold
#             coverage.append(y.mean())
        scale = max_weight - min_weight
        x = 1-(np.array(coverage)-1)**exp
        weights = min_weight + x * scale


        # # Compute histogram
        # bin_weight = np.histogram(coverage, bins=bins)[0]
        # # Weight for each bin is simply the inverse frequency.
        # bin_weight = 1 / np.maximum(bin_weight, 1)
        # # Get weight for each sample
        # bin_idxs = np.digitize(coverage, bins) - 1
        # weights = bin_weight[bin_idxs]
        return weights #, coverage

    def get_settings(self): 
        """returns key properties as pandas table"""

        options=pd.DataFrame()
        for key, value in iter(vars(self).items()):
            if not key in ['tigge', 'mrms', 'overlap_times', 'mins','maxs', 'rqmask','idxs']:
                options[key] = [value]

        options = options.transpose()
        options.columns.name='TiggeMRMSDataset_Settings:'
        return options

    
    
class TiggeMRMSHREFDataset(TiggeMRMSDataset):
    def __init__(self, href_dir, **kwargs):
        self.href = xr.open_mfdataset(href_dir)
        self.href = self.href.tp.diff('lead_time').sel(lead_time=np.timedelta64(12, 'h'))
        self.href['valid_time'] = self.href.init_time + self.href.lead_time
        self.href = self.href.swap_dims({'init_time': 'valid_time'})
        self.href.load()
        
        super().__init__(**kwargs)
        
    def _create_rqmask(self, rq_fn):
        """Coarsen radar mask to patch and check for full coverage"""
        rq = xr.open_dataarray(rq_fn).load()
        # Account for padding
#         rq = rq.isel(lat=slice(self.pad_mrms, -self.pad_mrms or None), lon=slice(self.pad_mrms, -self.pad_mrms or None))
#         self.rqmask = rq.coarsen(lat=self.patch_mrms, lon=self.patch_mrms, boundary='trim').min() >= 0
        # RQ mask checks for validity of patch indexed by lower left coordinate
        # Note: lat is oriented reversele, so in "real" coords it's the upper left corner
        self.rqmask = (rq[::-1, ::-1].rolling(
            {'lat': self.patch_mrms}, min_periods=1
        ).min().rolling(
            {'lon': self.patch_mrms}, min_periods=1
        ).min() >=0)[::-1, ::-1]
        
        hrefmask = np.isfinite(self.href).mean(('member', 'valid_time')) == 1
        hrefmask = hrefmask.assign_coords(
            {'lat': self.rqmask.lat.values, 'lon': self.rqmask.lon.values}
        )
        
        self.rqmask *= hrefmask
    
    def _crop_times(self):
        """Crop TIGGE and MRMS arrays to where they overlap"""
        # Make TIGGE file have valid_time as dimension
        valid_time = self.tigge.init_time + self.tigge.lead_time
        self.tigge.coords['valid_time'] = valid_time
        self.tigge = self.tigge.swap_dims({'init_time': 'valid_time'})

        # Compute intersect
        self.overlap_times = np.intersect1d(self.mrms.time, self.tigge.valid_time)

        if self.first_days: # Only select first X days
            dt = pd.to_datetime(self.overlap_times)
            self.overlap_times = self.overlap_times[dt.day <= self.first_days]
        if self.val_days: # Split into train and valid based on day of month
            dt = pd.to_datetime(self.overlap_times)
            self.overlap_times = self.overlap_times[
                dt.day <= self.val_days if self.split == 'valid' else dt.day > self.val_days
            ]
            
        self.overlap_times = np.intersect1d(
            self.overlap_times, 
            self.href.valid_time
        )
        
        # Apply selection
        self.mrms = self.mrms.sel(time=self.overlap_times)
        self.tigge = self.tigge.sel(valid_time=self.overlap_times)
        self.href = self.href.sel(valid_time=self.overlap_times)
        
    def __getitem__(self, idx, time_idx=None, full_array=False, no_cat=False,  member_idx=None):
        """Return individual sample. idx is the sample id, i.e. the index of self.idxs.
        X: TIGGE sample
        y: corresponding MRMS (radar) sample
        
        **Attention:**
        The self.tigge latitude variable is from ~50-20 degrees, i.e. not from small to large!
        Be careful when transforming indices to actual latitude values! 
        
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()
        time_idx_tmp, lat_idx, lon_idx = self.idxs[idx]
        time_idx = time_idx or time_idx_tmp

        # Get features for given time and patch
        if full_array:  # Return full lats, lons
            lat_slice = slice(0, None)
            lon_slice = slice(0, None)
        else:
#             lat_slice = slice(lat_idx * self.patch_tigge, (lat_idx+1) * self.patch_tigge + self.pad_tigge*2)
#             lon_slice = slice(lon_idx * self.patch_tigge, (lon_idx+1) * self.patch_tigge + self.pad_tigge*2)
            lat_slice = slice(lat_idx-self.pad_tigge, lat_idx + self.patch_tigge + self.pad_tigge)
            lon_slice = slice(lon_idx-self.pad_tigge, lon_idx + self.patch_tigge + self.pad_tigge)
        X = self.tigge.isel(
            valid_time=time_idx,
            lat=lat_slice,
            lon=lon_slice
        )
        if self.ensemble_mode == 'stack':
            X = X.rename({'variable': 'raw_variable'}).stack(variable = ['raw_variable', 'member']).transpose(
                'variable', 'lat', 'lon')
        if self.ensemble_mode == 'random':
            if member_idx is None:
                member_idx = np.random.choice(self.tigge.member)
            X = X.sel(member=member_idx)
        if self.ensemble_mode == 'stack_by_variable':
            X = xr.concat([X.rename({'variable': 'raw_variable'}).sel(raw_variable=self.var_names[i]).stack(variable=['member']).transpose(
                'variable', 'lat', 'lon').drop('raw_variable') for i in self.tigge_vars if 'ens10' in i] + 
           [X.sel(variable=[self.var_names[i] for i in self.tigge_vars if 'ens10' not in i], member=0).transpose(
                'variable', 'lat', 'lon')], 
          'variable')
            
            self.var_stack_idxs = {}
            ind_count = 0
            for i, var in enumerate(self.tigge_vars):
                if 'ens10' in var:
                    self.var_stack_idxs[self.var_names[var]] = ind_count + np.arange(10)
                    ind_count+=10
            for i, var in enumerate(self.tigge_vars):
                if 'ens10' not in var:
                    self.var_stack_idxs[self.var_names[var]] = ind_count + np.arange(1)
                    ind_count+=1
            
        X = X.values
        if hasattr(self, 'const'):  # Add constants
            X = self._add_const(X, lat_slice, lon_slice)
        

        # Get targets
        if full_array:   # Return corresponding MRMS slice; not used currently
            lat_slice = slice(0, len(self.tigge.lat) * self.ratio)
            lon_slice = slice(0, len(self.tigge.lon) * self.ratio)
        else:
            lat_slice = slice(lat_idx * self.ratio, lat_idx * self.ratio + self.patch_mrms)
            lon_slice = slice(lon_idx * self.ratio, lon_idx * self.ratio + self.patch_mrms)
        y = self.mrms.isel(
            time=time_idx,
            lat=lat_slice,
            lon=lon_slice
        ).values[None]  # Add dimension for channel
        if self.pure_sr_ratio:
            X = self._make_sr_X(y)
        if self.cat_bins is not None and not no_cat:
            y = self._categorize(y)
            
        href = self.href.isel(
            valid_time=time_idx,
            lat=lat_slice,
            lon=lon_slice
        ).values[None]
            
        if self.pad_tigge_channel:
            X_crop = X[:,self.pad_tigge:self.pad_tigge + self.patch_tigge, self.pad_tigge:self.pad_tigge + self.patch_tigge]
            X_downsample = resize(X[0:1,:,:], (1, self.patch_tigge, self.patch_tigge))
            X = np.concatenate((X_crop, X_downsample), axis=0)
            self.var_stack_idxs['pad_tigge_channel'] = ind_count + np.arange(1)
            ind_count+=1
            
        return X.astype(np.float32), y.astype(np.float32), href.astype(np.float32)   # [vars, patch, patch]
        

def create_valid_predictions(model, ds_valid):
    # Get predictions for full field
    preds = []
    for t in tqdm.tqdm(range(len(ds_valid.tigge.valid_time))):
        X, y = ds_valid.return_full_array(t)
        pred = model(torch.FloatTensor(X[None]).to(device)).to('cpu').detach().numpy()[0, 0]
        preds.append(pred)
    preds = np.array(preds)
    
    # Unscale
    preds = preds * (ds_valid.maxs.tp.values - ds_valid.mins.tp.values) + ds_valid.mins.tp.values
    
    # Convert to xarray
    preds = xr.DataArray(
        preds,
        dims=['valid_time', 'lat', 'lon'],
        coords={
            'valid_time': ds_valid.tigge.valid_time,
            'lat': ds_valid.mrms.lat.isel(lat=slice(0, preds.shape[1])),
            'lon': ds_valid.mrms.lon.isel(lon=slice(0, preds.shape[2]))
        },
        name='tp'
    )
    return preds

import os

class TiggeMRMSPatchLoadDataset(Dataset):

    def __init__(self, root_dir, samples_vars = {'tp':1}):
        self.root_dir = root_dir
        self.weights = np.load(self.root_dir+'/weights/weights.npz', allow_pickle=True)['weights']
        self.var_stack_idxs = np.load(self.root_dir+'/configs/var_stack_idxs.npz', allow_pickle=True)['var_stack_idxs']
        self.samples_vars = samples_vars
        
    def __len__(self):
        return len(self.weights)
    
    def __getitem__(self, idx):
        data = np.load(self.root_dir+f'/data/x_{idx}.npz')
        y = data['mrms']
        inds = np.zeros(sum(self.samples_vars.values()), dtype=np.int)
        pointer = 0
        for i, tot in self.samples_vars.items():
            inds[pointer:pointer+tot] = np.random.choice(self.var_stack_idxs.item()[i], size = tot, replace=False)
            pointer+=tot
        x = data['forecast'][inds]
        del data
        return x,y
    
class HREFMRMSPatchLoadDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.weights = np.load(self.root_dir+'/weights/weights.npz', allow_pickle=True)['weights']
        
    def __len__(self):
        return len(self.weights)
    
    def __getitem__(self, idx):
        y = np.load(self.root_dir+f'/data/x_{idx}.npz')['mrms']
        x = np.load(self.root_dir+f'/href/x_{idx}.npz')['href'].squeeze()
        return x,y
    

def save_images(ds, save_dir, split, starting_index):
    path = save_dir + split + '/'
    os.makedirs(path+'data')
    os.makedirs(path+'weights')
    os.makedirs(path+'configs')
    if split == 'test':
        os.makedirs(path+'href')   
    for i in range(len(ds)):
        dpt = ds[i]
        np.savez_compressed(path+f'data/x_{i+starting_index}.npz',forecast = dpt[0], mrms = dpt[1])
        if split == 'test':
            np.savez_compressed(path+f'href/x_{i+starting_index}.npz',href = dpt[2])
    weights = ds.compute_weights()
    np.savez_compressed(path+f'weights/weights.npz', weights = weights)
    np.savez_compressed(path+f'configs/var_stack_idxs.npz', var_stack_idxs = ds.var_stack_idxs)