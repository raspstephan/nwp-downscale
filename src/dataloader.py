import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import pandas as pd
from .utils import tqdm, device, to_categorical
from skimage.measure import block_reduce

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
                 val_days=None, split=None, mins=None, maxs=None, pad_tigge=0, tp_log=None,
                 cat_bins=None, pure_sr_ratio=None, dropna=True, ensemble_mode=None):
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
        
        # Open datasets
        self.tigge = xr.merge([
            xr.open_mfdataset(f'{tigge_dir}/{v}/*.nc') for v in tigge_vars
        ])  # Merge all TIGGE variables
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
            if cat_bins is None:   # No log transform for categorical output
                self.mrms = log_trans(self.mrms, tp_log)
        if scale:   # Apply min-max scaling
            self._scale(mins, maxs, scale_mrms=True if cat_bins is None else False)
        self.tigge = self.tigge.to_array()   # Doing this here saves time
         
        self.tigge_km = 32 # ds.tigge.lon.diff('lon').max()*100  # Currently hard-coded 
        self.mrms_km = 4
        self.ratio = self.tigge_km // self.mrms_km
        self.pad_tigge = pad_tigge
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
    
    def _scale(self, mins, maxs, scale_mrms=True):
        """Apply min-max scaling. Use same scaling for tp in TIGGE and MRMS."""
        # Use min/max if provided, otherwise compute
        self.mins = mins or self.tigge.min().astype('float32')  
        self.maxs = maxs or self.tigge.max().astype('float32')
        if (self.cat_bins is None) and (maxs is None):
            self.maxs['tp'] = self.mrms.max()   # Make sure to take MRMS max for tp
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
        """Create a list of indices containing (time, lat_idx, lon_idx). _idx is the patch index."""
        # Account for padding on each side
        nlat = (len(self.tigge.lat) - 2*self.pad_tigge)  // self.patch_tigge
        nlon = (len(self.tigge.lon) - 2*self.pad_tigge) // self.patch_tigge
        # This creates indices with (lat_idx, lon_idx)
        idxs = np.array([g.flatten() for g in np.mgrid[:nlat, :nlon]]).T
        if hasattr(self, 'rqmask'):   # Only take indices where radar coverage is available
            idxs = np.array([r for r in idxs if self.rqmask.isel(lat=r[0], lon=r[1])])
        # Now add time indices
        self.ntime = len(self.overlap_times)
        self.idxs = np.concatenate([
            np.concatenate(
                [np.ones((len(idxs), 1), dtype=int)*i, idxs], 1
            ) for i in range(self.ntime)
        ])
    
    def _create_rqmask(self, rq_fn):
        """Coarsen radar mask to patch and check for full coverage"""
        rq = xr.open_dataarray(rq_fn)
        # Account for padding
        rq = rq.isel(lat=slice(self.pad_mrms, -self.pad_mrms or None), lon=slice(self.pad_mrms, -self.pad_mrms or None))
        self.rqmask = rq.coarsen(lat=self.patch_mrms, lon=self.patch_mrms, boundary='trim').min() >= 0
        
    def __len__(self):
        return len(self.idxs)

    @property
    def input_vars(self):
        v = len(self.tigge.variable)
        if hasattr(self, 'const'):
            v += len(self.const.variable)
        return v
    
    def __getitem__(self, idx, time_idx=None, full_array=False, no_cat=False):
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
            lat_slice = slice(lat_idx * self.patch_tigge, (lat_idx+1) * self.patch_tigge + self.pad_tigge*2)
            lon_slice = slice(lon_idx * self.patch_tigge, (lon_idx+1) * self.patch_tigge + self.pad_tigge*2)
        X = self.tigge.isel(
            valid_time=time_idx,
            lat=lat_slice,
            lon=lon_slice
        )
        if self.ensemble_mode == 'stack':
            X = X.rename({'variable': 'raw_variable'}).stack(variable = ['raw_variable', 'member']).transpose(
                'variable', 'lat', 'lon')
        if self.ensemble_mode == 'random':
            member_idx = np.random.choice(self.tigge.member)
            X = X.sel(member=member_idx)
        X = X.values
        if hasattr(self, 'const'):  # Add constants
            X = self._add_const(X, lat_slice, lon_slice)
        

        # Get targets
        if full_array:   # Return corresponding MRMS slice; not used currently
            lat_slice = slice(0, len(self.tigge.lat) * self.ratio)
            lon_slice = slice(0, len(self.tigge.lon) * self.ratio)
        else:
            lat_slice = slice(
                lat_idx * self.patch_mrms + self.pad_mrms, 
                (lat_idx+1) * self.patch_mrms + self.pad_mrms
            )
            lon_slice = slice(
                lon_idx * self.patch_mrms + self.pad_mrms, 
                (lon_idx+1) * self.patch_mrms + self.pad_mrms
            )
        y = self.mrms.isel(
            time=time_idx,
            lat=lat_slice,
            lon=lon_slice
        ).values[None]  # Add dimension for channel
        if self.pure_sr_ratio:
            X = self._make_sr_X(y)
        if self.cat_bins is not None and not no_cat:
            y = self._categorize(y)
            
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
    
    def return_full_array(self, time_idx):
        """Shortcut to return a full scaled array for a single time index"""
        return self.__getitem__(0, time_idx, full_array=True)


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

        if self.tp_log: threshold = log_trans(threshold, self.tp_log)

        coverage = []
        for idx in range(len(self.idxs)):
            X, y = self.__getitem__(idx, no_cat=True)
            if compute_on_X: y = X
            y = y > threshold
            coverage.append(y.mean())
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
        return weights

    def get_settings(self): 
        """returns key properties as pandas table"""

        options=pd.DataFrame()
        for key, value in iter(vars(self).items()):
            if not key in ['tigge', 'mrms', 'overlap_times', 'mins','maxs', 'rqmask','idxs']:
                options[key] = [value]

        options = options.transpose()
        options.columns.name='TiggeMRMSDataset_Settings:'
        return options


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


