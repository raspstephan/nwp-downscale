import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import pandas as pd


class TiggeMRMSDataset(Dataset):
    """PyTorch Dataset for TIGGE MRMS pairing."""
    def __init__(self, tigge_dir, tigge_vars, mrms_dir, lead_time=12, patch_size=512, rq_fn=None, 
                 const_fn=None, const_vars=None, val_days=None, split=None, scale=True,
                 mins=None, maxs=None):
        """
        tigge_dir: Path to TIGGE data without variable name
        tigge_vars: List of TIGGE variables
        mrms_dir: Path to MRMS data including variable name
        lead_time: Lead time of TIGGE data. 12h means 6-12h accumulation
        patch_size: Size of patch in km
        rq_fn: Path to radar quality file
        const_fn: Path to constants file
        cons_vars: Variables to use from constants file
        val_days: Use first N days of each month for validation
        split: Either "train" or "valid"
        scale: Do min-max scaling
        mins: Dataset of mins for TIGGE vars. Computed if not given.
        maxs: Same for maxs
        """
        self.lead_time = lead_time
        self.patch_size = patch_size
        self.val_days = val_days
        self.split= split
        
        # Open datasets
        self.tigge = xr.merge([
            xr.open_mfdataset(f'{tigge_dir}/{v}/*.nc') for v in tigge_vars
        ])  # Merge all TIGGE variables
        self.tigge['tp'] = self.tigge.tp.diff('lead_time')   # Need to take diff to get 6h accumulation
        self.tigge = self.tigge.sel(lead_time=np.timedelta64(lead_time, 'h'))
        self.mrms = xr.open_mfdataset(f'{mrms_dir}/*.nc').tp
        # Make sure there are no negative values
        self.tigge['tp'] = self.tigge['tp'].where(self.tigge['tp'] >= 0, 0)  
        self.mrms = self.mrms.where(self.mrms >= 0, 0)
        self._crop_times()   # Only take times that overlap and (potentially) do train/val split
        self.tigge.load(); self.mrms.load()   # Load datasets into RAM
        if scale:   # Apply min-max scaling
            self._scale(mins, maxs)
        self.tigge = self.tigge.to_array()   # Doing this here saves time
         
        self.tigge_km = 32   # Currently hard-coded
        self.mrms_km = 4
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
    
    def _scale(self, mins, maxs):
        """Apply min-max scaling. Use same scaling for tp in TIGGE and MRMS."""
        self.mins = mins or self.tigge.min()   # Use min/max if provided, otherwise compute
        self.maxs = maxs or self.tigge.max()
        self.tigge = (self.tigge - self.mins) / (self.maxs - self.mins)
        self.mrms = (self.mrms - self.mins.tp) / (self.maxs.tp - self.mins.tp)
        
    def _crop_times(self):
        """Crop TIGGE and MRMS arrays to where they overlap"""
        # Make TIGGE file have valid_time as dimension
        valid_time = self.tigge.init_time + self.tigge.lead_time
        self.tigge.coords['valid_time'] = valid_time
        self.tigge = self.tigge.swap_dims({'init_time': 'valid_time'})

        # Compute intersect
        self.overlap_times = np.intersect1d(self.mrms.time, self.tigge.valid_time)

        if self.val_days: # Split into traina dn valid based on day of month
            dt = pd.to_datetime(self.overlap_times)
            self.overlap_times = self.overlap_times[
                dt.day <= self.val_days if self.split == 'valid' else dt.day > self.val_days
            ]
        
        # Apply selection
        self.mrms = self.mrms.sel(time=self.overlap_times)
        self.tigge = self.tigge.sel(valid_time=self.overlap_times)
        
    def _setup_indices(self):
        """Create a list of indices containing (time, lat_idx, lon_idx). _idx is the patch index."""
        nlat = len(self.tigge.lat) // self.patch_tigge
        nlon = len(self.tigge.lon) // self.patch_tigge
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
        self.rqmask = rq.coarsen(lat=self.patch_mrms, lon=self.patch_mrms, boundary='trim').min() >= 0
        
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, idx):
        """Return individual sample. idx is the sample id, i.e. the index of self.idxs."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        time_idx, lat_idx, lon_idx = self.idxs[idx]

        # Get features for given time and patch
        X = self.tigge.isel(
            valid_time=time_idx,
            lat=slice(lat_idx * self.patch_tigge, (lat_idx+1) * self.patch_tigge),
            lon=slice(lon_idx * self.patch_tigge, (lon_idx+1) * self.patch_tigge)
        ).values
        if hasattr(self, 'const'):  # Add constants
            X = self._add_const(X, lat_idx, lon_idx)

        # Get targets
        y = self.mrms.isel(
            time=time_idx,
            lat=slice(lat_idx * self.patch_mrms, (lat_idx+1) * self.patch_mrms),
            lon=slice(lon_idx * self.patch_mrms, (lon_idx+1) * self.patch_mrms)
        ).values[None]  # Add dimension for channel
        return X, y   # [vars, patch, patch]
    
    def _add_const(self, X, lat_idx, lon_idx):
        """Add constants to X"""
        Xs = [X]
        Xs.append(self.const[self.const_vars].to_array().isel(
            lat=slice(lat_idx * self.patch_tigge, (lat_idx+1) * self.patch_tigge),
            lon=slice(lon_idx * self.patch_tigge, (lon_idx+1) * self.patch_tigge)
        ))
        return np.concatenate(Xs)
    
    
    def compute_weights(self, bins=np.append(np.arange(0, 0.1, 0.01), 10)):
        """
        Compute sampling weights for each sample. Weights are computed so that 
        each bin is samples with the same frequency.
        This can then be used in torch.utils.data.WeightedRandomSampler, for example.
        """
        # Get the mean precipitation from MRMS for each sample
        mean_precip = []
        for idx in range(len(self.idxs)):
            X, y = self[idx]
            mean_precip.append(y.mean())
        # Compute histogram
        bin_weight = np.histogram(mean_precip, bins=bins)[0]
        # Weight for each bin is simply the inverse frequency.
        bin_weight = 1 / np.maximum(bin_weight, 1)
        # Get weight for each sample
        bin_idxs = np.digitize(mean_precip, bins) - 1
        weights = bin_weight[bin_idxs]
        return weights