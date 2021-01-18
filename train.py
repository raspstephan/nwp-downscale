from src.models import *
from src.dataloader import *
from src.utils import tqdm, device
from configargparse import ArgParser
import torch
from git import Repo
from datetime import datetime
import os

def train(
    tigge_dir=None,
    tigge_vars=None,
    mrms_dir=None,
    const_fn=None,
    const_vars=None,
    rq_fn=None,
    val_days=None,
    batch_size=None,
    learning_rate=None,
    epochs=None,
    save_dir=None,
    exp_id=None,
    nres=None,
    nf=None,
    relu_out=None
    ):

    # Allocate train and valid datasets
    ds_train = TiggeMRMSDataset(
        tigge_dir=tigge_dir,
        tigge_vars=tigge_vars,
        mrms_dir=mrms_dir,
        rq_fn=rq_fn,
        const_fn=const_fn,
        const_vars=const_vars,
        val_days=val_days,
        split='train'
    )
    ds_valid = TiggeMRMSDataset(
        tigge_dir=tigge_dir,
        tigge_vars=tigge_vars,
        mrms_dir=mrms_dir,
        rq_fn=rq_fn,
        const_fn=const_fn,
        const_vars=const_vars,
        val_days=val_days,
        split='valid',
        mins=ds_train.mins,
        maxs=ds_train.maxs
    )
    print('Training samples:', len(ds_train))
    print('Validation samples:', len(ds_valid))

    # Create dataloaders with weighted random sampling
    sampler_train = torch.utils.data.WeightedRandomSampler(
        ds_train.compute_weights(), len(ds_train)
    )
    sampler_valid = torch.utils.data.WeightedRandomSampler(
        ds_valid.compute_weights(), len(ds_valid)
    )

    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, sampler=sampler_train
    )
    dl_valid = torch.utils.data.DataLoader(
        ds_valid, batch_size=batch_size, sampler=sampler_valid
    )

    # Create network
    print('Device:', device)
    model = Generator(
        nres=nres,
        nf_in=ds_train.input_vars,
        nf=nf,
        relu_out=relu_out
    ).to(device)

    # Setup trainer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        dl_train,
        dl_valid
    )

    # Train model
    trainer.fit(epochs=epochs)

    # Save model and valid predictions
    if save_dir:
        save_path = f'{save_dir}/{exp_id}.pt'
        print('Saving model as:', save_path)
        torch.save(model, save_path)
        
        save_path = f'{save_dir}/{exp_id}_train.nc'
        print('Saving prediction as:', save_path)
        preds = create_valid_predictions(model, ds_train)
        preds.to_netcdf(save_path)

        save_path = f'{save_dir}/{exp_id}_valid.nc'
        print('Saving prediction as:', save_path)
        preds = create_valid_predictions(model, ds_valid)
        preds.to_netcdf(save_path)

        time_stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        cwd = os.getcwd()
        git_hash = str(Repo(cwd).active_branch.commit)
        with open(f'{save_dir}/{exp_id}.log', 'w+') as f:
            f.write(time_stamp + '\n')
            f.write(git_hash)



if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('-c', is_config_file=True, 
        help='Config file path'
    )
    p.add_argument('--tigge_dir', type=str, required=True, 
        help='Path to TIGGE data'
    )
    p.add_argument('--tigge_vars', type=str, required=True, nargs='+',
        help='Tigge variables'
    )
    p.add_argument('--mrms_dir', type=str, required=True, 
        help='Path to MRMS data'
    )
    p.add_argument('--const_fn', type=str, default=None, 
        help='Path to constants file'
    )
    p.add_argument('--const_vars', type=str, default=None, nargs='+',
        help='Constant variables'
    )
    p.add_argument('--rq_fn', type=str, default=None, 
        help='Path to radar quality file'
    )
    p.add_argument('--val_days', type=int, default=7, 
        help='First N days of each month used for validation'
    )
    p.add_argument('--nres', type=int, default=3, 
        help='Number of residual blocks before upscaling'
    )
    p.add_argument('--nf', type=int, default=64, 
        help='Number of filters in generator'
    )
    p.add_argument('--relu_out', type=bool, default=False, 
        help='Apply relu after final generator layer.'
    )
    p.add_argument('--batch_size', type=int, default=32, 
        help='Batch size'
    )
    p.add_argument('--learning_rate', type=float, default=1e-4, 
        help='Learning rate'
    )
    p.add_argument('--epochs', type=int, default=10, 
        help='Epochs'
    )
    p.add_argument('--save_dir', type=str, default=None, 
        help='Path to save model. Do not save if None.'
    )
    p.add_argument('--exp_id', type=str, default=None, 
        help='Experiment identifier'
    )
    
    args = vars(p.parse_args())
    args.pop('c')
    train(**args)