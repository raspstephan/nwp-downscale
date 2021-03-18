from src.models import Discriminator, Generator
from src.trainer import *
from src.dataloader import *
from src.utils import tqdm, device
from configargparse import ArgParser
import torch
from git import Repo
from datetime import datetime
import os
import dask
import logging
# This is to silence a large chunk warning. I do not know how this affects performance!
dask.config.set({"array.slicing.split_large_chunks": True})

def train(
    tigge_dir=None,
    tigge_vars=None,
    mrms_dir=None,
    const_fn=None,
    const_vars=None,
    rq_fn=None,
    train_period=None, 
    test_period=None,
    first_days=None,
    val_days=None,

    batch_norm = None, 
    spectralnorm=None,  # TODO: implement the four points here 
    use_noise=None, 
    cond_disc = None,
    D_loss = None,
    disc_repeats = None, 
    sigmoid=None,
    
    batch_size=None,
    learning_rate=None,
    epochs=None,
    early_stopping_patience=None,
    restore_best_weights=None,
    save_dir=None,
    exp_id=None,
    nres=None,
    nf=None,
    relu_out=None,
    ):

    # Allocate train and valid datasets
    ds_train = TiggeMRMSDataset(
        tigge_dir=tigge_dir,
        tigge_vars=tigge_vars,
        mrms_dir=mrms_dir,
        rq_fn=rq_fn,
        const_fn=const_fn,
        const_vars=const_vars,
        first_days=first_days,
        data_period=train_period,
        val_days=val_days,
        split='train',
    )
    ds_valid = TiggeMRMSDataset(
        tigge_dir=tigge_dir,
        tigge_vars=tigge_vars,
        mrms_dir=mrms_dir,
        rq_fn=rq_fn,
        const_fn=const_fn,
        const_vars=const_vars,
        first_days=first_days,
        data_period=train_period,
        val_days=val_days,
        split='valid',
        mins=ds_train.mins,
        maxs=ds_train.maxs
    )
    ds_test = TiggeMRMSDataset(
        tigge_dir=tigge_dir,
        tigge_vars=tigge_vars,
        mrms_dir=mrms_dir,
        rq_fn=rq_fn,
        const_fn=const_fn,
        const_vars=const_vars,
        first_days=first_days,
        data_period=test_period,
        mins=ds_train.mins,
        maxs=ds_train.maxs
    )
    logging.info('Train samples:', len(ds_train))
    logging.info('Valid samples:', len(ds_valid))
    logging.info('Test samples:', len(ds_test))

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

    # Create Generator 
    logging.info('Device:', device)
    gen = Generator(
        nres=nres,
        nf_in=ds_train.input_vars,
        nf=nf,
        relu_out=relu_out, 
        spectralnorm= spectralnorm,
        use_noise = use_noise,
    ).to(device)

    
    # Create Discriminator
    disc = Discriminator(  [16, 16, 32, 32, 64, 64], # maybe we should put this as input option
        batch_norm = batch_norm,
        sigmoid = sigmoid,
        conditional = cond_disc, 
        spectralnorm = spectralnorm ).to(device)
    
    
    
    # Setup trainer
    criterion = nn.MSELoss()
    betas = (0.5, 0.999)
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=learning_rate, betas=betas)
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=learning_rate, betas=betas)
    
    
    trainer = GANTrainer(gen, disc, 
        gen_optimizer, disc_optimizer, 
        dl_train, disc_repeats= disc_repeats,
        l_loss='l1', 
        dloss_type = D_loss,
        l_lambda=20)
    

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

        save_path = f'{save_dir}/{exp_id}_test.nc'
        print('Saving prediction as:', save_path)
        preds = create_valid_predictions(model, ds_test)
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
    p.add_argument('--train_period', type=str, default=['2018', '2019'], nargs='+',
        help='Years to be used for training. Default [2018, 2019].'
    )
    p.add_argument('--test_period', type=str, default=['2020', '2020'], nargs='+',
        help='Years to be used for testing. Default [2020, 2020].'
    )
    p.add_argument('--first_days', type=int, default=None, 
        help='First N days of each month used from data'
    )
    p.add_argument('--val_days', type=int, default=6, 
        help='First N days of each month used for validation'
    )
    p.add_argument('--batch_norm', type=bool, default=True, 
        help='If true, uses batchnormalization for the Discriminator'
    )
    p.add_argument('--spectralnorm', type=bool, default=True, 
        help='If true, uses spectral normalization for both Generator and Discriminator'
    )
    p.add_argument('--use_noise', type=bool, default=True, 
        help='uses a noise vector for the Generator'
    )
    p.add_argument('--cond_disc', type=bool, default=True, 
        help='If true, a conditional discriminator is used.'
    )
    p.add_argument('--D_loss', type=str, default='hinge', 
        help='type of loss to be used in discriminator: { Wasserstein, hinge} '
    )
    p.add_argument('--sigmoid', type=bool, default=False, 
        help='If False, no sigmoid is applied at the end of the Disc. --> non-binary output. '
    )
    p.add_argument('--disc_repeats', type=int, default=5, 
        help='How often to repeat discriminator learning per 1x gen.} '
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
    p.add_argument('--early_stopping_patience', type=int, default=None, 
        help='Patience for early stopping. None for no early stopping'
    )
    p.add_argument('--restore_best_weights', type=bool, default=True, 
        help='Restore weight for lowest validation loss when early stopping is on.'
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