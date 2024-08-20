import os
from torch.utils.data import DataLoader
from .basic_dataset import BasicDataset

def get_train_loaders(args, transforms):
    get_dir = lambda s: os.path.join(args.data_dir, s)
    get_ds = lambda s, t : BasicDataset(
                path = get_dir(s),
                res = args.img_res,
                cha = args.channels,
                alt_c = args.altchannels,
                preload = args.load_all,
                transforms = t
            )
    train_ds = get_ds('train', transforms)
    valid_ds = get_ds('val', None)
    warmup_ds = get_ds('train', None) if args.warmup > 0 else None
    loaders = dict(
        train  = DataLoader(
            dataset = train_ds,
            batch_size = args.batch_size,
            shuffle = True,
            pin_memory = True,
            num_workers = args.num_workers),
        valid  = DataLoader(
            dataset = valid_ds,
            batch_size = args.batch_size,
            shuffle = False,
            pin_memory = True,
            num_workers = args.num_workers),
        warmup = None if warmup_ds is None else DataLoader(
            dataset = warmup_ds,
            batch_size = 1,
            shuffle = False,
            pin_memory = True,
            num_workers = args.num_workers),
    )
    return loaders

def get_test_loaders(args):
    test_dir = os.path.join(args.data_dir, 'test')
    test_ds = BasicDataset(
            path = test_dir,
            res  = args.img_res,
            cha  = args.channels,
            alt_c = args.altchannels,
            preload = args.load_all,
            transforms = None)
    loader = DataLoader(
        dataset = test_ds,
        batch_size = 1,
        shuffle = False,
        pin_memory = True,
        num_workers = 0)
    return loader

def get_val_loaders(args):
    test_dir = os.path.join(args.data_dir, 'val')
    test_ds = BasicDataset(
            path = test_dir,
            res  = args.img_res,
            cha  = args.channels,
            alt_c = args.altchannels,
            preload = args.load_all,
            transforms = None)
    loader = DataLoader(
        dataset = test_ds,
        batch_size = 1,
        shuffle = False,
        pin_memory = True,
        num_workers = 0)
    return loader
