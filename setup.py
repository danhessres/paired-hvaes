import os
import json
import datetime


from data.loaders import get_train_loaders, get_test_loaders, get_val_loaders
from options.options import TrainOptions, TestOptions
from options.train_options import train_arg_parse
from options.test_options import test_arg_parse

def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return

def preprocess_args(args):
    import argparse
    if args.argfile is None:
        return args
    with open(args.argfile, 'r') as f:
        newargs = json.load(f)
    for k,v in vars(args).items():
        #Overwrite loaded arguments with CLAs
        if k not in newargs:
            newargs[k] = v
    return argparse.Namespace(**newargs)

def save_args(args):
    #Prep args
    args_d = vars(args)
    args_s = json.dumps(args_d, indent = 4)
    #Get timestamp
    dt = datetime.datetime.now()
    timestamp = dt.strftime('%Y%m%d_%H%M%S')
    root_dir = os.path.join(args.checkpoint_dir, args.model_name)
    os.makedirs(root_dir, exist_ok = True)
    ofile = f"train_args_{timestamp}.json"
    ofile = os.path.join(root_dir, ofile)
    with open(ofile, 'w') as f:
        f.write(args_s)
    return

def setup_train():
    opts = TrainOptions()
    args = opts.parse()
    args = preprocess_args(args)
    set_seed(args.random_state)
    params, full_arch, transforms = train_arg_parse(args)
    loaders = get_train_loaders(args, transforms)
    #Save arguments for logging
    save_args(args)
    return params, full_arch, loaders

def setup_test():
    opts = TestOptions()
    args = opts.parse()
    args = preprocess_args(args)
    set_seed(args.random_state)
    params, full_arch = test_arg_parse(args)
    args.img_res  = full_arch['kwargs']['v1kwargs']['res']
    args.channels = full_arch['kwargs']['v1kwargs']['cin']
    args.altchannels = full_arch['kwargs']['v2kwargs']['cin']
    if args.test_on_val:
        loaders = get_val_loaders(args)
    else:
        loaders = get_test_loaders(args)
    return params, full_arch, loaders

def main():
    return

if __name__ == "__main__":
    main()
