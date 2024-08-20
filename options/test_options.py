import os
from .train_options import reload_arch

def get_test_params(args):
    """
    Change names for legacy code
    """
    params = dict(
        model_name = args.model_name,
        out_dir = args.out_dir,
        lr = 0.0,
        half_precision = False,
        cuda = args.cuda,
        random_state = args.random_state,
    )
    return params

def test_arg_parse(args):
    arch, full_arch = reload_arch(args)
    params = get_test_params(args)
    return params, full_arch
