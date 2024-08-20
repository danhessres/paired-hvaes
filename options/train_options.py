import os
import json

from .options import TrainOptions

def get_params(args, arch):
    """
    Renaming for legacy code
    """
    params = dict(
        random_state = args.random_state,
        trans_only = args.translation_only,
        cuda = args.cuda,
        epochs = args.epochs,
        n_warmup = args.warmup,
        batch    = args.batch_size,
        num_workers = args.num_workers,
        step_freq = args.step_freq,
        save_freq = args.save_freq,
        eval_freq = args.eval_freq,
        example_img = args.example_img,
        aug_vflip = args.aug_vflip,
        aug_hflip = args.aug_hflip,
        aug_shift = args.aug_shift,
        aug_scale = args.aug_scale,
        lr = args.learning_rate,
        grad_skip = args.grad_skip,
        grad_threshold = args.grad_thres,
        half_precision = not args.no_half_precision,
        kl_factor  = args.loss_kl_fac,
        t_factor   = args.loss_tra_fac,
        tkl_factor = args.loss_tkl_fac,
        best_met   = args.best_metric,
        best_dec   = args.best_decreasing,
        **arch
    )
    return params

def get_transforms(args):
    transforms = dict(
        vflip = args.aug_vflip,
        hflip = args.aug_hflip,
        offs  = args.aug_shift,
        sf    = args.aug_scale
    )
    return transforms

def architecture_parse(
        channels,
        img_res,
        width,
        depth,
        layers_per_group,
        latent_per_group,
        no_bias_above,
        bottleneck,
        translational,
        dmol_output,
        dmol_mixtures,
        dmol_bits,
        *args, **kwargs):
    """
    Pretty architecture
    """
    arch = dict(
        res = img_res,
        cin = channels,
        e_pg  = [layers_per_group] * depth,
        ec_pg = [width] * depth,
        d_pg  = [layers_per_group] * depth,
        dc_pg = [width] * depth,
        lv_pg = [latent_per_group] * depth,
        no_bias_above = no_bias_above,
        n_mix = dmol_mixtures,
        bits = dmol_bits,
        bottleneck = bottleneck,
        translational = translational,
        dmol_out = dmol_output
    )
    return arch
def get_model_params(args, arch):
    model_root = os.path.join(args.checkpoint_dir, args.model_name)
    alt_arch = {k:v for k,v in arch.items()}
    if args.altchannels is not None:
        alt_arch['cin'] = args.altchannels
    arch_full = dict(
        tmp_name = os.path.join(model_root, 'latest.pth.tar'),
        best_name = os.path.join(model_root, 'best.pth.tar'),
        cuda = args.cuda,
        kwargs = dict(
            v1args = [],
            v1kwargs = arch,
            v2args = [],
            v2kwargs = alt_arch,
        )
    )
    return arch_full

def reload_arch(args):
    import torch
    model_root = os.path.join(args.checkpoint_dir, args.model_name)
    model_path = os.path.join(model_root, 'latest.pth.tar')
    raw_model = torch.load(model_path)
    full_arch = raw_model['architecture']
    arch = full_arch['kwargs']['v1kwargs']
    return arch, full_arch

def train_arg_parse(args):
    #Get architecture
    model_root = os.path.join(args.checkpoint_dir, args.model_name)
    if args.reload_model:
        model_path = os.path.join(model_root, 'latest.pth.tar')
        if not os.path.exists(model_path):
            raise ValueError(f"Cannot reload model {model_path}: does not exist")
        arch, full_arch = reload_arch(args)
    else:
        model_path = os.path.join(model_root, 'latest.pth.tar')
        if os.path.exists(model_path):
            raise ValueError(f"Cannot create model {model_path}: model already exists")
        model_path = os.path.join(model_root, 'best.pth.tar')
        if os.path.exists(model_path):
            raise ValueError(f"Cannot create model {model_path}: model already exists")
        if args.arch_file is not None:
            with open(args.arch_file,'r') as f:
                arch = architecture_parse(json.load(f))
        else:
            arch = architecture_parse(**vars(args))
        full_arch = get_model_params(args, arch)
    params = get_params(args, arch)
    transforms = get_transforms(args)
    return params, full_arch, transforms

if __name__ == "__main__":
    pass

