#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Imports~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Package imports
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import autocast
from torch.nn.utils import clip_grad_norm_

#Local imports
#from arguments import params, model_params, get_train_loaders
from setup import setup_train
from models.reload_model import reload_model, save_model

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Eval functions~~~~~~~~~~~~~~~~~~~~~

def parse_loss(losses, params):
    out = {k: v for k, v in losses.items()}
    rkey = 'nll' if params['dmol_out'] else 'mae'
    tf = params['t_factor'] 
    kf = params['kl_factor']
    tkf = params['tkl_factor']
    recon_k = [a for a in losses if rkey in a]
    kldiver = [a for a in losses if "kll" in a]
    #Get keys
    trar = [a for a in recon_k if a[-1] == "t"]
    recr = [a for a in recon_k if a not in trar]
    trak = [a for a in kldiver if a[-1] == "t"]
    reck = [a for a in kldiver if a not in trak]
    #Get values
    trV = 1.0 * sum([losses[k] for k in trar])
    tkV = tkf * sum([losses[k] for k in trak])
    rrV = 1.0 * sum([losses[k] for k in recr])
    rkV =  kf * sum([losses[k] for k in reck])
    #Final losses
    out['t_loss'] = trV + tkV
    out['r_loss'] = rrV + rkV
    out['loss']   = out['r_loss'] + tf*out['t_loss']
    return out

def evaluate(f_model, params, data):
    f_model['model'].eval()
    with torch.no_grad():
        num_it = 0
        metrics = dict()
        for i, cdata in enumerate(tqdm(data, leave = False)):
            a,b = cdata
            a = maybe_cuda(a, params['cuda'])
            b = maybe_cuda(b, params['cuda'])
            f_model['model'](a,b)
            num_it += len(a)
            for k,v in f_model['model'].losses.items():
                if k not in metrics:
                    metrics[k] = 0
                metrics[k] += v.item() if v.numel() == 1 else v.mean().item()
        for k,v in metrics.items():
            metrics[k] /= num_it
    metrics_parsed = parse_loss(metrics, params)
    return metrics_parsed

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Misc. Util functions~~~~~~~~~~~~~~~~~~~~~

class none_with:
    def __enter__(self, *args, **kwargs):
        pass
    def __exit__(self, *args, **kwargs):
        pass

def maybe_cuda(v, cuda = False):
    if cuda:
        return v.cuda(non_blocking = True)
    return v

def human_readable_time(seconds):
    out = ""
    tr = int(seconds)
    if (h:=int(tr//3600)):
        tr = tr - (3600 * h)
        out += f"{h}h"
    if (m:=int(tr//60)):
        tr = tr - (60 * m)
        out += f"{m}m"
    out += f"{tr}s"
    return out

def makegt0_1(data):
    #data[data>0] = 1.0
    data = 0.5*data + 0.5
    return data

def save_example_img(cdata, f_model, fname, cuda = False):
    if not fname:
        return
    a,b = cdata
    a = maybe_cuda(a, cuda)
    b = maybe_cuda(b, cuda)
    res = a.shape[-1]
    out = f_model['model'](a,b, return_hats = True)
    nr = 1 + len(out) // 2
    nc = 2
    oimg = np.zeros((nr * res, nc * res, 3))
    for i,ci in enumerate([a,b,*out]):
        cr = i // nc
        cc = i %  nc
        #Expect ci shape: [Batch, Channels, Height, Width]
        ci = ci.detach().cpu().numpy()[0]
        if ci.shape[0] == 1:
            ci = np.repeat(ci, 3, axis=0)
        ci = np.moveaxis(ci, 0, -1)
        oimg[(cr)*res:(cr+1)*res, (cc)*res:(cc+1)*res] = ci
    targ_dir = os.path.dirname(f_model['architecture']['tmp_name'])
    os.makedirs(targ_dir, exist_ok = True)
    plt.imsave(os.path.join(targ_dir,fname), oimg)
    f_model['model'].losses.clear()
    return

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Training functions~~~~~~~~~~~~~~~~~~~~~

def train_step(cdata, model, params, optim, scaler, do_step = True, lfact = 1.0, gnorm = 0.0, recon_only = False, **kwargs):
    grad = -1
    if scaler is not None:
        with_t = autocast(device_type = 'cuda' if params['cuda'] else 'cpu', dtype = torch.float16)
    else:
        with_t = none_with()
    #Do forward
    with with_t:
        a, b = cdata
        a = maybe_cuda(a, params['cuda'])
        b = maybe_cuda(b, params['cuda'])
        if recon_only:
            model.forward_rec_only(a,b)
        elif params['trans_only']:
            model.forward_tra_only(a,b)
        else:
            model(a,b)
        all_losses = parse_loss(model.losses, params)
        loss = all_losses['loss'].mean()
        loss = lfact * loss
    #Do backward
    if not (loss.isnan().any() or loss.isinf().any()):
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
    #Do step
    if do_step:
        if gnorm > 0:
            if scaler is not None:
                scaler.unscale_(optim)
            grad = clip_grad_norm_(model.parameters(), gnorm).item()
        if grad == 0:
            optim.zero_grad()
            return grad
        if scaler is not None:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()
        optim.zero_grad()
    return grad

def train(f_model, params, all_dls):
    print("Starting training")
    print(f"Best performance ({params['best_met']}): {f_model['best_perf']}")
    lfact = 1 / params['step_freq']
    gnorm = params['grad_threshold']
    is_best = False #To track whether to override save freq
    for e in range(params['epochs']):
        st_time = time.time()
        print(f"Starting epoch: {e+1}")
        save_example_img(
                next(iter(all_dls['valid'])),
                f_model,
                params['example_img'],
                cuda = params['cuda'])
        f_model['model'].train()
        for i,cdata in enumerate(tqdm(all_dls["train"], leave = False)):
            do_step = i % params['step_freq'] == 0
            grad = train_step(cdata, **f_model,
                              params = params,
                              do_step = do_step,
                              lfact = lfact,
                              gnorm = gnorm)
        #Evaluate model
        f_model['model'].eval()
        if (e % params['eval_freq'] == 0):
            print("Evaluating model")
            eval_metrics = evaluate(f_model, params, all_dls['valid'])
            is_ltb = eval_metrics[params['best_met']] < f_model['best_perf']
            if is_ltb == params['best_dec']:
                f_model['best_perf'] = eval_metrics[params['best_met']]
                is_best = True
                print(f"New best performance ({params['best_met']}): {f_model['best_perf']}")
        #Save model
        f_model['c_epoch'] += 1
        if is_best or (e % params['save_freq'] == 0):
            print(f"Saving model for epoch: {e+1}")
            save_model(f_model, params, as_best = is_best)
        is_best = False
        ep_time = time.time() - st_time
        print(f"Epoch time: {human_readable_time(ep_time)}")
    return

def warmup(f_model, params, all_dls):
    print(f"Starting warmup")
    for e in range(params['n_warmup']):
        print(f"Starting warmup epoch: {e+1}")
        f_model['model'].train()
        for i, cdata in enumerate(tqdm(all_dls['warmup'], leave = False)):
            wdata = [makegt0_1(a) for a in cdata]
            grad = train_step(wdata, **f_model, 
                              params = params,
                              do_step = True,
                              lfact = 1.0,
                              gnorm = 0.0,
                              recon_only = True)
    return

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
    params, model_params, all_dls = setup_train()
    p_full = {'all': params, 'architecture': model_params}
    f_model = reload_model(p_full, True)
    if (f_model['c_epoch'] == 0) and (params['n_warmup'] > 0):
        warmup(f_model, params, all_dls)
    train(f_model, params, all_dls)

if __name__ == "__main__":
    main()
