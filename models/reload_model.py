#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Imports~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Package imports
import os
import shutil
from copy import deepcopy

import torch
from torch import GradScaler

#Local imports
from models.paired_vae import PVAE

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Creating the models~~~~~~~~~~~~~~~~~~~~

def create_model(model_params):
    model = PVAE(**model_params["kwargs"])
    if model_params["cuda"]:
        model = model.cuda()
    return model

def create_optimizer(model, lr, half_precision = True, cuda = False):
    optimizer = torch.optim.Adamax(model.parameters(), lr = lr)
    #For single precision
    scaler = GradScaler('cuda' if cuda else 'cpu') if half_precision else None
    return optimizer, scaler

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Model loading and saving~~~~~~~~~~~~~~~

def reload_model(params, try_reload = True, load_best = False, error_on_fail = False):
    rel_name = 'best_name' if load_best else 'tmp_name'
    if try_reload and os.path.exists(params['architecture']['tmp_name']):
        print(f"Reloading model from: {params['architecture'][rel_name]}")
        #Reload state
        r_model = torch.load(params['architecture'][rel_name])
        c_model = create_model(r_model["architecture"])
        c_model.load_state_dict(r_model["model"])
        optimizer, scaler = create_optimizer(c_model, 
                                             params['all']['lr'], 
                                             params['all']['half_precision'])
        optimizer.load_state_dict(r_model["optim"])
        #Set vars
        epoch = r_model["c_epoch"]
        best_perf = r_model["best_perf"]
        used_arch = r_model["architecture"]
    else:
        if error_on_fail:
            raise ValueError(f"Could not load model: {params['architecture'][rel_name]}")
        print("Creating new model")
        #Create model
        c_model = create_model(params['architecture'])
        optimizer, scaler = create_optimizer(c_model, params['all']['lr'], params['all']['half_precision'])
        #Set vars
        epoch = 0
        best_perf = float('inf') if params['all']['best_dec'] else float('-inf')
        used_arch = params["architecture"]
    #Create state dict
    f_model = dict(
            model           = c_model,
            optim           = optimizer,
            scaler          = scaler,
            architecture    = used_arch,
            c_epoch         = epoch,
            best_perf       = best_perf,
    )
    return f_model

def save_model(f_model, params, as_best = False):
    s_model = dict(
        model           = f_model['model'].state_dict(),
        optim           = f_model['optim'].state_dict(),
        architecture    = f_model['architecture'],
        c_epoch         = f_model['c_epoch'],
        best_perf       = f_model['best_perf'],
        params          = params,
    )
    bname = f_model['architecture']['best_name']
    tname = f_model['architecture']['tmp_name']
    os.makedirs(os.path.dirname(tname), exist_ok = True)
    os.makedirs(os.path.dirname(bname), exist_ok = True)
    torch.save(s_model, tname)
    if as_best:
        shutil.copyfile(tname, bname)
    return
