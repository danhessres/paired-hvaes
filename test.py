#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Imports~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Packages
import os
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#Local imports
from setup import setup_test
from models.reload_model import reload_model
from evaluate.evaluate import runner as evaluate_dir

SUB_DIRS = ["gtA", "gtB", "A2A", "B2B", "A2B", "B2A"]

def timed(fn):
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    res = fn()
    t1.record()
    torch.cuda.synchronize()
    return res, t0.elapsed_time(t1) / 1000

def maybe_cuda(v, cuda = False):
    if cuda:
        return v.cuda(non_blocking = True)
    return v

def save_data(t, fname, tdir, params, variant = None):
    """
    Assume t is tensor of shape [1,C,H,W]
    """
    img = t#.detach()
    img = img.cpu().numpy()[0]
    if len(img) == 1:
        img = np.repeat(img, 3, axis = 0)
    img = np.moveaxis(img, 0, -1)
    #best_dir = "best" if use_best else "latest"
    #variant  = f"{best_dir}_t{temp}"
    variant_dir = "" if variant is None else variant
    out_dir = os.path.join(params['out_dir'], params['model_name'], variant_dir, tdir)
    os.makedirs(out_dir, exist_ok = True) #Ensure folder exists
    plt.imsave(os.path.join(out_dir, fname), img)
    return

def run_model(cdata, fname, f_model, params, temp = None, t_dmol = None, variant = None):
    """
    Given data and model/params, run inference on data
    and save to results directory.

    Times just the inference and returns that value

    Assume batch size of 1 - this is for name handling
    """
    a, b = cdata
    save_data(a, fname, SUB_DIRS[0], params, variant = variant)
    save_data(b, fname, SUB_DIRS[1], params, variant = variant)
    a = maybe_cuda(a, params['cuda'])
    b = maybe_cuda(b, params['cuda'])
    # Inference time of each task independently
    (_, hatA2B), t0 = timed(
            lambda: f_model['model'].forward_tra_only(a, None, t = temp, t_dmol = t_dmol))
    (hatB2A, _), t1 = timed(
            lambda: f_model['model'].forward_tra_only(None, b, t = temp, t_dmol = t_dmol))
    (hatA2A, _), t2 = timed(
            lambda: f_model['model'].forward_rec_only(a, None, t = temp, t_dmol = t_dmol))
    (_, hatB2B), t3 = timed(
            lambda: f_model['model'].forward_rec_only(None, b, t = temp, t_dmol = t_dmol))
    #Save outputs
    st0 = time.time()
    save_data(hatA2A, fname, SUB_DIRS[2], params, variant = variant)
    save_data(hatB2B, fname, SUB_DIRS[3], params, variant = variant)
    save_data(hatA2B, fname, SUB_DIRS[4], params, variant = variant)
    save_data(hatB2A, fname, SUB_DIRS[5], params, variant = variant)
    st1 = time.time()
    return { 'A2B': t0, 'B2A': t1, 'A2A': t2, 'B2B': t3 , 'Saving': st1 - st0}

def test_model(use_best, p_full, data, temp = None, t_dmol = None, variant = None, eval_fid = True):
    """
    predefined use_best
    """
    f_model = reload_model(p_full, True, use_best, True)
    num_params = sum(p.numel() for p in f_model['model'].parameters())
    print(f"Number of parameters: {num_params:,}")
    f_model['model'].eval()
    times = {}
    with torch.no_grad():
        #Warmup
        fname = os.path.basename(data.dataset.dir_files[0])
        cdata = next(iter(data))
        _ = run_model(cdata, fname, f_model, p_full['all'], temp = temp, t_dmol = t_dmol, variant = variant)
        for i, cdata in enumerate(tqdm(data, leave = False)):
            fname = os.path.basename(data.dataset.dir_files[i])
            last_time = run_model(cdata, fname, f_model, p_full['all'], temp = temp, t_dmol = t_dmol, variant = variant)
            for k,v in last_time.items():
                if k not in times:
                    times[k] = []
                times[k].append(v)
    print(f"Average times:")
    for k, v in times.items():
        print(f"  {k}: {np.mean(v):.5f}s")
    print(f"Running evaluation")
    variant_dir = "" if variant is None else variant
    base_dir = os.path.join(p_full['all']['out_dir'], p_full['all']['model_name'], variant_dir)
    eval = evaluate_dir(base_dir, skip_fid = (not eval_fid))
    eval_str = json.dumps(eval, indent = 4)
    eval_file = os.path.join(base_dir, 'eval.log')
    with open(eval_file, 'w') as f:
        f.write(eval_str)
    print(f"Evaluation done and written to: {eval_file}")
    return


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
    params, model_params, loader = setup_test() 
    print(f"Testing script started with random_state: {params['random_state']}")
    p_full = {'all': params, 'architecture': model_params}
    for is_best in [True, False]:
        for temp in [1.0, 0.0]:
            best_dir = "best" if is_best else "latest"
            variant  = f"{best_dir}_t{temp}"
            test_model(is_best,  p_full, loader, temp = temp, t_dmol = temp, variant = variant, eval_fid = True)
            print()
    return

if __name__ == "__main__":
    main()
