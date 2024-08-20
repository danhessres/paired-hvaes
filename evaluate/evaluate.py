#!/usr/bin/python3
import os
import sys
import math
import json
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as _ssim
from skimage.metrics import peak_signal_noise_ratio as _psnr

#Modified FID from pytorch-fid
if __name__ == "__main__":
    from custom_fid import fid_from_stats, path_stats
else:
    from .custom_fid import fid_from_stats, path_stats


def PSNR(y,yhat):
    """
    y:      Ground truth image
    yhat:   Predicted image
    """
    if np.all(y == yhat):
        return 50.0
    return _psnr(y,yhat)

def SSIM(y,yhat):
    """
    y:      Ground truth image
    yhat:   Predicted image
    """
    return _ssim(y, yhat, data_range = 1, channel_axis = 2)

def FIDS(im_dir):
    """
    im_dir: Directory of images to calculate stats from
    """
    m,s = path_stats(im_dir,1,'cuda',2048,1)
    return m,s


def FIDV(yS,yhatS):
    """
    y:      Ground truth stats
    yhat:   Output stats
    """
    try:
        fid_val = fid_from_stats(*yS,*yhatS)
    except ValueError:
        return float('nan')
    return fid_val

def get_representative_index(met_mean, met_raws):
    representatives = dict()
    for c_type, type_v in met_raws.items():
        csum = None
        for c_metr, raw_vs in type_v.items():
            target = met_mean[c_type][c_metr]['mean']
            cdiff = np.array([(cv - target)**2 for cv in raw_vs])
            if c_metr == "SSIM":
                cdiff *= 10
            if csum is None:
                csum = np.zeros_like(cdiff)
            csum += cdiff
        representatives[c_type] = np.argmin(csum)
    return representatives

def process_modality(gtd,oud,base='.',skip_fid = False):
    """
    gtd:    Ground truth directory
    oud:    Output directories to compare - assume file-names are same as gtd
    base:   Base directory to prepend above directories
    """
    mets_raw = {a:{'PSNR':[],'SSIM':[]} for a in oud}
    gt_bdir = os.path.join(base,gtd)
    gt_fnames = os.listdir(gt_bdir)
    for c_fname in gt_fnames:
        c_y = plt.imread(os.path.join(gt_bdir,c_fname))[:,:,:3]
        for c_type in oud:
            x_fname = os.path.join(base, c_type, c_fname)
            c_x = plt.imread(x_fname)[:,:,:3]
            mets_raw[c_type]['PSNR'].append(PSNR(c_y,c_x))
            mets_raw[c_type]['SSIM'].append(SSIM(c_y,c_x))
    metrics = {
        k:{
            b:{
                'mean': np.mean(v[b]),'std': np.std(v[b])} \
            for b in v}
        for k,v in mets_raw.items()
    }
    if not skip_fid:
        gt_fid = FIDS(gt_bdir)
        for c_type in oud:
            c_dir = os.path.join(base, c_type)
            tar_fid = FIDS(c_dir)
            metrics[c_type]['FID'] = FIDV(gt_fid, tar_fid)
    #Include raw performances
    for c_type in oud:
        metrics[c_type]['raw-psnr'] = {k:v for k,v in zip(gt_fnames, mets_raw[c_type]['PSNR'])}
        metrics[c_type]['raw-ssim'] = {k:v for k,v in zip(gt_fnames, mets_raw[c_type]['SSIM'])}
    return metrics

def clean_dict_floats(d_in):
    out = {}
    for k,v in d_in.items():
        if type(v) is np.float32:
            newv = float(v)
        elif type(v) is dict:
            newv = clean_dict_floats(v)
        else:
            newv = v
        out[k] = newv
    return out

def runner(base_dir, skip_fid = False):
    #Directories
    modalities = ["A","B"]
    gt_dirs = ['gt'+a for a in modalities]
    folders = [[f"{x}2{y}" for x in modalities]for y in modalities]
    out = {}
    for gt, comps in zip(gt_dirs, folders):
        acomps = [a for a in comps if os.path.exists(os.path.join(base_dir,a))]
        c_met = process_modality(gt,acomps,base_dir,skip_fid = skip_fid)
        for k,v in c_met.items():
            out[k] = v
    out = clean_dict_floats(out)
    return out

if __name__ == '__main__':
    #Argument parse
    assert len(sys.argv) == 2,'Expected exactly 1 argument to process'
    base_dir = sys.argv[-1]
    out = runner(base_dir)
    out_str = json.dumps(out, indent=4)
    print(out_str)


