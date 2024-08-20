import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.shape) - 1
    m = x.max(dim=axis, keepdim=True)[0]
    return x - m - torch.log(torch.exp(x - m).sum(dim=axis, keepdim=True))

def const_max(t, constant):
    return torch.clamp(t,min = constant)

def const_min(t, constant):
    return torch.clamp(t,max = constant)

def discretized_mix_logistic_loss(x, l, nr_mix, bits = 8):
    # log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval    
    # Adapted from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    # - Adapted version allows for arbitrary number of channels and arbitrary nr_mix while still allowing full subpixel dependence

    # Input has channel on second index, need to permute    
    x = x.moveaxis(1,-1)    
    l = l.moveaxis(1,-1)    

    xs = [s for s in x.shape]  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)    
    ls = [s for s in l.shape]  # predicted distribution, e.g. (B,32,32,100)    
    nr_cha = int(xs[3])    
    nr_coe = int(nr_cha*(nr_cha-1)/2) #Number of coefficients    
    coe_si = int((nr_coe*nr_mix)/nr_cha) #Size of coefficients channel    

    #Unpack    
    logit_probs = l[:, :, :, :nr_mix]    
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [-1])    
    means = l[:, :, :, :, :nr_mix]    
    scales = l[:, :, :, :, nr_mix:2 * nr_mix]
    scales = F.softplus(scales,np.log(2))
    log_scales = const_max(scales.log(), -7.) 
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix: 2*nr_mix + coe_si])    
    coeffs = coeffs.reshape(xs[:-1] + [nr_coe] + [nr_mix])    

    #Centering values    
    x = torch.reshape(x, xs + [1]) + torch.zeros(xs + [nr_mix]).to(x.device)    
    all_means = [means[:,:,:,0:1]]    
    coe_tracker = 0    
    for i in range(1,nr_cha):    
        cm = means[:,:,:,i]    
        for j in range(i):    
            cm = coeffs[:,:,:,coe_tracker] * x[:,:,:,j] + cm    
            coe_tracker += 1    
        cm = cm.reshape(xs[:-1] + [1] + [nr_mix])    
        all_means.append(cm)    
    means = torch.cat(all_means, dim=3)    
    centered_x = x - means    
    inv_stdv = torch.exp(-log_scales)

    bit_scale = 2**bits - 1    

    plus_in = inv_stdv * (centered_x + 1. / bit_scale)    
    cdf_plus = torch.sigmoid(plus_in)    
    min_in = inv_stdv * (centered_x - 1. / bit_scale)

    cdf_min = torch.sigmoid(min_in)    
    log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0 (before scaling)    
    log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)  # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    log_probs = torch.where(x < -0.999,
                            log_cdf_plus,
                            torch.where(x > 0.999,
                                        log_one_minus_cdf_min,
                                        torch.where(cdf_delta > 1e-5,
                                                    torch.log(const_max(cdf_delta, 1e-12)),
                                                    log_pdf_mid - np.log(bit_scale/2))))
    log_probs = log_probs.sum(dim=3) + log_prob_from_logits(logit_probs)
    mixture_probs = torch.logsumexp(log_probs, -1)
    return -1. * mixture_probs.sum(dim=[1, 2]) / np.prod(xs[1:])

def sample_from_discretized_mix_logistic(l, nr_mix, nr_cha, temp = None, gtemp = None):
    l = l.permute(0,2,3,1)

    ls = [s for s in l.shape]
    xs = ls[:-1] + [nr_cha]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [-1])

    gt = gtemp if gtemp is not None else 1.0
    if gt > 0:
        sel = F.gumbel_softmax(logit_probs, tau = gt, hard = True, dim = 3)
    else:
        #Allows for temp = 0 mixtue sampling
        amax = torch.argmax(logit_probs, dim = 3)
        sel = F.one_hot(amax, num_classes = nr_mix).float()
    sel = torch.reshape(sel, xs[:-1] + [1, nr_mix])

    # select logistic parameters
    means = (l[:, :, :, :, :nr_mix] * sel).sum(dim=4)
    scales = l[:, :, :, :, nr_mix:2 * nr_mix]
    scales = F.softplus(scales,np.log(2))
    log_scales = scales.log() 
    log_scales = const_max((log_scales * sel).sum(dim=4), -7.)
    coeffs = torch.tanh(l[:, :, :, :, nr_mix * 2:])
    coeffs = coeffs.reshape(xs[:-1] + [-1] + [nr_mix])*sel
    coeffs = coeffs.sum(dim=4)

    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    t = temp if temp is not None else 1.0
    u = torch.empty(means.shape, device=means.device).uniform_(1e-5, 1. - 1e-5)
    x = means + t * torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))

    xh = []
    coe_tracker = 0
    cx = const_min(const_max(x[:, :, :, 0], -1.), 1.)
    cx = cx.reshape(xs[:-1]+[1])
    xh.append(cx)
    for i in range(1,nr_cha):
        cx = x[:,:,:,i]
        for j in range(i):
            cx = cx + coeffs[:,:,:,coe_tracker]*xh[j][:,:,:,0]
            coe_tracker += 1
        cx = const_min(const_max(cx,-1.),1.)
        cx = cx.reshape(xs[:-1]+[1])
        xh.append(cx)
    xh = torch.cat(xh,dim=3)
    xh = xh.permute(0,3,1,2)
    return xh

class DMOL(nn.Module):
    def __init__(self, cin, cou, nr_mix):
        super().__init__()
        self.cin = cin
        self.cou = cou# 3 colours
        self.nr_mix = nr_mix

        nr_coe = int(cou*(cou-1)/2) #Number of coefficients        
        coe_si = int((nr_coe*nr_mix)/cou) #Size of coefficients channel    
        c_trans = cou * (2 * nr_mix + coe_si) + nr_mix #Total required channels

        self.out_conv = nn.Conv2d(cin, c_trans, 1, padding='same')

    def nll(self, px_z, x, bits):
        l = self(px_z)
        nll = discretized_mix_logistic_loss(x, l, nr_mix = self.nr_mix, bits = bits)
        return nll

    def forward(self, px_z):
        l = self.out_conv(px_z)
        return l

    def sample(self, px_z, t = None):
        l = self(px_z)
        im = sample_from_discretized_mix_logistic(l, self.nr_mix, self.cou, temp = t)
        return im
