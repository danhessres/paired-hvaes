import torch
from torch import nn
from torch.nn import functional as F
from .dmol import DMOL
from collections import defaultdict
import numpy as np
import itertools

def has_len(xs):
    try:
        l = len(xs)
        out = True
    except:
        out = False
    return out

def get_temp(t, i = 0):
    default_ret = -1
    try:
        temp = t[i]
    except TypeError:
        temp = t
    except IndexError:
        #Numpy scalers will index error instead
        if has_len(t):
            if len(t) == 0:
                temp = None
            else:
                temp = t[default_ret]
        else:
            temp = t
    return temp

def gaussian_analytical_kl(pm, qm, pv, qv, var_smooth = None):
    if var_smooth is not None:
        ps = var_smooth(pv)
        pv = ps.log()
        qs = var_smooth(qv)
        qv = qs.log()
    else:
        ps = pv.exp()
        qs = qv.exp()
    return -0.5 + qv - pv + 0.5 * (ps ** 2 + (pm - qm) ** 2) / (qs ** 2)    
    
def gaussian_sampler(mu, std,var_smooth = None): 
    eps = torch.randn_like(mu)
    if var_smooth is not None:
        std = var_smooth(std)
    return std * eps + mu

class swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1))
    def forward(self,x):
        return x*torch.sigmoid(x*self.beta)

class gConv(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.drop = nn.Dropout(p=0.0)
    def forward(self,inp,*args,**kwargs):
        x = self.drop(inp)
        x = super().forward(x,*args,**kwargs)
        return x

class Pool(nn.Module):
    def __init__(self,cin,cou,down_rate):
        super().__init__()
        self.conv = gConv(cin,cou,kernel_size=down_rate,stride=down_rate)
        self.act = nn.LeakyReLU()
    def forward(self,x):
        cx = self.conv(x)
        cx = self.act(cx)
        return cx

class UnPool(nn.Module):
    def __init__(self,cin,cou,up_rate,res_in):
        super().__init__()
        self.conv = gConv(cin,cou,1)
        self.upsa = nn.Upsample(scale_factor = up_rate)
        self.resi = res_in
        self.reso = up_rate * res_in
        self.bias = nn.Parameter(torch.zeros((1,cou,self.reso,self.reso)))
    def forward(self,x,acts=None,**kwargs):
        self.kl = torch.zeros((1,1,1,1),device = x.device)
        cx = x
        cx = self.conv(cx)
        cx = self.upsa(cx)
        cx = cx + self.bias
        return cx,None #Need to return None for "z"
    def forward_uncond(self,x,t=None,lvs=None):
        return self(x)[0]

class Block(nn.Module):
    def __init__(self, cin, cmi, cou, down_rate=None, residual=False, use_3x3=True, zero_last=False, use_swish = False): 
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual

        mk_size = 3 if use_3x3 else 1
        self.c1 = gConv(cin, cmi, 1, padding = 'same')
        self.c2 = gConv(cmi, cmi, mk_size, padding = 'same')
        self.c3 = gConv(cmi, cmi, mk_size, padding = 'same')
        self.c4 = gConv(cmi, cou, 1, padding = 'same')
        if zero_last:
            self.c4.weight.data *= 0
        
        if down_rate is not None:
            self.pool = Pool(cin,cou,down_rate)

        if use_swish:
            self.a1 = swish()
            self.a2 = swish()
            self.a3 = swish()
            self.a4 = swish()
        else:
            self.a1 = nn.GELU()
            self.a2 = nn.GELU()
            self.a3 = nn.GELU()
            self.a4 = nn.GELU()

    def forward(self, x):
        xhat = self.c1(self.a1(x))
        xhat = self.c2(self.a2(xhat))
        xhat = self.c3(self.a3(xhat))
        xhat = self.c4(self.a4(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = self.pool(out)
        return out

def pad_channels(t, width):
    d1, d2, d3, d4 = t.shape
    empty = torch.zeros(d1, width, d3, d4, device=t.device)
    empty[:, :d2, :, :] = t
    return empty

def get_encoder_settings(res,e_pg,ec_pg):
    settings = []
    channels = {}
    c_res = res
    d = None
    for g, (n,c) in enumerate(zip(e_pg,ec_pg)):
        for i in range(n):
            if d is not None:
                c_res = c_res // 2
            d = 2 if i==n-1 and g < len(e_pg) - 1 else None
            u = c_res > 2
            w = c
            settings.append([w,d,u])
            if d is None:
                channels[c_res] = c
    return settings, channels

class Encoder(nn.Module):
    def __init__(self, res, cin, e_pg, ec_pg, bottleneck):
        super().__init__()
        self.preprocess = gConv(cin, ec_pg[0], 3, padding = 'same')
        self.e_set, self.channels = get_encoder_settings(res,e_pg,ec_pg)
        self.enc_blocks = nn.ModuleList()
        for w,d,u in self.e_set:
            self.enc_blocks.append(Block(w, int(w * bottleneck), w, down_rate=d, residual=True, use_3x3=u))
        n_blocks = len(self.enc_blocks)
        for b in self.enc_blocks:
            b.c4.weight.data *= np.sqrt(1 / n_blocks)

    def forward(self, x):
        x = self.preprocess(x)
        activations = {'batch':x.shape[0]}
        activations[x.shape[2]] = x
        for block in self.enc_blocks:
            x = block(x)
            res = x.shape[2]
            x = x if x.shape[1] == self.channels[res] else pad_channels(x, self.channels[res])
            activations[res] = x
        return activations


class DecBlock(nn.Module):
    def __init__(self,res, mixin, cin, ein, latents, n_blocks, bottleneck, gsmooth=np.log(2), translational=False):
        super().__init__()
        self.is_translational = translational
        self.base = res
        self.mixin = mixin
        self.cin = cin
        use_3x3 = res > 2
        cond_width = int(cin * bottleneck)
        self.varsmooth = nn.Softplus(beta = gsmooth)
        self.zdim = latents
        self.enc = Block(cin + ein, cond_width, self.zdim * 2, residual=False, use_3x3=use_3x3, use_swish=True)
        if translational:
            self.encT = Block(cin + ein, cond_width, self.zdim, residual=False, use_3x3=use_3x3, use_swish=True)
        self.prior = Block(cin, cond_width, self.zdim * 2 + cin, residual=False, use_3x3=use_3x3, zero_last=True, use_swish=True)
        self.z_proj = gConv(self.zdim, cin, 1, padding = 'same')
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.resnet = Block(cin, cond_width, cin, residual=True, use_3x3=use_3x3)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.z_fn = lambda x: self.z_proj(x)

    def sample_uncond(self, x, t=None, lvs=None):
        n, c, h, w = x.shape
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp
        if t is not None:
            pv = pv + torch.ones_like(pv) * np.log(t)
        if lvs is not None:
            qs,qc = lvs
            ps,pc = gaussian_sampler(pm,pv).chunk(2,dim=1)
            zs = qs if qs is not None else ps
            zc = qc if qc is not None else pc
            z = torch.cat([zs,zc],dim=1)
        else:
            z = gaussian_sampler(pm, pv, self.varsmooth)
        return z, x

    def forward(self, x, activations, get_latents=False, translate = False, t=None):
        #Compute prior
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]

        #Compute posterior
        acts = activations[self.base]
        if translate and self.is_translational:
            psm, _ = pm.chunk(2,dim=1)
            psv, _ = pv.chunk(2,dim=1)
            qcm, qcv = self.encT(torch.cat([x,acts],dim=1)).chunk(2,dim=1)
            qm = torch.cat((psm, qcm), dim = 1)
            qv = torch.cat((psv, qcv), dim = 1)
        else:
            qm, qv = self.enc(torch.cat([x,acts], dim=1)).chunk(2, dim=1)

        #Compute kl
        self.kl = gaussian_analytical_kl(qm, pm, qv, pv, self.varsmooth)

        #Sample from posterior
        if t is not None:
            qv = qv + torch.ones_like(qv) * np.log(t)
        z = gaussian_sampler(qm, qv, self.varsmooth)

        #Output
        x = x + xpp + self.z_fn(z)
        x = self.resnet(x)
        if get_latents:
            qs,qc = z.chunk(2,dim=1)
            return x, [qs,qc]
        return x, None

    def forward_uncond(self, x, t=None, lvs=None):
        #Assume lvs of form lvs = [style,content]
        z, x = self.sample_uncond(x, t, lvs=lvs)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        return x


def get_decoder_settings(res, d_pg, dc_pg, ec_pg, lv_pg, bottleneck):
    settings = []
    channels = {}
    n_blocks = sum(d_pg)
    c_res = res
    for g,(n,c,e,l) in enumerate(zip(d_pg, dc_pg, ec_pg, lv_pg)):
        channels[c_res] = c
        for i in range(n):
            mix = c_res//2 if i==(n-1) and g < len(d_pg) - 1 else None
            cset = [c_res, mix, c, e, l, n_blocks, bottleneck]
            settings.append(cset)
        c_res = c_res//2
    settings = list(reversed(settings))
    return settings, channels


class Decoder(nn.Module):
    def __init__(self, res, cou, d_pg, dc_pg, ec_pg, lv_pg, bottleneck, no_bias_above, n_mix, bits,translational=False, dmol_out=True):
        super().__init__()
        self.res = res
        self.dmol_out = dmol_out
        resos = set()
        dec_blocks = []
        self.set, self.channels = get_decoder_settings(res, d_pg, dc_pg, ec_pg, lv_pg, bottleneck)
        last_res = self.set[0][0]
        last_cha = self.set[0][2]
        for idx, d_parms in enumerate(self.set):
            if last_res != d_parms[0]:
                dec_blocks.append(UnPool(last_cha,d_parms[2],2,last_res))
            dec_blocks.append(DecBlock(*d_parms,translational=translational))
            resos.add(d_parms[0])
            last_res = d_parms[0]
            last_cha = d_parms[2]
        resos = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.dec_bias = nn.Parameter(torch.zeros(1,self.set[0][2],self.set[0][0],self.set[0][0]))
        
        self.gain = nn.Parameter(torch.ones(1, dc_pg[0], 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dc_pg[0], 1, 1))
        if self.dmol_out:
            self.final_fn = lambda x: x * self.gain + self.bias
            self.out_net = DMOL(dc_pg[0],cou,n_mix)
        else:
            self.out_net = gConv(dc_pg[0],cou,1)
            self.final_fn = lambda x: self.out_net(x * self.gain + self.bias)

    def get_kll(self):
        kl = 0
        for block in self.dec_blocks:
            c_kl = block.kl.sum(dim=(1,2,3))
            kl = kl + c_kl
        return kl

    def forward(self, activations, get_latents = False, translate = False, t=None):
        """
        Conditional forward decode
        """
        latents = []
        batch = activations['batch']
        x = self.dec_bias.repeat(batch,1,1,1)
        for i,block in enumerate(self.dec_blocks):
            temp = get_temp(t, i)
            x, z = block(x, activations, get_latents = get_latents, translate = translate, t=temp)
            latents.append(z)
        x = self.final_fn(x)
        return x, latents

    def forward_uncond(self, n, t = None):
        """
        Unconditional forward decode
        """
        x = self.dec_bias.repeat(n,1,1,1)
        for i, block in enumerate(self.dec_blocks):
            temp = get_temp(t, i)
            x = block.forward_uncond(x, temp)
        x = self.final_fn(x)
        return x

    def forward_latents(self, latents, t=None):
        """
        Semi-deterministic forward decode
        """
        n = latents[0][0].size(0) if latents[0][0] is not None else latents[0][1].size(0) #Determine batch
        x = self.dec_bias.repeat(n,1,1,1)
        for block, lvs in itertools.zip_longest(self.dec_blocks, latents):
            x = block.forward_uncond(x, t, lvs=lvs)
        x = self.final_fn(x)
        return x


class VDVAE(nn.Module):
    def __init__(self, res, cin, e_pg, ec_pg, d_pg, dc_pg, lv_pg, no_bias_above, n_mix, bits, bottleneck,translational=False,dmol_out=True,cou=None):
        super().__init__()
        self.nll = None
        self.mae = None
        self.kll = None
        self.kl_rate = res*res*cin
        self.bits = bits
        self.dmol_out = dmol_out
        self.encoder = Encoder(res,cin,e_pg,ec_pg,bottleneck)
        if cou is None:
            acou = cin
        else:
            acou = cou
        self.decoder = Decoder(res,acou,d_pg,dc_pg,ec_pg,lv_pg,bottleneck,no_bias_above, n_mix, self.bits,translational=translational,dmol_out=dmol_out)
    
    def activate(self,x, y = None, t = None, t_dmol = None):
        """
        Applies final activation for the decoder output 'x'
        Computes recon loss if y is not None
        """
        if self.dmol_out:
            if y is not None:
                self.nll = self.decoder.out_net.nll(x, y, bits = self.bits)
            x = self.decoder.out_net.sample(x, t = t_dmol)
            x = (x + 1)/2
        else:
            x = torch.sigmoid(x)
            if y is not None:
                self.mae = (y - x).abs().mean(dim=(1,2,3))
        return x

    def forward_encode(self, x):
        """
        Perform forward pass for encode only
        Assumes x input in range 0-1
        """
        x = 2 * x - 1
        return self.encoder.forward(x)

    def forward_decode(self, x, y = None, return_lvs = False, translate = False, t = None, t_dmol = None):
        """
        Perform forward pass for decoder only
        Assumes x is encoder output, y in range 0->1
        """
        #Forward
        if self.dmol_out and y is not None:
            y = 2 * y - 1
        x, lvs = self.decoder.forward(x, get_latents = return_lvs, translate = translate, t = t)
        x = self.activate(x, y = y, t = t, t_dmol = t_dmol)
        self.kll = self.decoder.get_kll() / self.kl_rate
        out = (x,lvs) if return_lvs else x
        return out
    
    def decode_from_latents(self, lvs, y = None, t = None, t_dmol = None):
        """
        Uses the pre-sampled lvs for each layer in hierarchy
        Assume lvs of the form lvs = [[style,content]]
        Layers with no LV are sampled from prior
        """
        if self.dmol_out and y is not None:
            y = 2 * y - 1
        x = self.decoder.forward_latents(lvs, t = t)
        x = self.activate(x, y = y, t = t, t_dmol = t_dmol)
        self.kll = self.decoder.get_kll() / self.kl_rate
        return x
    
    def forward(self, x, y = None, target_x = True, return_enc = False, return_lvs = False, translate = False, t = None, t_dmol = None):
        """
        Does forward encode/decode pass on the input x
        If y is None and target_x is True, computes loss for target = x
        If y is not None, computes loss for target = y
        """
        if (y is None) and target_x:
            y = x
        x = self.forward_encode(x)
        if return_enc:
            e = x
        x = self.forward_decode(x, y = y, return_lvs = return_lvs, translate = translate, t = t, t_dmol = t_dmol)
        if return_enc:
            if return_lvs:
                x, l = x
                return x, e, l
            return x, e
        return x

    def get_losses(self, append_name = None):
        name_ext = ""
        if append_name is not None:
            name_ext = f"_{append_name}"
        out = {}
        out[f"kll{name_ext}"] = self.kll
        if self.dmol_out:
            out[f"nll{name_ext}"] = self.nll
        else:
            out[f"mae{name_ext}"] = self.mae
        return out

    def generate(self, n_batch, t=None, t_dmol = None):
        """
        Unconditional sampling on the prior
        """
        x = self.decoder.forward_uncond(n_batch, t=t)
        x = self.activate(x, y = None, t = t, t_dmol = None)
        return x
