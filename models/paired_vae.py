import random
import numpy as np
import torch
import torch.nn as nn

def psnr(y,yh):
    """
    PSNR estimate for performance checking mid-epoch.
    Uses a MAX_PSNR/min_mse value to avoid log(0)
    Assume on range from 0-1
    """
    #Not for backprop
    with torch.no_grad():
        D_RANGE=1
        MAX_PSNR = 50
        min_mse =10**(-(MAX_PSNR/10))
        mse = ((y - yh) ** 2).mean(dim=(1, 2, 3))
        mse = torch.clamp(mse,min=min_mse)
        psnr_unscaled = D_RANGE / mse
        psnr = 10 * psnr_unscaled.log10()
    return psnr

from .vae import VDVAE

#Paired VAE
class PVAE(nn.Module):
    def __init__(self,v1args,v1kwargs,v2args,v2kwargs):
        super().__init__()
        self.v1 = VDVAE(*v1args,**v1kwargs)
        self.v2 = VDVAE(*v2args,**v2kwargs)
        #Initloss
        self.losses = {}

    def forward_encode(self, x1 = None, x2 = None):
        if x1 is not None:
            x1 = self.v1.forward_encode(x1)
        if x2 is not None:
            x2 = self.v2.forward_encode(x2)
        return x1, x2

    def forward_decode(self, e1 = None, e2 = None, x1 = None, x2 = None, translate = False, t = None, t_dmol = None, loss_append = None):
        if e1 is not None:
            e1 = self.v1.forward_decode(e1, y = x1, translate = translate, t = t, t_dmol = t_dmol)
            if loss_append is not None:
                self.losses.update(self.v1.get_losses(f'x1{loss_append}'))
                if x1 is not None:
                    self.losses[f"psnr_x1{loss_append}"] = psnr(x1, e1)
        if e2 is not None:
            e2 = self.v2.forward_decode(e2, y = x2, translate = translate, t = t, t_dmol = t_dmol)
            if loss_append is not None:
                self.losses.update(self.v2.get_losses(f'x2{loss_append}'))
                if x2 is not None:
                    self.losses[f"psnr_x2{loss_append}"] = psnr(x2, e2)
        return e1, e2

    def forward_rec_only(self, x1 = None, x2 = None, t = None, t_dmol = None):
        """
        Utility function if no need to store encodings
        """
        if x1 is not None:
            x1 = self.v1.forward(x1, target_x = True, t = t, t_dmol = t_dmol)
            self.losses.update(self.v1.get_losses('x1'))
            #self.losses[f"psnr_x1"] = psnr()
        if x2 is not None:
            x2 = self.v2.forward(x2, target_x = True, t = t, t_dmol = t_dmol)
            self.losses.update(self.v2.get_losses('x2'))
            #self.losses[f"psnr_x2"]
        return x1, x2

    def forward_tra_only(self, x1 = None, x2 = None, t = None, t_dmol = None):
        """
        Utility function if no need to store encodings
        """
        e1, e2 = self.forward_encode(x1, x2)
        x1, x2 = self.forward_decode(e2, e1, x1, x2, translate = True, t = t, loss_append = 't', t_dmol = t_dmol)
        return x1, x2
    
    def forward(self, x1 = None, x2 = None, return_hats = False, t = None, t_dmol = None):
        """
        Runs all possible computes on the inputs while minimizing recomputation
        """
        e1, e2 = self.forward_encode(x1, x2)
        r1, r2 = self.forward_decode(e1, e2, x1, x2, translate = False, t = t, loss_append = '', t_dmol = t_dmol)
        t1, t2 = self.forward_decode(e2, e1, x1, x2, translate = True, t = t, loss_append = 't', t_dmol = t_dmol)
        out = [r1, r2, t2, t1] if return_hats else None
        return out
