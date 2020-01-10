import torch
import torch.nn as nn
import torch.nn.functional as func
from utils import *
import numpy as np

from nnet import *

class Funcs:
      
    def recfunc(self, X, N=1, H=None, return_map=False, sample=True):
        
        img_h, img_w = X.shape[-2], X.shape[-1]
    
        scaled_x = (X - self.rec_pars['offset'])/self.rec_pars['factor']
        
        if X.ndimension() == 3:
            scaled_x = scaled_x[:,None]
            single_h = self.single_net(scaled_x)
            if self.local_context:
                zeros = torch.zeros_like(single_h[:1])
                h_t0 = single_h
                h_tm1 = torch.cat([zeros, single_h], 0)[:-1]
                h_tp1 = torch.cat([single_h, zeros], 0)[1:]
                single_h = torch.cat([h_tm1, h_t0, h_tp1], 1)   
        else:        
            single_h = self.single_net(scaled_x.reshape([-1, 1, img_h, img_w])).reshape(-1, self.n_filters*self.n_inp, img_h, img_w)
        
        if return_map:
            if self.local_context:
                return single_h[:, self.n_filters:2*self.n_filters] 
            else:
                return single_h
        else:
        
            if self.global_context:
                if H is None: H = torch.zeros_like(scaled_x[:,:1]).repeat_interleave(self.n_filters,1)  
                comb_h = torch.cat([single_h, H],1)
            else:
                comb_h = single_h

            features = self.comb_net(comb_h)
            outputs = self.out_net.forward(features)
            
            if self.sig_pred:
                xyzi_s = torch.sigmoid(outputs['xyzi_sig']) + 0.01
            else:
                xyzi_s = 0.2*torch.ones_like(outputs['xyzi'])
            
            probs = torch.clamp(outputs['p'],-8.,8.)
            
            xyzi_m = outputs['xyzi']
            xyzi_m[:,:2] = torch.tanh(xyzi_m[:,:2])
            xyzi_m[:,3] = torch.sigmoid(xyzi_m[:,3])
            if '3D' in self.modality:
                xyzi_m[:,2] = torch.tanh(xyzi_m[:,2])
                
            bg = torch.sigmoid(outputs['bg'])[:,0] if self.bg_pred else None
            s = torch.distributions.Binomial(1,logits = probs.repeat_interleave(N,0)).sample()[:,0] if sample else None
            
            return probs[:,0], s, xyzi_m, xyzi_s, bg
