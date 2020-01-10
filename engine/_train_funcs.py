import torch
import torch.nn as nn
import torch.nn.functional as func
from utils import *
import numpy as np
import copy


class Funcs:
    
    def storm_iterator(self, traces, print_freq):
        """ Basic mini-batch iterator """
        if self.train_mode == 'co':
            print_freq = print_freq//2
        
        for _ in range(print_freq):
            if self.local_context:
                choice = np.random.choice(np.arange(2,len(traces)-3), self.batch_size, replace=False)
                yield np.array([traces[c-1:c+2] for c in choice]).astype('float32'), np.array([traces[c-2:c+1] for c in choice]).astype('float32'), np.array([traces[c:c+3] for c in choice]).astype('float32'), choice
            else:
                choice = np.random.choice(np.arange(len(traces)), self.batch_size, replace=False)
                yield np.array([traces[c] for c in choice])[:,None].astype('float32'), 0, 0, choice

    def train_ae(self, x, x_m1, x_p1):

        maps = gpu(np.array(self.train_map).mean(0,keepdims=True).repeat(self.batch_size, 0))

        if self.global_context:
            x = gpu(x.astype('float32'))
            hbar = gpu(np.array(self.train_hbar).mean(0, keepdims=True))
            skip_cont = np.random.binomial(1, p=0.5)
            hbar = (1 - skip_cont) * hbar
            hbar = hbar.repeat_interleave(self.batch_size, 0)
            hiddens = self.recfunc(x, return_map=True)

            p, _, _, _, _ = self.recfunc(x, self.n_samples, hbar, sample=False)
            p_store = cpu(torch.sigmoid(p).mean(0))
            P, S, XYZI_m, XYZI_s, BG = self.recfunc(x[self.sl_4d], self.n_samples, hbar[self.sl_4d])

        else:
            x = gpu(x.astype('float32'))
            P, S, XYZI_m, XYZI_s, BG = self.recfunc(x[self.sl_4d], self.n_samples)

        if self.local_context:

            x_m1 = gpu(x_m1.astype('float32'))
            x_p1 = gpu(x_p1.astype('float32'))
            
            H = None

            if self.global_context: H = hbar[self.sl_4d]

            P_m1, S_m1, XYZI_m1, XYZIs_m1, BG = self.recfunc(x_m1[self.sl_4d], N=self.n_samples, H=H)
            P_p1, S_p1, XYZI_p1, XYZIs_p1, BG = self.recfunc(x_p1[self.sl_4d], N=self.n_samples, H=H)

            CS = torch.cat([S_m1[:,None], S_p1[:,None]],1) 
            CP = torch.cat([P_m1[:,None], P_p1[:,None]],1)
            CC = torch.cat([XYZI_m1, XYZI_p1],1)
            CCs = torch.cat([XYZIs_m1, XYZIs_p1],1)

        else:

            CS = None; CP = None; CC = None; CCs = None

        F = self.mgen.genfunc(S, XYZI_m)

        self.optimizer_rec.zero_grad()
        if not self.fixed_psf:
            self.optimizer_gen.zero_grad()
            self.optimizer_wmap.zero_grad()

        self.loss_var = x[self.sl_4d],P,S,XYZI_m,XYZI_s,F,maps,BG, CS, CP, CC, CCs

        loss = self.elbo_loss(x[self.sl_4d],P,S,XYZI_m,XYZI_s,F,maps[self.sl_3d],BG,CS=CS,CP=CP,CC=CC, CCs=CCs)
        
        if not self.fixed_psf:
            loss += 1e3 * torch.norm(self.mgen.w_map.sum(-1).sum(-1), 1)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.net_pars, max_norm=self.ae_norm, norm_type=2)

        self.optimizer_rec.step()
        self._iter_count += 1
        self.scheduler_rec.step()

        if not self.fixed_psf:
            self.optimizer_gen.step()
            self.optimizer_wmap.step()

        if self.global_context:
            if self._iter_count > 100:
                self.train_map.append(p_store.astype('float32'))
            self.train_hbar.append(cpu(hiddens.mean(0)).astype('float32'))
            if len(self.train_hbar) > 100:
                self.train_hbar.pop(0)
            if len(self.train_map) > 100:
                self.train_map.pop(0)

        return loss.detach()

    def train_sl(self):
        
        x_sim, xyzi_mat, s_mask, bg = self.mgen.sim_func(M=gpu(np.array(self.train_map)[self.sl_3d].mean(0, keepdims=True)), batch_size=self.batch_size, local_context=self.local_context, add_wmap=True, add_noise=True, sim_iters=self.sim_iters)
       
        BG = None
        if self.global_context:
            hbar = gpu(np.array(self.train_hbar).mean(0, keepdims=True))
            skip_cont = np.random.binomial(1, p=0.5)
            hbar = (1 - skip_cont) * hbar[self.sl_4d]
            hbar = hbar.repeat_interleave(self.batch_size, 0)
            P, S, XYZI_m, XYZI_s, BG = self.recfunc(x_sim, H=hbar, sample=False)
        else:
            P, S, XYZI_m, XYZI_s, BG = self.recfunc(x_sim, sample=False)

        self.optimizer_rec.zero_grad()

        loss = self.simu_loss(P,XYZI_m,XYZI_s, xyzi_mat, s_mask, BG, bg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.net_pars, max_norm=self.sl_norm, norm_type=2)

        self.optimizer_rec.step()
        self.scheduler_rec.step()
        self._iter_count += 1

        return loss.detach()