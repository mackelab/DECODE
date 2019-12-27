import torch
import torch.nn as nn
import torch.nn.functional as func
from utils import *
import numpy as np

import _loss_funcs

class LikelihoodModel(_loss_funcs.Funcs):

    def __init__(self, psf_params, filt_size = 51):
         
        self.psf_pars = psf_params
        self.taylor_corr = 1/12
        self.filt_size = filt_size
        self.loop_func = 'unique'
        self.beads_fit = True if 'beads_fit' in self.psf_pars else False
        self.interpolate = True
        
            
        self.n_maps = int(2 * self.psf_pars['z_scale'] + 1)
        self.w_map = torch.nn.Parameter(gpu(np.zeros([self.n_maps, self.filt_size, self.filt_size])))
        
        if self.psf_pars['modality'] == 'GAUSS_2D':
            self.trainable_pars = {'w_prop','width2'}
        if self.psf_pars['modality'] == 'ASTIG_3D':
            self.trainable_pars = {'om_0','c_x','c_y','d'}
        if self.psf_pars['modality'] == 'HELIX_3D':
            self.trainable_pars = {'rad','rot_multi','width','zero_rot'}
        if self.psf_pars['modality'] == 'ELIPSE_2D':
            self.trainable_pars = {'om_0'}
            
        for d in self.trainable_pars:
            self.psf_pars[d] = torch.nn.Parameter(gpu([self.psf_pars[d]]))
        
    def interpolate_tri(self, wmaps, xos, yos, zs, interpolate=True):

        z_ind = torch.floor(torch.clamp(zs,0,wmaps.shape[0]-2)).long()
        
        if not interpolate:
            
            return wmaps[z_ind]

        zd = zs - z_ind.float()

        xd = xos
        yd = yos
        zd = zd[:,None,None]

        x_p = torch.where(xd>=0,torch.ones_like(xd),torch.zeros_like(xd))
        y_p = torch.where(yd>=0,torch.ones_like(yd),torch.zeros_like(yd))

        p_p = x_p*y_p
        p_n = x_p*(1-y_p)
        n_p = (1-x_p)*y_p
        n_n = (1-x_p)*(1-y_p)

        pad_wmaps = torch.cuda.FloatTensor(*[s + 2 for s in wmaps.shape]).fill_(0)
        pad_wmaps[1:-1,1:-1,1:-1] += wmaps
        pad_wmaps = pad_wmaps[1:-1]
        
        return_maps = torch.zeros_like(pad_wmaps[z_ind,1:-1,1:-1])

        xd = abs(xd)
        yd = abs(yd)
        
        pad_w_imgs = [pad_wmaps[z_ind],pad_wmaps[z_ind+1]]
        
        facs = [(1-xd)*(1 - yd)*(1 - zd),(xd)*(1 - yd)*(1 - zd),(1-xd)*(yd)*(1 - zd),(xd)*(yd)*(1 - zd),(1-xd)*(1 - yd)*(zd),(xd)*(1 - yd)*(zd),(1-xd)*(yd)*(zd),(xd)*(yd)*(zd)]
        z_shift = [0,0,0,0,1,1,1,1]

        # p_p
        
        sls = [
            np.s_[:,1:-1,1:-1],np.s_[:,1:-1,:-2],
            np.s_[:,:-2,1:-1],np.s_[:,:-2,:-2],
            np.s_[:,1:-1,1:-1],np.s_[:,1:-1,:-2],
            np.s_[:,:-2,1:-1],np.s_[:,:-2,:-2],
        ]
        
        for s,f,zs in zip(sls,facs,z_shift):
            return_maps += pad_w_imgs[zs][s] * f * p_p
            
        # p_n
        
        sls = [
            np.s_[:,1:-1,1:-1],np.s_[:,1:-1,:-2],
            np.s_[:,2:,1:-1],np.s_[:,2:,:-2],
            np.s_[:,1:-1,1:-1],np.s_[:,1:-1,:-2],
            np.s_[:,2:,1:-1],np.s_[:,2:,:-2],
        ]

        for s,f,zs in zip(sls,facs,z_shift):
            return_maps += pad_w_imgs[zs][s] * f * p_n

        # np
        
        sls = [
            np.s_[:,1:-1,1:-1],np.s_[:,1:-1,2:],
            np.s_[:,:-2,1:-1],np.s_[:,:-2,2:],
            np.s_[:,1:-1,1:-1],np.s_[:,1:-1,2:],
            np.s_[:,:-2,1:-1],np.s_[:,:-2,2:],
        ]

        for s,f,zs in zip(sls,facs,z_shift):
            return_maps += pad_w_imgs[zs][s] * f * n_p

        # nn
        
        sls = [
            np.s_[:,1:-1,1:-1],np.s_[:,1:-1,2:],
            np.s_[:,2:,1:-1],np.s_[:,2:,2:],
            np.s_[:,1:-1,1:-1],np.s_[:,1:-1,2:],
            np.s_[:,2:,1:-1],np.s_[:,2:,2:],
        ]

        for s,f,zs in zip(sls,facs,z_shift):
            return_maps += pad_w_imgs[zs][s] * f * n_n 

        return return_maps
    
    def transform_offsets(self, S, XYZI, beads=False):
        
        if beads:
            
            n_imgs = S.shape[0]
            n_beads = XYZI[0].shape[-1]
            
            s_inds = S.nonzero()
            xo_vals = XYZI[0].repeat_interleave(n_imgs,0)[:,0].reshape([-1])[:,None,None]
            yo_vals = XYZI[0].repeat_interleave(n_imgs,0)[:,1].reshape([-1])[:,None,None]
            i_vals = XYZI[1].repeat_interleave(n_imgs,0).reshape([-1])[:,None,None]
            zo_vals = XYZI[2].repeat_interleave(n_beads,0)[:,None,None]
            self.wmap_inds = zo_vals[:,0,0] + self.psf_pars['z_scale']
            
        else:
        
            n_samples = S.shape[0] // XYZI.shape[0]
            XYZI_rep = XYZI.repeat_interleave(n_samples, 0)

            s_inds = tuple(S.nonzero().transpose(1,0))
            xo_vals = (XYZI_rep[:,0][s_inds])[:,None, None]
            yo_vals = (XYZI_rep[:,1][s_inds])[:,None, None]
            i_vals = (XYZI_rep[:,3][s_inds])[:,None,None]
            if '3D' in self.psf_pars['modality']:
                zo_vals = self.psf_pars['z_scale'] * XYZI_rep[:,2][s_inds][:,None,None]
                self.wmap_inds = zo_vals[:,0,0] + self.psf_pars['z_scale']
            else:
                zo_vals = (1 + func.softplus(XYZI_rep[:,2][s_inds]))[:,None,None]
                self.wmap_inds = XYZI_rep[:,2][s_inds] + self.psf_pars['z_scale']
        
        return xo_vals, yo_vals, zo_vals, i_vals
        
    def psf_func(self, XO, YO, ZO, I, add_wmap=True):
        
        if add_wmap:
            self.filt_size = self.w_map.shape[-1]
        
        v = torch.arange(self.filt_size) - self.filt_size // 2
        v = v.reshape([1, self.filt_size]).float().cuda()
        
        if self.psf_pars['modality'] == 'GAUSS_2D':
                 
            W_x = (v[None,:,:] - XO) ** 2
            W_y = (v.transpose(1,0)[None,:,:] - YO) ** 2
            W = self.psf_pars['w_prop'] * torch.exp(-(W_x + W_y) / (2 * (ZO * self.psf_pars['width1']) ** 2 + self.taylor_corr)) / (2 * np.pi * (ZO * self.psf_pars['width1']) ** 2 + self.taylor_corr)
            W += (1 - self.psf_pars['w_prop']) * torch.exp(-(W_x + W_y) / (2 * (ZO * self.psf_pars['width2']) ** 2 + self.taylor_corr)) / (2 * np.pi * (ZO * self.psf_pars['width2']) ** 2 + self.taylor_corr)
                
        if self.psf_pars['modality'] == 'ASTIG_3D':
            
            om_x = 0.5 * self.psf_pars['om_0'] * torch.sqrt(1 + ((ZO - self.psf_pars['c_x']) / self.psf_pars['d']) ** 2)
            om_y = 0.5 * self.psf_pars['om_0'] * torch.sqrt(1 + ((ZO - self.psf_pars['c_y']) / self.psf_pars['d']) ** 2)
            W_x = (v[None,:,:] - XO) ** 2
            W_y = (v.transpose(1,0)[None,:,:] - YO) ** 2
            W = torch.exp(-W_x / (2 * (om_x ** 2 + self.taylor_corr))) * torch.exp(-W_y / (2 * (om_y ** 2 + self.taylor_corr))) / (2 * np.pi * (om_x * om_y + self.taylor_corr))       
             
        if self.psf_pars['modality'] == 'HELIX_3D':
            
            x_shift = -self.psf_pars['rad']*torch.cos(self.psf_pars['rot_multi']*ZO + self.psf_pars['zero_rot'])
            y_shift = self.psf_pars['rad']*torch.sin(self.psf_pars['rot_multi']*ZO + self.psf_pars['zero_rot'])   
            
            W1 = torch.sqrt((v[None,:,:] - (XO + x_shift)) ** 2 + (v.transpose(1,0)[None,:,:] - (YO + y_shift)) ** 2)
            W = torch.exp(-W1 ** 2 / (2 * (self.psf_pars['width']) ** 2 + self.taylor_corr)) / (2 * np.pi * (self.psf_pars['width']) ** 2 + self.taylor_corr)
            W2 = torch.sqrt((v[None,:,:] - (XO - x_shift)) ** 2 + (v.transpose(1,0)[None,:,:] - (YO - y_shift)) ** 2)
            W += torch.exp(-W2 ** 2 / (2 * (self.psf_pars['width']) ** 2 + self.taylor_corr)) / (2 * np.pi * (self.psf_pars['width']) ** 2 + self.taylor_corr)
            
            W /= 2            
            
        if self.psf_pars['modality'] == 'ELIPSE_2D':
            
            om_x = ZO * self.psf_pars['om_0'] * (1 + self.psf_pars['ellipticity'])
            om_y = ZO * self.psf_pars['om_0']
            
            W_x = (v[None,:,:] - XO) ** 2
            W_y = (v.transpose(1,0)[None,:,:] - YO) ** 2
            
            W = torch.exp(-W_x / (2 * (om_x ** 2 + self.taylor_corr))) * torch.exp(-W_y / (2 * (om_y ** 2 + self.taylor_corr))) / (2 * np.pi * (om_x * om_y + self.taylor_corr))
                
        if add_wmap: 
            maps = self.interpolate_tri(self.w_map, XO, YO, self.wmap_inds,self.interpolate)
            W += maps
              
        W /= W.sum(-1).sum(-1)[:,None,None] 
        W *= I     
        
        return W
    
    def place_psfs(self, W, S):
            
        recs = torch.zeros_like(S)
        h,w = S.shape[1], S.shape[2]
        
        s_inds = tuple(S.nonzero().transpose(1,0))
        relu = nn.ReLU()
        
        if self.loop_func == 'single':
            
            x_rl = relu(s_inds[1]-self.filt_size//2)
            y_rl = relu(s_inds[2]-self.filt_size//2)

            x_wl = relu(self.filt_size//2-s_inds[1])
            x_wh = self.filt_size-(s_inds[1]+self.filt_size//2-h)-1

            y_wl = relu(self.filt_size//2-s_inds[2])
            y_wh = self.filt_size-(s_inds[2]+self.filt_size//2-w)-1

            for i in range(len(W)):

                w_cut = W[i][x_wl[i] : x_wh[i], y_wl[i] : y_wh[i]]  
                recs[s_inds[0][i],x_rl[i]:x_rl[i]+w_cut.shape[0],y_rl[i]:y_rl[i]+w_cut.shape[1]] += w_cut            
                
        if self.loop_func == 'unique':
            
            r_inds = S.nonzero()[:,1:]
            uni_inds = S.sum(0).nonzero()

            x_rl = relu(uni_inds[:,0]-self.filt_size//2)
            y_rl = relu(uni_inds[:,1]-self.filt_size//2)

            x_wl = relu(self.filt_size//2-uni_inds[:,0])
            x_wh = self.filt_size-(uni_inds[:,0]+self.filt_size//2-h)-1

            y_wl = relu(self.filt_size//2-uni_inds[:,1])
            y_wh = self.filt_size-(uni_inds[:,1]+self.filt_size//2-w)-1

            r_inds_r = h*r_inds[:,0] + r_inds[:,1]
            uni_inds_r = h*uni_inds[:,0] + uni_inds[:,1]

            for i in range(len(uni_inds)):

                curr_inds = torch.nonzero(r_inds_r == uni_inds_r[i])[:,0]
                w_cut = W[curr_inds, x_wl[i] : x_wh[i], y_wl[i] : y_wh[i]]

                recs[s_inds[0][curr_inds],x_rl[i]:x_rl[i]+w_cut.shape[1],y_rl[i]:y_rl[i]+w_cut.shape[2]] += w_cut
                
        return recs
    
    def genfunc(self, S, XYZI, add_wmap=True):
        
        XO, YO, ZO, I = self.transform_offsets(S, XYZI, self.beads_fit)
        W = self.psf_func(XO, YO, ZO, I, add_wmap)
        return 1000 * self.psf_pars['scale'] * self.place_psfs(W, S)
    
    def datagen_func(self, s, xo, yo, zo, ints, bg, add_noise=True):
        
        batch_size,n_inp, w, h = s.shape[0], s.shape[1], s.shape[2], s.shape[3]
        xyzi = torch.cat([xo.reshape([-1, 1, h, w]), yo.reshape([-1, 1, h, w]),zo.reshape([-1, 1, h, w]), ints.reshape([-1, 1, h, w])], 1)
        recs = self.genfunc(s.reshape([-1, h, w]), xyzi)
        torch.clamp_min_(recs,0)
        x_sim = recs.reshape([batch_size, n_inp, h, w]) 
        if 'backg_max' in self.ll_pars: 
            x_sim += bg[:,None, None, None] * self.ll_pars['backg_max'] + self.ll_pars['baseline']
        else: 
            x_sim += self.ll_pars['backg']
                
        if add_noise:
            
            conc = (x_sim - self.ll_pars['baseline'])/self.ll_pars['theta']
            x_sim = torch.distributions.Gamma(conc, 1/self.ll_pars['theta']).sample() + self.ll_pars['baseline']

        return x_sim
    
    def draw_func(self, M, batch_size=1, local_context=False):

        λ = self.ll_pars['p_act'] * torch.tanh(M / self.ll_pars['p_act']).to('cuda')
        λ = λ.reshape(1,1,λ.shape[-2], λ.shape[-1]).repeat_interleave(batch_size, 0)
        
        locs1 = torch.distributions.Binomial(1, λ).sample().to('cuda')
        
        zeros = torch.zeros_like(locs1).to('cuda')

        z = torch.distributions.Normal(zeros+self.ll_pars['z_prior'][0], zeros+self.ll_pars['z_prior'][1]).sample().to('cuda')
        if '3D' in self.psf_pars['modality']:
            torch.tanh_(z)
            
        x = torch.distributions.Uniform(zeros-0.5, zeros+0.5).sample().to('cuda')
        y = torch.distributions.Uniform(zeros-0.5, zeros+0.5).sample().to('cuda')
        
        if 'backg_max' in self.ll_pars:
            bg = torch.distributions.Uniform(torch.zeros(batch_size).to('cuda')+0.01, torch.ones(batch_size).to('cuda')-0.01).sample().to('cuda')
        else:
            bg = None

        if local_context:
        
            α = self.ll_pars['surv_p']
            a11 = 1 - (1 - λ) * (1 - α)

            locs2 = torch.distributions.Binomial(1, (1 - locs1) * λ + locs1 * a11).sample().to('cuda')
            locs3 = torch.distributions.Binomial(1, (1 - locs2) * λ + locs2 * a11).sample().to('cuda')

            locs = torch.cat([locs1, locs2, locs3], 1)
            
            x = x.repeat_interleave(3,1)
            y = y.repeat_interleave(3,1)
            z = z.repeat_interleave(3,1)
     
        else:
            
            locs = locs1            
        
        ints = torch.distributions.Uniform(torch.zeros_like(locs)+self.ll_pars['min_int'], torch.ones_like(locs)).sample().to('cuda')

        x *= locs
        y *= locs
        z *= locs
        ints *= locs
        
        return locs,x,y,z,ints,bg