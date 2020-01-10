import torch
import torch.nn as nn
import torch.nn.functional as func
from utils import *
import numpy as np

from torch import distributions as D
from GMM import *
     
class Funcs:
    
    def eval_p_x_z(self, X, F, BG, noise='gamma'):

        target = X[:, X.shape[1]//2][:,None]
        pred = F

        if BG is not None:

            bg_facs = BG.mean(-1).mean(-1)
            bg_map = BG.mean(0)
            bg_map /= bg_map.mean(-1).mean(-1)
            BG_res = bg_map[None,:,:] * bg_facs[:,None,None]

            pred = pred.reshape([self.batch_size, self.n_samples, target.shape[-2], target.shape[-1]]) + BG_res[:,None] * self.ll_pars['backg_max'] + self.ll_pars['baseline']

        else:

            pred = pred.reshape([self.batch_size, self.n_samples, target.shape[-2], target.shape[-1]]) + self.ll_pars['backg']

        if noise == 'poisson':

            return (target * torch.log(pred + 1e-6) - pred - torch.lgamma(target+1)).sum(-1).sum(-1)

        if noise == 'gamma':

            target = torch.clamp(target-self.ll_pars['baseline'],1,np.inf)
            pred = torch.clamp(pred-self.ll_pars['baseline'],1,np.inf)
            k = pred/self.ll_pars['theta']

            return ((k-1)*torch.log(target) - target/self.ll_pars['theta'] - k*torch.log(self.ll_pars['theta']) - torch.lgamma(k)).sum(-1).sum(-1)    

    def eval_q_z_x_sl(self,P, XYZI_m,XYZI_s,XYZI_mat,S_mask):
            
        log_prob = 0
 
        P = torch.sigmoid(P)

        prob_mean = P.sum(-1).sum(-1)

        prob_var = (P - P**2).sum(-1).sum(-1)
        prob_gauss = D.Normal(prob_mean, torch.sqrt(prob_var))
        log_prob += prob_gauss.log_prob(S_mask.sum(-1)) * S_mask.sum(-1)

        prob_normed = P/(P.sum(-1).sum(-1)[:,None,None])

        p_inds = tuple((P+1).nonzero().transpose(1,0))

        xyzi_mu = XYZI_m[p_inds[0],:,p_inds[1],p_inds[2]]
        xyzi_mu[:,0] += p_inds[2].type(torch.cuda.FloatTensor) + 0.5
        xyzi_mu[:,1] += p_inds[1].type(torch.cuda.FloatTensor) + 0.5

        xyzi_mu = xyzi_mu.reshape(self.batch_size,-1,4)
        xyzi_sig = XYZI_s[p_inds[0],:,p_inds[1],p_inds[2]].reshape(self.batch_size,-1,4)

        mix = D.Categorical(prob_normed[p_inds].reshape(self.batch_size,-1))
        comp = D.Independent(D.Normal(xyzi_mu, xyzi_sig), 1)
        gmm = MixtureSameFamily(mix, comp)

        gmm_log = gmm.log_prob(XYZI_mat.transpose(0,1)).transpose(0,1)
        gmm_log = (gmm_log * S_mask).sum(-1)

        log_prob += gmm_log 

        log_prob = log_prob.reshape(self.batch_size, 1)           
                                    
        return log_prob
    
    def eval_q_z_x_ae(self, S, P):
        
        n_samples = self.n_samples         
        log_prob = 0
            
        prob = P[:,None].repeat_interleave(n_samples,1)
        samp = S.reshape((self.batch_size, n_samples, S.shape[-2], S.shape[-1]))

        loss = nn.BCEWithLogitsLoss(reduction='none')
        log_prob -= loss(prob, samp).sum(-1).sum(-1)             

        return log_prob

    def eval_p_z(self, S, M, C=None):

        λ = self.ll_pars['p_act'] * torch.tanh(M / self.ll_pars['p_act'])
        λ = λ[:,None]
        λ = λ.repeat_interleave(self.n_samples,1)

        if C is None:
            prior_map = λ
        else:
            α = self.ll_pars['surv_p']
            a10 = λ                      # 1|0
            a11 = 1 - (1 - λ) * (1 - α)  # 1|1

            pre = C[:,0].reshape((self.batch_size, self.n_samples, S.shape[-2], S.shape[-1]))
            prior_map = (1 - pre) * a10 + pre * a11

        torch.clamp_(prior_map, 0.0001, 0.9999)
        samp = S.reshape((self.batch_size, self.n_samples, S.shape[-2], S.shape[-1]))

        loss = nn.BCELoss(reduction='none')

        return -loss(prior_map, samp).sum(-1).sum(-1)

    def eval_bg_sq_loss(self, BG_pred, BG_true):

        loss = nn.MSELoss(reduction='none')

        cost = loss(BG_pred, BG_true[:,None,None])
        cost = cost.sum(-1).sum(-1)

        return cost

    def eval_cs_align_cost(self, S, XYZI_m, C, CC, CCs):

        cost = 0

        CCs = torch.detach(CCs)
        CC = torch.detach(CC)

        samp = S.reshape((self.batch_size, self.n_samples, 1, S.shape[-2], S.shape[-1]))

        pre_s = C[:,0].reshape((self.batch_size, self.n_samples, 1, S.shape[-2], S.shape[-1]))
        post_s = C[:,1].reshape((self.batch_size, self.n_samples, 1, S.shape[-2], S.shape[-1]))
        pre_xyz = CC[:,0:3]
        post_xyz = CC[:,4:7]
        pre_xyzs = CCs[:,0:3]
        post_xyzs = CCs[:,4:7]

        mask_pre = (pre_s * samp).repeat_interleave(3,2)
        mask_post = (post_s * samp).repeat_interleave(3,2)

        dist_pre = torch.distributions.normal.Normal(pre_xyz,pre_xyzs) 
        dist_post = torch.distributions.normal.Normal(post_xyz,post_xyzs) 

        log_pre = dist_pre.log_prob(XYZI_m[:,:3])[:,None]
        log_post = dist_post.log_prob(XYZI_m[:,:3])[:,None]

        cost += (mask_pre* log_pre).sum(2)
        cost += (mask_post * log_post).sum(2)

        cost =-cost.sum(-1).sum(-1)

        return cost

    def elbo_loss(self,X,P,S,XYZI_m,XYZI_s,F,M,BG,CS=None,CP=None,CC=None,CCs=None):

        log_px_given_z = self.eval_p_x_z(X, F, BG)
        log_qd_given_x = self.eval_q_z_x_ae(S, P)
        log_pz_d = self.eval_p_z(S, M, CS)

        log_weight = log_px_given_z + self.ll_pars['prior_fac']*log_pz_d - log_qd_given_x
        log_p = log_px_given_z + self.ll_pars['prior_fac']*log_pz_d

        total_cost = 0

        with torch.no_grad():
            log_f_i = log_px_given_z + self.ll_pars['prior_fac']*log_pz_d - log_qd_given_x
            omega_i = torch.exp(log_f_i - log_f_i.logsumexp(-1,keepdim=True))

        total_cost -= torch.mean((omega_i * (log_px_given_z + log_qd_given_x)).sum(-1,keepdim=True))

        if CP is not None:

            align_cost = self.eval_cs_align_cost(S, XYZI_m, CS, CC, CCs)
            total_cost += torch.mean(align_cost)

        return total_cost

    def simu_loss(self,P,XYZI_m,XYZI_s, xyzi_mat, s_mask, BG, BG_true):

        log_qd_given_x = self.eval_q_z_x_sl(P, XYZI_m,XYZI_s, xyzi_mat, s_mask)
        bg_sq_error = self.eval_bg_sq_loss(BG, BG_true) if BG is not None else 0

        total_cost = torch.mean(bg_sq_error - log_qd_given_x[:,0])

        return total_cost