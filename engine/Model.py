from LikelihoodModel import *
import collections
import torch.optim as optim 
import time
import sys
import pickle

from eval_funcs import *
from utils import *
from nnet import *

import _loss_funcs
import _rec_funcs
import _train_funcs

class Model(_loss_funcs.Funcs, _train_funcs.Funcs, _rec_funcs.Funcs):
    
    def __init__(self, rec_pars, psf_pars, ll_pars):
        
        """DECODE Model

        This is the main class for a DECODE model

        Parameters
        ----------
        rec_pars : dict
            Dictionary of parameters for the recognition model
        psf_pars : dict
           Dictionary of parameters for the Point Spread Function
        ll_pars : dict
            Dictionary of parameters for the Generative Model
        """
        
        self.rec_pars = rec_pars
        self.local_context = rec_pars['local_context']
        self.global_context = rec_pars['global_context']
        self.sig_pred = rec_pars['sig_pred']
        self.bg_pred = rec_pars['bg_pred']
        self.modality = psf_pars['modality']
        self.n_filters = rec_pars['n_filters']
        
        self.n_inp = 3 if self.local_context else 1
        n_features = self.n_filters*(self.n_inp + 1*int(self.global_context))
        self.single_net = SUNNet(1,self.n_filters,2).to(torch.device('cuda'))
        self.comb_net = SUNNet(n_features,self.n_filters,2).to(torch.device('cuda'))
        self.out_net = Out_net(self.n_filters, self.sig_pred, self.bg_pred)
        
        self.mgen = LikelihoodModel(psf_pars)
        self.mgen.ll_pars = ll_pars
        self.ll_pars = ll_pars
        
        self.train_mode = 'co'
        
        self.lr_decay = 0.9
        self.ae_norm = 0.03
        self.sl_norm = 0.03
        self.warm_up = 1000
        self.fixed_psf = False
        self.sim_iters = 5
        
        self.wobble = [0,0]

        self.filename = None
        self.exp_params = None
        self.description = None  
        
        self.col_dict = {}
        self._iter_count = 0
        
    def init_dicts(self):
        
        self.col_dict['exp_params'] = self.exp_params
        self.col_dict['cost_hist'] = collections.OrderedDict([])
        self.col_dict['update_time'] = collections.OrderedDict([])
        self.col_dict['n_per_img'] = collections.OrderedDict([])
        
        self.col_dict['cost_sl'] = collections.OrderedDict([])
        self.col_dict['cost_ae'] = collections.OrderedDict([])

        if self.eval_csv is not None:
         
            self.col_dict['factor'] = collections.OrderedDict([])
            self.col_dict['recall'] = collections.OrderedDict([])
            self.col_dict['precision'] = collections.OrderedDict([])
            self.col_dict['jaccard'] = collections.OrderedDict([])
            self.col_dict['rmse_lat'] = collections.OrderedDict([])
            self.col_dict['jor'] = collections.OrderedDict([])
            self.col_dict['eff_lat'] = collections.OrderedDict([])
            self.col_dict['eff_ax'] = collections.OrderedDict([])
            self.col_dict['eff_3d'] = collections.OrderedDict([])
            
    def eval_func(self, imgs):
            
        arr_infs = decode_func(self,imgs,len(imgs),z_scale=self.mgen.psf_pars['z_scale'], int_scale=self.mgen.psf_pars['scale'],use_tqdm=False) 
            
        if self.eval_csv is not None:
            
            nms_sampling(arr_infs, threshold=0.7, batch_size=len(imgs), nms=True, nms_cont=True)
            preds = array_to_list(arr_infs, wobble=self.wobble)
            
            if self.train_mode in ('sl','co'):
                preds = filt_preds(preds,95)

            tol_ax = 500 if '3D' in self.mgen.psf_pars['modality'] else np.inf
            match_dict, _ = matching(self.eval_csv, preds, print_res=False, min_int=False, tolerance_ax=tol_ax)
            
            for k in self.col_dict.keys():
                if k in match_dict:
                    self.col_dict[k][self._iter_count] = match_dict[k]
                    
        self.col_dict['n_per_img'][self._iter_count] = arr_infs['Probs'].sum(-1).sum(-1).mean()
    
    
    def fit(self, trainf=None, batch_size=15,n_samples=20, win_size=40, max_iters=50000, learning_rate=5e-4, print_output=True, print_freq=100):
        """Trains the model
        
        Parameters
        ----------
        trainf : numpy array
            Training data when performing combined learning
        batch_size: int
            Batch size for stochastic gradient descent
        n_samples: int
            Number of samples for reweighted wake updates
        win_size : int
            Size of the images used for training
        max_iters: int
            Number of training iterations 
        learning_rate: float
            Learning rate
        print_output: bool
            If False will not output evaluation of the training progress
        print_freq:
            Number of iterations between evaluations of the training progress
        """
        self.batch_size = batch_size
        self.win_size = win_size
        self.n_samples = n_samples
        
        self.init_dicts()
        self.print_freq = print_freq

        self.lr = learning_rate
        self.sl_3d = np.index_exp[:, :, :]
        
        if trainf is None:
            map_ini = np.ones([self.win_size,self.win_size])
        else:
            map_ini = trainf.mean(0) - trainf.mean(0).min()
            
        map_ini /= map_ini.sum()
        map_ini *= ((self.ll_pars['p_act'] * self.ll_pars['p_lambda'] * trainf[0].size))
        
        win_inds = self.window_map.nonzero()

        self.train_map = [map_ini.astype('float32')]
        if self.global_context: self.train_hbar = [np.zeros_like(trainf[:self.n_filters])]

        ''' TRAINING '''
        last_print = 0
        tot_t = 0

        self.net_pars = list(self.single_net.parameters()) + list(self.comb_net.parameters()) + list(self.out_net.parameters())        
        self.optimizer_rec = torch.optim.AdamW(self.net_pars, lr=self.lr, weight_decay=0.1)

        if not self.fixed_psf:
            self.optimizer_gen = torch.optim.AdamW([self.mgen.psf_pars[d] for d in self.mgen.trainable_pars], lr=25*self.lr)
            self.optimizer_wmap = torch.optim.AdamW([self.mgen.w_map], lr=0.005 *self.lr)
        
        self.scheduler_rec = torch.optim.lr_scheduler.StepLR(self.optimizer_rec, step_size=1000, gamma=0.9)
        
        while self._iter_count < max_iters:

            t0 = time.time()
            tot_cost = []
            
            if self.train_mode in ('us','co'):
                
                Iterator = self.storm_iterator(trainf, self.print_freq)

                for x, x_m1, x_p1, ch in Iterator:

                    rand = np.random.randint(0,len(win_inds[0]))
                    y_off,x_off = win_inds[0][rand],win_inds[1][rand]
                    self.sl_3d = np.index_exp[:, y_off:(y_off + self.win_size),x_off:(x_off + self.win_size)]
                    self.sl_4d = (slice(None, None, None),) + self.sl_3d

                    if self.train_mode == 'co' or self._iter_count < self.warm_up:

                        loss = self.train_sl()
                        self.col_dict['cost_sl'][self._iter_count] = cpu(loss)

                    if self._iter_count >= self.warm_up:

                        loss = self.train_ae(x, x_m1, x_p1)
                        tot_cost.append(cpu(loss))
                        self.col_dict['cost_ae'][self._iter_count] = cpu(loss)
                
            else:
                
                for _ in range(self.print_freq):
                    
                    rand = np.random.randint(0,len(win_inds[0]))
                    y_off,x_off = win_inds[0][rand],win_inds[1][rand]
                    self.sl_3d = np.index_exp[:, y_off:(y_off + self.win_size),x_off:(x_off + self.win_size)]
                    
                    loss = self.train_sl()
                    tot_cost.append(cpu(loss))                  
                    
            tot_t += (time.time() - t0)

            self.col_dict['cost_hist'][self._iter_count] = np.mean(tot_cost)
            updatetime = 1000 * (tot_t) / (self._iter_count - last_print)
            last_print = self._iter_count
            tot_t = 0

            ''' EVALUATION '''

            self.eval_func(self.eval_imgs)
            self.col_dict['update_time'][self._iter_count] = updatetime

            if print_output:
                
                if self.eval_csv is not None:
                
                    print('{}{:0.3f}'.format('JoR: ', float(self.col_dict['jor'][self._iter_count])), end='')
                    if '3D' not in self.mgen.psf_pars['modality']:
                        print('{}{}{:0.3f}'.format(' || ', 'Eff_lat: ', self.col_dict['eff_lat'][self._iter_count]), end='')
                    else:
                        print('{}{}{:0.3f}'.format(' || ', 'Eff_3d: ', self.col_dict['eff_3d'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'Jaccard: ', self.col_dict['jaccard'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'Factor: ', self.col_dict['n_per_img'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'RMSE: ', self.col_dict['rmse_lat'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'Cost: ', self.col_dict['cost_hist'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'Recall: ', self.col_dict['recall'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'Precision: ', self.col_dict['precision'][self._iter_count]), end='')
                    print('{}{}{:0.1f}'.format(' || ', 'Time Upd.: ', float(updatetime), ' ms '), end='')
                    print('{}{}{}'.format(' || ', 'BatchNr.: ', self._iter_count))
                    
                else:
                    
                    print('{}{:0.3f}'.format( 'Factor: ', self.col_dict['n_per_img'][self._iter_count]), end='')
                    print('{}{}{:0.3f}'.format(' || ', 'Cost: ', self.col_dict['cost_hist'][self._iter_count]), end='')
                    print('{}{}{:0.1f}'.format(' || ', 'Time Upd.: ', float(updatetime), ' ms '), end='')
                    print('{}{}{}'.format(' || ', 'BatchNr.: ', self._iter_count))

                sys.stdout.flush()

            if self.filename:
                self.col_dict['description'] = self.description
                with open(self.filename+'.pkl', 'wb') as f:
                    pickle.dump(self, f)
                with open(self.filename + '_dicts.pkl', 'wb') as f:
                    pickle.dump(self.col_dict, f)