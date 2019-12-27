import torch
from utils import *
import numpy as np
import torch.optim as optim
from IPython.display import display, clear_output

def get_peaks(image,threshold=500,min_distance=20, double_helix=False):
    """Peak finding functions. Provides position estimate for bead stacks that are used as initialization for PSF fitting.
    
    Parameters
    ----------
    image: 2D array
        Single bead recording
    threshold: float
        Initial threshold to identify pixel that are considered as possible peaks. 
    min_distance: float
        Minimal distance between two peaks in pixels
    double_helix fact: bool
        If true adjusts the fitting procedure to work with double helix data. 
        
    Returns
    -------
    peaks: array
        Array of x,y peak positions
    """            
    peaks = []
    t_img = np.where(image>threshold,image,0)
    inds = t_img.nonzero()
    vals = t_img[inds]
    inds_yx = [[y,x] for _,y,x in sorted(zip(vals,inds[0],inds[1]))][::-1]    
    
    while len(inds_yx) > 0:
        
        valid = True
        yx = inds_yx[0]
        y,x = yx
        inds_yx.remove(yx)
        
        for pyx in peaks:
            if np.sqrt((y-pyx[0])**2 + (x-pyx[1])**2) < min_distance:
                valid = False
                break
        if valid:       
            peaks.append(yx)
            
    if double_helix:
        
        dh_peaks = []
        while len(peaks)>1:
            yx = np.array(peaks[0])
            del(peaks[0])
            min_ind = 0

            min_dist = np.inf
            for i in range(len(peaks)):
                dist = np.sqrt((peaks[i][0]-yx[0])**2 + (peaks[i][1]-yx[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    min_ind = i
            dh_peaks.append((yx + np.array(peaks[min_ind]))/2)
            del(peaks[min_ind])
        peaks = dh_peaks
         
    return np.array(peaks)[:,::-1]

def sort_true_locs(true_locs, peaks):
    """Matches ground truth positions to peaks.
    
    Parameters
    ----------
    true_locs: 2D array
        Ground truth bead locations
    peaks: 2D array
        Array of x,y peak positions provided by get_peaks. 
        
    Returns
    -------
    matched_locs: 2D array
        Ground truth positions ordered to match the peaks
    """     
    matched_locs = []

    for pp in peaks:
        min_dist = np.inf
        matched_locs.append(None)
        for tp in true_locs:
            dist = ((100*np.array(pp) - np.array(tp))**2).sum()
            if dist < min_dist:
                matched_locs[-1] = copy.deepcopy(tp)
                min_dist = dist    
                
    return np.array(matched_locs)

def StormIterator(print_freq, batch_size, imgs, zos):
    """ Basic mini-batch iterator """
    for _ in range(print_freq):
        choice = np.random.choice(np.arange(len(imgs)), batch_size, replace=False)
        yield np.array([imgs[c] for c in choice]), np.array([zos[c] for c in choice])

def set_optimizers(model, lr):
    """Creates optimizers and learning rate schedulers to train the generative model parameters and emitter locations. 
    
    Parameters
    ----------
    model: LikelihoodModel
    lr: Learning rate used for stochastic gradient descent.
    """       
    model.optimizer_gen = optim.Adam([model.psf_pars[d] for d in model.trainable_pars], lr=model.lr)
    model.optimizer_wmap = optim.Adam([model.w_map], lr=0.0002*model.lr)
    model.optimizer_locs = optim.Adam([model.XY, model.I], lr=model.lr)

    model.scheduler_gen = torch.optim.lr_scheduler.StepLR(model.optimizer_gen, step_size=100, gamma=0.95)
    model.scheduler_wmap = torch.optim.lr_scheduler.StepLR(model.optimizer_wmap, step_size=100, gamma=0.95)
    model.scheduler_locs = torch.optim.lr_scheduler.StepLR(model.optimizer_locs, step_size=100, gamma=0.90)
    
    model._iter_count = 0
    
def train_beads(model, x, s, z_true):
    """Performs one step of iteration of gradient descent for PSF fitting on beads. 
    
    Parameters
    ----------
    model: LikelihoodModel
    x: 3D array
        Batch of bead recordings
    s: 3D array
        Discrete estimate of bead positions. These stay fixed during training, continues offsets are trained to fit the bead locations exactly. 
    z_true: 1D array
        Ground truth z values (in 100 nano meter) corresponding to the current batch.
        
    Returns
    -------
    p_x_y: float
        Current loss
    """        
    F = model.genfunc(gpu(s), [model.XY, model.I, gpu(z_true)], add_wmap = True) 
    
    model.optimizer_gen.zero_grad(); model.optimizer_locs.zero_grad(); model.optimizer_wmap.zero_grad()
    
    p_x_z = model.eval_p_x_z(gpu(x), F, BG=None, noise='poisson').mean()

    loss = -p_x_z + 1e4 * torch.norm(model.w_map.sum(-1).sum(-1), 1)
    loss.backward()
    
    model.optimizer_locs.step(); model.optimizer_gen.step();
    model.scheduler_gen.step(); model.scheduler_locs.step(); 

    model.optimizer_wmap.step()
    model.scheduler_wmap.step()
        
    model._iter_count += 1
    
    return p_x_z.detach()

def get_rmses(mgen, peaks, true_locs, print_output=True):
    """Evaluates the inferred bead localizations if ground truth positions are available (challenge data)
    Provides two measurements:
    RMSE: average distance of the inferred locations to ground truth in nano meter
    Corrected RMSE: average distance after correcting for a constant shift in x and y directions.
    This is the more meaningful metric as the absolute position cannot be inferred from the recordings. 
    
    Parameters
    ----------
    mgen: LikelihoodModel
    peaks: 3D array
        Discrete estimate of bead positions. These stay fixed during training, continues offsets are trained to fit the bead locations exactly. 
    true_locs: 2D array
        Ground truth bead locations
    print_output: bool
        If True prints output. 
        
    """       
    rmse = tot_rmse(true_locs, 100*(cpu(mgen.XY)[0].T + peaks))
    rmse_corr = shifted_rmse(true_locs, 100*(cpu(mgen.XY)[0].T + peaks))[0]
    
    mgen.col_dict['RMSE'][mgen._iter_count] = rmse
    mgen.col_dict['RMSE_corr'][mgen._iter_count] = rmse_corr
    
    if print_output:
        display('RMSE: ' + str(rmse) + ' Corrected RMSE: ' +  str(rmse_corr))
        
def tot_rmse(true_locs, pred_locs):
    return np.sqrt((true_locs - pred_locs)**2).mean()

def shifted_rmse(true_locs, pred_locs):
    
    abs_err = true_locs - pred_locs
    x_shift = abs_err[:,0].mean()
    y_shift = abs_err[:,1].mean()

    pred_locs[:,0] += x_shift
    pred_locs[:,1] += y_shift
    
    return tot_rmse(true_locs, pred_locs), x_shift, y_shift