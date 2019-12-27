import numpy as np
import torch 
import csv
import copy
import scipy.stats as stats   
from matplotlib import pyplot as plt

def gpu(x):
    '''Transforms numpy array or torch tensor it torch.cuda.FloatTensor'''
    if isinstance(x, np.ndarray):
        return torch.cuda.FloatTensor(x.astype('float32'))
    else:
        return torch.cuda.FloatTensor(x)

def cpu(x):
    '''Transforms torch tensor into numpy array'''
    return x.cpu().detach().numpy()

def softp(x):
    '''Returns softplus(x)'''
    return(np.log(1+np.exp(x)))

def sigmoid(x):
    '''Returns sigmoid(x)'''
    return 1 / (1 + np.exp(-x))

def inv_softp(x):
    '''Returns inverse softplus(x)'''
    return np.log(np.exp(x)-1)

def inv_sigmoid(x):
    '''Returns inverse sigmoid(x)'''
    return -np.log(1/x-1)

def torch_arctanh(x):
    '''Returns arctanh(x) for tensor input'''
    return 0.5*torch.log(1+x) - 0.5*torch.log(1-x)

def torch_softp(x):
    '''Returns softplus(x) for tensor input'''
    return (torch.log(1+torch.exp(x)))

def flip_filt(filt):
    '''Returns filter flipped over x and y dimension'''
    return np.ascontiguousarray(filt[...,::-1,::-1])

def get_bg_stats(images, percentile=10,plot=False,xlim=None, floc=0):
    """Infers the parameters of a gamma distribution that fit the background of SMLM recordings. 
    Identifies the darkest pixels from the averaged images as background and fits a gamma distribution to the histogram of intensity values.
    
    Parameters
    ----------
    images: array
        3D array of recordings
    percentile: float
        Percentile between 0 and 100. Sets the percentage of pixels that are assumed to only containg background activity (i.e. no fluorescent signal)
    plot: bool
        If true produces a plot of the histogram and fit
    xlim: list of floats
        Sets xlim of the plot
    floc: float
        Baseline for the the gamma fit. Equal to fitting gamma to (x - floc)
        
    Returns
    -------
    mean, scale: float
        Mean and scale parameter of the gamma fit
    """     
    map_empty = np.where(images.mean(0) < np.percentile(images.mean(0),percentile))
    pixel_vals = images[:,map_empty[0],map_empty[1]].reshape(-1)
    fit_alpha, fit_loc, fit_beta=stats.gamma.fit(pixel_vals,floc=floc)
    
    if plot:
        if xlim is None: 
            low,high = pixel_vals.min(),pixel_vals.max()
        else:
            low,high = xlim[0],xlim[1]
        
        _ = plt.hist(pixel_vals,bins=np.linspace(low,high),  histtype ='step',label='data')
        _ = plt.hist(np.random.gamma(fit_alpha,fit_beta,size=len(pixel_vals))+floc,bins=np.linspace(low,high),  histtype ='step',label='fit')     
        plt.xlim(low,high)
        plt.legend()
        plt.show()
    return fit_alpha*fit_beta,fit_beta

def get_window_map(img, winsize=40, percentile=20):
    """Helper function 
    
    Parameters
    ----------
    images: array
        3D array of recordings
    percentile: float
        Percentile between 0 and 100. Sets the percentage of pixels that are assumed to only containg background activity (i.e. no fluorescent signal)
    plot: bool
        If true produces a plot of the histogram and fit
    xlim: list of floats
        Sets xlim of the plot
    floc: float
        Baseline for the the gamma fit. Equal to fitting gamma to (x - floc)
        
    Returns
    -------
    binmap: array
        Mean and scale parameter of the gamma fit
    """      
    img = img.mean(0)
    res = np.zeros([int(img.shape[0]-winsize), int(img.shape[1]-winsize)])
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i,j] = img[i:i+int(winsize), j:j+int(winsize)].mean()
    thresh = np.percentile(res,percentile)
    binmap = np.zeros_like(res)
    binmap[res>thresh] = 1
    return binmap