from matplotlib import pyplot as plt
from matplotlib import gridspec
import pickle
import numpy as np
import seaborn as sns
import os

from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv
import cv2
import PIL
from PIL import ImageEnhance

def plot_od(od, label=None, col=None):
    """Produces a line plot from a ordered dictionary as used to store training process in the Model class
    
    Parameters
    ----------
    od: OrderedDict of floats
        DECODE model
    label: str
        Label
    col: 'str'
        Color
    """         
    plt.plot(*zip(*sorted(od.items())), label = label, color = col)

def create_3d_hist(preds, z_clip=None, pix_size=5, sigma=3, contrast_fac=10, clip_density=100): 
    """Produces a coloured histogram to display 3D reconstructions.
    
    Parameters
    ----------
    preds: list
        List of localizations with columns: 'localization', 'frame', 'x', 'y', 'z'
    z_clip: list of ints
        Clips the the z values at the given lower and upper limit to control the colorrange. 
    pix_size: float
        Size of the pixel (nano meter) in the reconstruction plot
    sigma:
        Size of Gaussian used for blurring
    constrast fact: float
        Contrast can be scaled with this variable if the output image is to bright/dark
    clip_density: float 
        Percentile between 0 and 100. Artifacts that produce extremely dense regions in the histrogram can 
        mess up the contrast scaling. This parameter can be used to exclude the brightest regions. 
        
    Returns
    -------
    Image: PIL image
        Coloured histogram of 3D reconstruction
    """                 
    # adjust colormap
    lin_hue = np.linspace(0,1,256)
    cmap=plt.get_cmap('jet', lut=256);
    cmap = cmap(lin_hue)
    cmap_hsv = rgb_to_hsv(cmap[:,:3])
    storm_hue = cmap_hsv[:,0]
    _,b = np.unique(storm_hue, return_index=True)
    storm_hue = [storm_hue[index] for index in sorted(b)]
    n_val = len(storm_hue)
    storm_hue = np.interp(np.linspace(0,n_val,256), np.arange(n_val), storm_hue)
    
    x_pos = np.clip(np.array(preds)[:,2],0,np.inf)
    y_pos = np.clip(np.array(preds)[:,3],0,np.inf)
    z_pos = np.array(preds)[:,4] 
    
    min_z = min(z_pos)
    max_z = max(z_pos)

    if z_clip is not None:
        z_pos[z_pos<z_clip[0]] = z_clip[0]
        z_pos[z_pos>z_clip[1]] = z_clip[1]
        zc_val = (z_pos -z_clip[0] ) / (z_clip[1] - z_clip[0])

    else:
        zc_val = (z_pos - min_z) / (max_z - min_z)

    z_hue = np.interp(zc_val,lin_hue,storm_hue)

    nx = int((np.max(x_pos))//pix_size+1)
    ny = int((np.max(y_pos))//pix_size+1)
    dims = (nx,ny)
    
    x_vals = np.array(x_pos//pix_size, dtype='int')
    y_vals = np.array(y_pos//pix_size, dtype='int')
    
    lin_idx = np.ravel_multi_index((x_vals, y_vals), dims)
    density = np.bincount(lin_idx, weights=np.ones(len(lin_idx)), minlength=np.prod(dims)).reshape(dims)
    density = np.clip(density,0,np.percentile(density,clip_density))
    zsum = np.bincount(lin_idx, weights=z_hue, minlength=np.prod(dims)).reshape(dims)
    zavg = zsum/density
    zavg[np.isnan(zavg)]=0

    hue = zavg[:,:,None]
    sat = np.ones(density.shape)[:,:,None]
    val = (density/np.max(density))[:,:,None]
    sr_HSV = np.concatenate((hue,sat,val),2)
    sr_RGB = hsv_to_rgb(sr_HSV)
    # %have to gaussian blur in rgb domain
    sr_RGBblur = cv2.GaussianBlur(sr_RGB,(11,11),sigma/pix_size)
    sr_HSVblur = rgb_to_hsv(sr_RGBblur)

    val = sr_HSVblur[:,:,2]

    sr_HSVfinal = np.concatenate((sr_HSVblur[:,:,:2],val[:,:,None]),2)
    sr_RGBfinal= hsv_to_rgb(sr_HSVfinal)
    
    sr_Im = PIL.Image.fromarray(np.array(np.round(sr_RGBfinal*256), dtype='uint8'))
    enhancer = ImageEnhance.Contrast(sr_Im)
    sr_Im = enhancer.enhance(contrast_fac)

    return sr_Im.transpose(PIL.Image.TRANSPOSE)

def create_2d_hist(preds, pix_size=5, sigma=3, contrast_fac=2, clip_density=100):   
    """Produces a coloured histogram to display 3D reconstructions.
    
    Parameters
    ----------
    preds: list
        List of localizations with columns: 'localization', 'frame', 'x', 'y', 'z'
    pix_size: float
        Size of the pixel (nano meter) in the reconstruction plot
    sigma:
        Size of Gaussian used for blurring
    constrast fact: float
        Contrast can be scaled with this variable if the output image is to bright/dark
    clip_density: float 
        Percentile between 0 and 100. Artifacts that produce extremely dense regions in the histrogram can 
        mess up the contrast scaling. This parameter can be used to exclude the brightest regions. 
        
    Returns
    -------
    sr_blur: array
        Histogram of 2D reconstruction
    """            

    x_pos = np.clip(np.array(preds)[:,2],0,np.inf)
    y_pos = np.clip(np.array(preds)[:,3],0,np.inf) 
    
    nx = int((np.max(x_pos))//pix_size+1)
    ny = int((np.max(y_pos))//pix_size+1)
        
    dims = (nx,ny)
    
    x_vals = np.array(x_pos//pix_size, dtype='int')
    y_vals = np.array(y_pos//pix_size, dtype='int')
    
    lin_idx = np.ravel_multi_index((x_vals, y_vals), dims)
    density = np.bincount(lin_idx, weights=np.ones(len(lin_idx)), minlength=np.prod(dims)).reshape(dims)
    density = np.clip(density,0,np.percentile(density,clip_density))

    val = (density/np.max(density)).T[:,:,None]

    sr_blur = cv2.GaussianBlur(val,(3,3),sigma/pix_size)
    sr_blur = np.clip(sr_blur, 0, sr_blur.max()/contrast_fac)

    return sr_blur
