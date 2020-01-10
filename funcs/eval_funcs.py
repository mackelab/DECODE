import torch
import torch.nn as nn
import torch.nn.functional as func
from utils import *
import numpy as np
from operator import itemgetter
from tqdm import tqdm

def decode_func(model, images, batch_size=100, z_scale=10, int_scale=10, use_tqdm=False):   
    """Performs inference for a given set of images.
    
    Parameters
    ----------
    model: Model
        DECODE model
    images: numpy array
        Three dimensional array of smlm recordings
    batch_size: int
        Images are proccessed in batches of the given size. 
        When the images are large, the batch size has to be lowered to save GPU memory. 
    z_scale: float
        The model outputs z values between -1 and 1 that are rescaled.
    int_scale: float
        The model outputs intensity values between 0 and 1 that are rescaled.
        
    Returns
    -------
    infs: dict
        Dictionary of arrays with the rescaled network outputs
    """        
    
    with torch.no_grad():  
    
        N = len(images)
        h, w = images.shape[1], images.shape[2]

        if use_tqdm:
            tqdm_func = tqdm
        else:
            def tqdm_func(x):
                return x

        infs_list = []
        images = np.concatenate([images[1:2],images,images[-2:-1]],0).astype('float32')
        hbar_inp = np.zeros([batch_size+2, model.n_filters, h, w]).astype('float32')

        if model.global_context:

            hbar_inp = np.zeros([N, model.n_filters, h, w])
            for i in range(int(np.ceil(N/batch_size))):
                hbar_inp[i*batch_size:(i+1)*batch_size] = model.recfunc(gpu(images[i*batch_size:(i+1)*batch_size+2].astype('float32')), return_map=True)[1:-1].cpu()
            hbar_inp = hbar_inp.mean(0, keepdims=True).repeat(batch_size+2,0)  

        infs = {'Probs':[], 'XO':[], 'YO':[], 'ZO':[], 'Int':[]}
        if model.bg_pred:  infs['BG'] = []
        if model.sig_pred:  infs['XO_sig'] = []; infs['YO_sig'] = []; infs['ZO_sig'] = []

        for i in tqdm_func(range(int(np.ceil(N/batch_size)))):
            p,s,xyzi,xyzi_sig,bg = model.recfunc(gpu(images[i*batch_size:(i+1)*batch_size+2]), H=gpu(hbar_inp[:len(images[i*batch_size:(i+1)*batch_size+2])]), sample=False)

            infs['Probs'].append(torch.sigmoid(p)[1:-1].cpu())
            infs['XO'].append(xyzi[1:-1,0].cpu())
            infs['YO'].append(xyzi[1:-1,1].cpu())
            infs['ZO'].append(xyzi[1:-1,2].cpu())
            infs['Int'].append(xyzi[1:-1,3].cpu())
            if model.sig_pred:
                infs['XO_sig'].append(xyzi_sig[1:-1,0].cpu())
                infs['YO_sig'].append(xyzi_sig[1:-1,1].cpu())
                infs['ZO_sig'].append(xyzi_sig[1:-1,2].cpu())
            if model.bg_pred: 
                infs['BG'].append(bg[1:-1].cpu())

        for k in infs.keys():
            infs[k] = np.vstack(infs[k])

        if '3D' in str(model.mgen.psf_pars['modality']):
            infs['ZO'] = z_scale*infs['ZO']
        else:
            infs['ZO'] = 1 + softp(infs['ZO'])
        infs['Int'] = int_scale*infs['Int']

        return infs
    
def nms_sampling(res_dict, threshold=0.7, batch_size=500, nms=True, nms_cont=False):
    """Performs Non-maximum Suppression to obtain deterministic samples from the probabilities provided by the decode function. 
    
    Parameters
    ----------
    res_dict: dict
        Dictionary of arrays created with decode_func
    threshold: float
        Processed probabilities above this threshold are considered as final detections
    batch_size: int
        Outputs are proccessed in batches of the given size. 
        When the arrays are large, the batch size has to be lowered to save GPU memory. 
    nms: bool
        If False performs Non-maximum Suppression and simply applies a theshold to the probablities to obtain detections 
    nms_cont: bool
        If true also averages the offset variables according to the probabilties that count towards a given detection
        
    Returns
    -------
    res_dict: dict
        Dictionary of arrays where 'Samples_ps' contains the final detections
    """           
    res_dict['Probs_ps'] = res_dict['Probs'] + 0
    res_dict['XO_ps'] = res_dict['XO'] + 0
    res_dict['YO_ps'] = res_dict['YO'] + 0
    res_dict['ZO_ps'] = res_dict['ZO'] + 0 
    
    if nms:
    
        N = len(res_dict['Probs'])
        for i in range(int(np.ceil(N/batch_size))):
            sl = np.index_exp[i*batch_size:(i+1)*batch_size]
            if nms_cont:
                res_dict['Probs_ps'][sl], res_dict['XO_ps'][sl], res_dict['YO_ps'][sl], res_dict['ZO_ps'][sl] = nms_func(res_dict['Probs'][sl],res_dict['XO'][sl],res_dict['YO'][sl],res_dict['ZO'][sl])
            else:
                res_dict['Probs_ps'][sl] = nms_func(res_dict['Probs'][sl])
        
    res_dict['Samples_ps'] = np.where(res_dict['Probs_ps'] > threshold, 1, 0)
    
def rescale(arr_infs, rescale_bins=50, sig_3d=False):
    """Rescales x and y offsets (inplace) so that they are distributed uniformly within [-0.5, 0.5] to correct for biased outputs. 
    
    Parameters
    ----------
    arr_infs: dict
        Dictionary of arrays created with decode_func and nms_sampling
    rescale_bins: int
        The bias scales with the uncertainty of the localization. Therefore all detections are binned according to their predicted uncertainty.
        Detections within different bins are then rescaled seperately. This specifies the number of bins. 
    sig_3d: bool
        If true also the uncertainty in z when performing the binning
    """      
    if arr_infs['Samples_ps'].sum()>0:
        
        s_inds = arr_infs['Samples_ps'].nonzero()

        x_sig_var = np.var(arr_infs['XO_sig'][s_inds])
        y_sig_var = np.var(arr_infs['YO_sig'][s_inds])
        z_sig_var = np.var(arr_infs['ZO_sig'][s_inds])

        tot_sig = arr_infs['XO_sig']**2 + (np.sqrt(x_sig_var/y_sig_var) * arr_infs['YO_sig'])**2
        if sig_3d:
            tot_sig += (np.sqrt(x_sig_var/z_sig_var) * arr_infs['ZO_sig'])**2
            
        arr = np.where(arr_infs['Samples_ps'],tot_sig,0)
        bins = histedges_equalN(arr[s_inds], rescale_bins)
        for i in range(rescale_bins):
            
            inds = np.where((arr>bins[i]) & (arr<bins[i+1]) & (arr!=0))
            arr_infs['XO_ps'][inds] = uniformize(arr_infs['XO_ps'][inds]) + np.mean(arr_infs['XO_ps'][inds])
            arr_infs['YO_ps'][inds] = uniformize(arr_infs['YO_ps'][inds]) + np.mean(arr_infs['YO_ps'][inds])
    
    
def array_to_list(infs, wobble=[0,0], pix_nm=[100,100], drifts=None, start_img=0, start_n=0):
    """Transform the the output of the DECODE inference procedure (dictionary of outputs at imaging resolution) into a list of predictions. 
    
    Parameters
    ----------
    infs: dict
        Dictionary of arrays created with decode_func
    wobble: list of floats
        When working with challenge data two constant offsets can be substracted from the x,y variables to account for shifts introduced in the PSF fitting. 
    pix_nm: list of floats
        x, y pixel size of the recording in nano meter
    drifts: bool
        If False performs Non-maximum Suppression and simply applies a theshold to the probablities to obtain detections 
    start_img: int
        When processing data in multiple batches this variable should be set to the last image count of the previous batch to get continuous counting 
    start_n: int
        When processing data in multiple batches this variable should be set to the last localization count of the previous batch to get continuous counting 
        
    Returns
    -------
    res_dict: pred_list
        List of localizations with columns: 'localization', 'frame', 'x', 'y', 'z', 'intensity', 'x_sig', 'y_sig', 'z_sig'
    """           
    samples = infs['Samples_ps']
    probs = infs['Probs_ps']
    
    if drifts is None:
        drifts = np.zeros([len(samples),4])

    pred_list = []
    count = 1 + start_n
        
    for i in range(len(samples)):                                           
        pos = np.nonzero(samples[i])
        xo = infs['XO_ps'][i] - drifts[i,1]
        yo = infs['YO_ps'][i] - drifts[i,2]
        zo = infs['ZO_ps'][i] - drifts[i,3]

        ints = infs['Int'][i]
        
        if 'XO_sig' in infs:
            xos = infs['XO_sig'][i]
            yos = infs['YO_sig'][i]
            zos = infs['ZO_sig'][i]    
            
        for j in range(len(pos[0])):
            pred_list.append([count, i + 1 + start_img, 
                              (0.5 + pos[1][j] + xo[pos[0][j], pos[1][j]]) * pix_nm[0] + wobble[0],
                              (0.5 + pos[0][j] + yo[pos[0][j], pos[1][j]]) * pix_nm[1] + wobble[1],
                              zo[pos[0][j], pos[1][j]] * 100, 1000*ints[pos[0][j], pos[1][j]]])
            if 'XO_sig' in infs:
                pred_list[-1] += [xos[pos[0][j], pos[1][j]], yos[pos[0][j], pos[1][j]], zos[pos[0][j], pos[1][j]]]
            else:
                pred_list[-1] += [None,None,None]
            count += 1
            
    return pred_list

    
def filt_preds(preds, sig_perc=100, is_3d=True):
    """Removes the localizations with the highest uncertainty estimate
    
    Parameters
    ----------
    preds: list
        List of localizations
    sig_perc: float between 0 and 100
        Percentage of localizations that remain
    is_3d: int
        If false only uses x and y uncertainty to filter
        
    Returns
    -------
    preds: list
        List of remaining localizations
    """      
    if len(preds):
        if preds[0][-1] is not None:
        
            preds = np.array(preds)

            x_sig_var = np.var(preds[:,-3])
            y_sig_var = np.var(preds[:,-2])
            z_sig_var = np.var(preds[:,-1])

            tot_var = preds[:,-3]**2 + (np.sqrt(x_sig_var/y_sig_var) * preds[:,-2])**2
            if is_3d:
                tot_var += (np.sqrt(x_sig_var/z_sig_var) * preds[:,-1])**2

            max_s = np.percentile(tot_var, sig_perc)
            filt_sig = np.where(tot_var<max_s)

            preds = list(preds[filt_sig])

    return preds
    
    
def matching(test_csv, pred_inp, size_xy = [6400,6400], tolerance=250, border=450, print_res=False, min_int=False, tolerance_ax=np.inf):
    """Matches localizations to ground truth positions and provides assessment metrics used in the SMLM2016 challenge. (see http://bigwww.epfl.ch/smlm/challenge2016/index.html?p=methods#6)
    When using default parameters exactly reproduces the procedure used for the challenge (i.e. produces same numbers as the localization tool). 
    
    Parameters
    ----------
    test_csv: str or list
        Ground truth positions with columns: 'localization', 'frame', 'x', 'y', 'z'
        Either list or str with locations of csv file. 
    pred_inp: list
        List of localizations
    size_xy: list of floats
        Size of processed recording in nano meters
    tolerance: float
        Localizations are matched when they are within a circle of the given radius. 
    tolerance_ax: float
        Localizations are matched when they are closer than this value in z direction. Should be ininity for 2D recordings. 500nm is used for 3D recordings in the challenge.
    border: float
        Localizations that are close to the edge of the recording are excluded because they often suffer from artifacts. 
    print_res: bool
        If true prints a list of assessment metrics.
    min_int: bool
        If true only uses the brightest 75% of ground truth locations. 
        This is the setting used in the leaderboard of the challenge. However this implementation does not exactly match the method used in the localization tool.
    
    Returns
    -------
    perf_dict, matches: dict, list
        Dictionary of perfomance metrics.
        List of all matches localizations for further evaluation in format: 'localization', 'frame', 'x_true', 'y_true', 'z_true', 'x_pred', 'y_pred', 'z_pred', 'int_true', 'x_sig', 'y_sig', 'z_sig'
        
    """          

    perf_dict = None
    matches = []

    test_list = []
    if isinstance(test_csv, str):
        with open(test_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if 'truth' not in row[0]:
                    test_list.append([float(r) for r in row])
    else:
        for r in test_csv:
            test_list.append([i for i in r])

    test_list = sorted(test_list, key=itemgetter(1))

    if min_int: 
        min_int = np.percentile(np.array(test_list)[:,-1], 25)
    else:
        min_int = 0

    if isinstance(pred_inp, str):
        pred_list = []
        with open(pred_inp, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if 'truth' not in row[0]:
                    pred_list.append([float(r) for r in row])

    pred_list = copy.deepcopy(pred_inp)
    if len(pred_list) == 0:
        perf_dict = {'recall': np.nan, 'precision': np.nan, 'jaccard': np.nan, 'f_score': np.nan, 'rmse_lat': np.nan, 'rmse_ax': np.nan, 
                'rmse_x': np.nan, 'rmse_y': np.nan, 'jor': np.nan, 'eff_lat': np.nan, 'eff_ax': np.nan, 'eff_3d': np.nan}

        return perf_dict, matches                

        
    if border:
        test_arr = np.array(test_list)
        pred_arr = np.array(pred_list)

        t_inds = np.where((test_arr[:, 2] < border) | (test_arr[:, 2] > (size_xy[0] - border)) | (test_arr[:, 3] < border) | (test_arr[:, 3] > (size_xy[1] - border)))
        p_inds = np.where((pred_arr[:, 2] < border) | (pred_arr[:, 2] > (size_xy[0] - border)) | (pred_arr[:, 3] < border) | (pred_arr[:, 3] > (size_xy[1] - border)))
        for t in reversed(t_inds[0]):
            del (test_list[t])
        for p in reversed(p_inds[0]):
            del (pred_list[p])

    TP = 0
    FP = 0.0001
    FN = 0.0001
    MSE_lat = 0
    MSE_ax = 0
    MSE_vol = 0
    
    if len(pred_list):

        for i in range(1, int(pred_list[-1][1])+1):

            tests = []
            preds = []
            while test_list[0][1] == i:
                tests.append(test_list.pop(0))
                if len(test_list) < 1:
                    break
            while pred_list[0][1] == i:
                preds.append(pred_list.pop(0))
                if len(pred_list) < 1:
                    break
            dist_arr = np.zeros([len(tests), len(preds)])
            ax_arr = np.zeros([len(tests), len(preds)])
            tot_arr = np.zeros([len(tests), len(preds)])

            for t in range(len(tests)):
                for p in range(len(preds)):
                    dist_arr[t, p] = np.sqrt((tests[t][2] - preds[p][2]) ** 2 + (tests[t][3] - preds[p][3]) ** 2)
                    ax_arr[t, p] = np.abs((tests[t][4] - preds[p][4]))
                    tot_arr[t, p] = np.sqrt((tests[t][2] - preds[p][2]) ** 2 + (tests[t][3] - preds[p][3]) ** 2 + (tests[t][4] - preds[p][4]) ** 2)
            if tolerance_ax == np.inf:
                tot_arr = dist_arr

            match_tests = copy.deepcopy(tests)
            match_preds = copy.deepcopy(preds)

            if dist_arr.size > 0:
                while dist_arr.min() < tolerance:
                    r, c = np.where(tot_arr == tot_arr.min())
                    r = r[0]
                    c = c[0]
                    if ax_arr[r, c] < tolerance_ax and dist_arr[r, c] < tolerance:

                        if match_tests[r][-1] > min_int:

                            MSE_lat += dist_arr[r, c] ** 2
                            MSE_ax += ax_arr[r, c] ** 2
                            MSE_vol += dist_arr[r, c] ** 2 + ax_arr[r, c] ** 2
                            TP += 1
                            matches.append([match_tests[r][2], match_tests[r][3], match_tests[r][4], 
                                            match_preds[c][2], match_preds[c][3], match_preds[c][4], match_tests[r][5],
                                            match_preds[c][-3], match_preds[c][-2], match_preds[c][-1]])

                        dist_arr[r, :] = np.inf
                        dist_arr[:, c] = np.inf
                        tot_arr[r, :] = np.inf
                        tot_arr[:, c] = np.inf

                        tests[r][-1] = -100
                        preds.pop()

                    dist_arr[r, c] = np.inf
                    tot_arr[r, c] = np.inf

            for i in reversed(range(len(tests))):
                if tests[i][-1] < min_int:
                    del(tests[i])   

            FP += len(preds)
            FN += len(tests)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    jaccard = TP / (TP + FP + FN)
    rmse_lat = np.sqrt(MSE_lat / (TP + 0.00001))
    rmse_ax = np.sqrt(MSE_ax / (TP + 0.00001))
    rmse_vol = np.sqrt(MSE_vol / (TP + 0.00001))
    jor = 100*jaccard/rmse_lat

    eff_lat = 100-np.sqrt((100-100*jaccard)**2 + 1**2 * rmse_lat**2)
    eff_ax = 100-np.sqrt((100-100*jaccard)**2 + 0.5**2 * rmse_ax**2)
    eff_3d = (eff_lat+eff_ax)/2

    
    matches = np.array(matches)
    rmse_x = np.nan
    rmse_y = np.nan
    if len(matches):
        rmse_x = np.sqrt(((matches[:,0]-matches[:,3])**2).mean())
        rmse_y = np.sqrt(((matches[:,1]-matches[:,4])**2).mean())

    if print_res:
        print('{}{:0.3f}'.format('Recall: ', recall))
        print('{}{:0.3f}'.format('Precision: ', precision))
        print('{}{:0.3f}'.format('Jaccard: ', 100 * jaccard))
        print('{}{:0.3f}'.format('RMSE_lat: ', rmse_lat))
        print('{}{:0.3f}'.format('RMSE_ax: ', rmse_ax))
        print('{}{:0.3f}'.format('RMSE_vol: ', rmse_vol))
        print('{}{:0.3f}'.format('Jaccard/RMSE: ', jor))
        print('{}{:0.3f}'.format('Eff_lat: ', eff_lat))
        print('{}{:0.3f}'.format('Eff_ax: ', eff_ax))
        print('{}{:0.3f}'.format('Eff_3d: ', eff_3d))
        print('FN: ' + str(np.round(FN)) + ' FP: ' + str(np.round(FP)))

    perf_dict = {'recall': recall, 'precision': precision, 'jaccard': jaccard, 'rmse_lat': rmse_lat, 'rmse_ax': rmse_ax, 'rmse_vol': rmse_vol, 
            'rmse_x': rmse_x, 'rmse_y': rmse_y, 'jor': jor, 'eff_lat': eff_lat, 'eff_ax': eff_ax, 'eff_3d': eff_3d}
                
    return perf_dict, matches

    
def nms_func(p, xo=None, yo=None, zo=None):

    with torch.no_grad():

        diag = 0 #1/np.sqrt(2)

        p = gpu(p)

        p_copy = p + 0
        
        # probability values > 0.3 are regarded as possible locations

        p_clip = torch.where(p>0.3,p,torch.zeros_like(p))[:,None]

        # localize maximum values within a 3x3 patch
        
        pool = func.max_pool2d(p_clip,3,1,padding=1)
        max_mask1 = torch.eq(p[:,None], pool).float()
        
        # Add probability values from the 4 adjacent pixels
        
        filt = np.array([[diag,1,diag],[1,1,1],[diag,1,diag]],ndmin=4)
        conv = func.conv2d(p[:,None], gpu(filt),padding=1)
        p_ps1 = max_mask1*conv
        
        # In order do be able to identify two fluorophores in adjacent pixels we look for probablity values > 0.6 that are not part of the first mask

        p_copy *= (1-max_mask1[:,0])
        p_clip = torch.where(p_copy>0.6,p_copy,torch.zeros_like(p_copy))[:,None]
        max_mask2 = torch.where(p_copy>0.6,torch.ones_like(p_copy),torch.zeros_like(p_copy))[:,None]
        p_ps2 = max_mask2*conv
        
        # This is our final clustered probablity which we then threshold (normally > 0.7) to get our final discrete locations 
        p_ps = p_ps1 + p_ps2
        
        if xo is None:
            return p_ps[:,0].cpu()
        
        xo = gpu(xo)
        yo = gpu(yo)
        zo = gpu(zo)

        max_mask = torch.clamp(max_mask1 + max_mask2, 0, 1)

        mult_1 = max_mask1/p_ps1
        mult_1[torch.isnan(mult_1)] = 0
        mult_2 = max_mask2/p_ps2
        mult_2[torch.isnan(mult_2)] = 0
        
        # The rest is weighting the offset variables by the probabilities

        z_mid = zo * p
        z_conv1 = func.conv2d((z_mid * (1-max_mask2[:,0]))[:,None],gpu(filt),padding=1)
        z_conv2 = func.conv2d((z_mid * (1-max_mask1[:,0]))[:,None],gpu(filt),padding=1)

        zo_ps = z_conv1*mult_1 + z_conv2*mult_2
        zo_ps[torch.isnan(zo_ps)] = 0

        x_mid = xo * p
        x_mid_filt = np.array([[0,1,0],[0,1,0],[0,1,0]],ndmin=4)
        xm_conv1 = func.conv2d((x_mid * (1-max_mask2[:,0]))[:,None],gpu(x_mid_filt),padding=1)
        xm_conv2 = func.conv2d((x_mid * (1-max_mask1[:,0]))[:,None],gpu(x_mid_filt),padding=1)

        x_left = (xo+1) * p
        x_left_filt = flip_filt(np.array([[diag,0,0],[1,0,0],[diag,0,0]],ndmin=4))
        xl_conv1 = func.conv2d((x_left * (1-max_mask2[:,0]))[:,None],gpu(x_left_filt),padding=1)
        xl_conv2 = func.conv2d((x_left * (1-max_mask1[:,0]))[:,None],gpu(x_left_filt),padding=1)

        x_right = (xo-1) * p
        x_right_filt = flip_filt(np.array([[0,0,diag],[0,0,1],[0,0,diag]],ndmin=4))
        xr_conv1 = func.conv2d((x_right * (1-max_mask2[:,0]))[:,None],gpu(x_right_filt),padding=1)
        xr_conv2 = func.conv2d((x_right * (1-max_mask1[:,0]))[:,None],gpu(x_right_filt),padding=1)

        xo_ps = (xm_conv1+xl_conv1+xr_conv1)*mult_1 + (xm_conv2+xl_conv2+xr_conv2)*mult_2

        y_mid = yo * p
        y_mid_filt = np.array([[0,0,0],[1,1,1],[0,0,0]],ndmin=4)
        ym_conv1 = func.conv2d((y_mid * (1-max_mask2[:,0]))[:,None],gpu(y_mid_filt),padding=1)
        ym_conv2 = func.conv2d((y_mid * (1-max_mask1[:,0]))[:,None],gpu(y_mid_filt),padding=1)

        y_up = (yo+1) * p
        y_up_filt = flip_filt(np.array([[diag,1,diag],[0,0,0],[0,0,0]],ndmin=4))
        yu_conv1 = func.conv2d((y_up * (1-max_mask2[:,0]))[:,None],gpu(y_up_filt),padding=1)
        yu_conv2 = func.conv2d((y_up * (1-max_mask1[:,0]))[:,None],gpu(y_up_filt),padding=1)

        y_down = (yo-1) * p
        y_down_filt = flip_filt(np.array([[0,0,0],[0,0,0],[diag,1,diag]],ndmin=4))
        yd_conv1 = func.conv2d((y_down * (1-max_mask2[:,0]))[:,None],gpu(y_down_filt),padding=1)
        yd_conv2 = func.conv2d((y_down * (1-max_mask1[:,0]))[:,None],gpu(y_down_filt),padding=1)

        yo_ps = (ym_conv1+yu_conv1+yd_conv1)*mult_1 + (ym_conv2+yu_conv2+yd_conv2)*mult_2
    
        return p_ps[:,0].cpu(),xo_ps[:,0].cpu(),yo_ps[:,0].cpu(),zo_ps[:,0].cpu()


def cdf_get(cdf,val):

    ind = (val+1)/2*200 - 1.
    dec = ind - np.floor(ind)
    
    return(dec*cdf[[int(i) + 1 for i in ind]] + (1-dec)*cdf[[int(i) for i in ind]])
    
def uniformize(x):
    
    x = np.clip(x,-0.99,0.99)
    x_cdf = np.histogram(x, bins=np.linspace(-1,1,201))
    x_re = cdf_get(np.cumsum(x_cdf[0])/sum(x_cdf[0]), x)
    
    return (x_re-0.5)

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))


def list_to_arr(list_or_path, img_size=64, pix_xy=[100,100], wobble=[0,0]):
    
    if isinstance(list_or_path, str):
        pos = []
        reader = csv.reader(open(list_or_path, 'r'))
        for row in reader:
            gt, f, x, y, z, i = row
            if f != 'frame':
                pos.append([float(gt), float(f), float(x)-wobble[0], float(y)-wobble[1], float(z), float(i)])

        pos = np.array(pos).T.astype('float32')
    else:
        pos = np.array(list_or_path).T
        
    N = int(pos[1].max())
    
    locs = {'Samples':np.zeros([N,img_size,img_size]), 'XO':np.zeros([N,img_size,img_size]), 'YO':np.zeros([N,img_size,img_size]), 'ZO':np.zeros([N,img_size,img_size]), 'Int':np.zeros([N,img_size,img_size])}
    
    for i in range(N):
        curr = pos[:,np.where(pos[1]-1 == i)][:,0]
        for p in curr.T:
            x_ind = np.min([int(p[2]/pix_xy[0]),img_size-1])
            y_ind = np.min([int(p[3]/pix_xy[0]),img_size-1])
            if locs['Samples'][i,y_ind,x_ind] == 0:
                locs['Samples'][i,y_ind,x_ind] = 1
                locs['XO'][i,y_ind,x_ind] -= (x_ind*100 - p[2])/100 + 0.5
                locs['YO'][i,y_ind,x_ind] -= (y_ind*100 - p[3])/100 + 0.5
                locs['ZO'][i,y_ind,x_ind] = p[4]
                locs['Int'][i,y_ind,x_ind] = p[5]
                
    return locs


