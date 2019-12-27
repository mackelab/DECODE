import torch 
import torch.nn as nn 
import torch.nn.functional as F

import math
from torch.optim.optimizer import Optimizer, required

# torch.manual_seed(0)

class AdamW(torch.optim.Optimizer):
    """Implements AdamW algorithm.
    It has been proposed in `Fixing Weight Decay Regularization in Adam`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. Fixing Weight Decay Regularization in Adam:
    https://arxiv.org/abs/1711.05101
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # according to the paper, this penalty should come after the bias correction
                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # w = w - wd * lr * w
                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)
                
                # w = w - lr * w.grad
                p.data.addcdiv_(-step_size, exp_avg, denom)

                # w = w - wd * lr * w - lr * w.grad
                # See http://www.fast.ai/2018/07/02/adam-weight-decay/

        return loss


class Out_net(nn.Module):

    def __init__(self, n_filters, pred_sig=False, pred_bg=False):
        super(Out_net, self).__init__()
        
        self.pred_bg = pred_bg
        self.pred_sig = pred_sig
        
        self.p_out1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1).cuda()
        self.p_out2 = nn.Conv2d(n_filters, 1, kernel_size=1, padding=0).cuda()
        self.xyzi_out1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1).cuda()
        self.xyzi_out2 = nn.Conv2d(n_filters, 4, kernel_size=1, padding=0).cuda()
        
        nn.init.kaiming_normal_(self.p_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.p_out2.weight, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.p_out2.bias,-6.)
        
        nn.init.kaiming_normal_(self.xyzi_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.xyzi_out2.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.xyzi_out2.bias)
        
        if self.pred_sig:
            self.xyzis_out1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1).cuda()
            self.xyzis_out2 = nn.Conv2d(n_filters, 4, kernel_size=1, padding=0).cuda()

            nn.init.kaiming_normal_(self.xyzis_out1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.xyzis_out2.weight, mode='fan_in', nonlinearity='linear')
            nn.init.zeros_(self.xyzis_out2.bias)
        
        if self.pred_bg:
            self.bg_out1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1).cuda()
            self.bg_out2 = nn.Conv2d(n_filters, 1, kernel_size=1, padding=0).cuda()
            
            nn.init.kaiming_normal_(self.bg_out1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.bg_out2.weight, mode='fan_in', nonlinearity='linear')
            nn.init.zeros_(self.bg_out2.bias)
        
    def forward(self, x):
        
        outputs = {}
        
        p = F.elu(self.p_out1(x))
        outputs['p'] = self.p_out2(p)
    
        xyzi = F.elu(self.xyzi_out1(x))
        outputs['xyzi'] = self.xyzi_out2(xyzi)

        if self.pred_sig:
        
            xyzis = F.elu(self.xyzis_out1(x))
            outputs['xyzi_sig'] = self.xyzis_out2(xyzis)
            
        if self.pred_bg:

            bg = F.elu(self.bg_out1(x))
            outputs['bg'] = self.bg_out2(bg)
     
        return outputs
    
class SUNNet(nn.Module):
    def __init__(self, n_inp, n_filters=64, n_stages=5):
        super(SUNNet, self).__init__()
        curr_N = n_filters
        self.n_stages = n_stages
        self.layer_path = nn.ModuleList()
        
        self.layer_path.append(nn.Conv2d(n_inp, curr_N, kernel_size=3, padding=1).cuda())
        self.layer_path.append(nn.Conv2d(curr_N, curr_N, kernel_size=3, padding=1).cuda())
        
        for i in range(n_stages):
            self.layer_path.append(nn.Conv2d(curr_N, curr_N, kernel_size=2, stride=2, padding=0).cuda())
            self.layer_path.append(nn.Conv2d(curr_N, curr_N*2, kernel_size=3, padding=1).cuda())
            curr_N *=2
            self.layer_path.append(nn.Conv2d(curr_N, curr_N, kernel_size=3, padding=1).cuda())

        for i in range(n_stages):
            
            self.layer_path.append(nn.UpsamplingNearest2d(scale_factor=2).cuda())
            self.layer_path.append(nn.Conv2d(curr_N, curr_N//2, 3, padding=1).cuda())
            
            curr_N = curr_N//2
            
            self.layer_path.append(nn.Conv2d(curr_N*2, curr_N, kernel_size=3, padding=1).cuda())
            self.layer_path.append(nn.Conv2d(curr_N, curr_N, kernel_size=3, padding=1).cuda())
  
        for m in self.layer_path:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                
        
    def forward(self,x):
        
        n_l = 0
        x_bridged = []
        
        x = F.elu(list(self.layer_path)[n_l](x)); n_l += 1; 
        x = F.elu(list(self.layer_path)[n_l](x)); n_l += 1; 
        x_bridged.append(x)

        for i in range(self.n_stages):
            for n in range(3):
                x = F.elu(list(self.layer_path)[n_l](x)); n_l += 1; 
                if n == 2 and i < self.n_stages-1: 
                    x_bridged.append(x)

        for i in range(self.n_stages):
            for n in range(4):
                x = F.elu(list(self.layer_path)[n_l](x)); n_l += 1;  
                if n == 1: 
                    x = torch.cat([x,x_bridged.pop()],1) 

        return x 
    
    