import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.optim.optimizer import Optimizer, required

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



class SUNNet3D(nn.Module):
    def __init__(self, n_inp, n_filters=64, n_stages=5):
        super(SUNNet, self).__init__()
        curr_N = n_filters
        self.n_stages = n_stages
        self.layer_path = nn.ModuleList()

        self.layer_path.append(nn.Conv3d(n_inp, curr_N, kernel_size=3, padding=1).cuda())
        self.layer_path.append(nn.Conv3d(curr_N, curr_N, kernel_size=3, padding=1).cuda())

        for i in range(n_stages):
            self.layer_path.append(nn.Conv3d(curr_N, curr_N, kernel_size=2, stride=2, padding=0).cuda())
            self.layer_path.append(nn.Conv3d(curr_N, curr_N*2, kernel_size=3, padding=1).cuda())
            curr_N *=2
            self.layer_path.append(nn.Conv3d(curr_N, curr_N, kernel_size=3, padding=1).cuda())

        for i in range(n_stages):

            self.layer_path.append(nn.Upsample(scale_factor=2).cuda())
            self.layer_path.append(nn.Conv3d(curr_N, curr_N//2, 3, padding=1).cuda())

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


class Out_net3D(nn.Module):

    def __init__(self, n_filters, pred_sig=False, pred_bg=False):
        super(Out_net, self).__init__()

        self.pred_bg = pred_bg
        self.pred_sig = pred_sig

        self.p_out1 = nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1).cuda()
        self.p_out2 = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0).cuda()
        self.xyzi_out1 = nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1).cuda()
        self.xyzi_out2 = nn.Conv3d(n_filters, 4, kernel_size=1, padding=0).cuda()

        nn.init.kaiming_normal_(self.p_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.p_out2.weight, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.p_out2.bias,-6.)

        nn.init.kaiming_normal_(self.xyzi_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.xyzi_out2.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.xyzi_out2.bias)

        if self.pred_sig:
            self.xyzis_out1 = nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1).cuda()
            self.xyzis_out2 = nn.Conv3d(n_filters, 4, kernel_size=1, padding=0).cuda()

            nn.init.kaiming_normal_(self.xyzis_out1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.xyzis_out2.weight, mode='fan_in', nonlinearity='linear')
            nn.init.zeros_(self.xyzis_out2.bias)

        if self.pred_bg:
            self.bg_out1 = nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1).cuda()
            self.bg_out2 = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0).cuda()

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


