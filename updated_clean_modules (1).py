import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import os
from torch.nn import Parameter, init
from torch.autograd import Function
import math
from torch.nn.modules.utils import _pair
import cv2
import pdb
from pdb import set_trace as bp

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

###How things are generally defined?
class ComplexConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding

        self.conv_real = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_imag = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x):
        input_dim = x.shape[1] // 2
        x_real = x[:, :input_dim, :, :]
        x_imag = x[:, input_dim:, :, :]
        real_real = self.conv_real(x_real)
        imag_imag = self.conv_imag(x_imag)
        real_imag = self.conv_real(x_imag)
        imag_real = self.conv_imag(x_real)
        real = real_real - imag_imag
        imag = real_imag + imag_real
        conv_map = torch.cat((real, imag),1)
        return conv_map
    
###Hard-coded filters.
class New_ComplexConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(New_ComplexConv,self).__init__()
        self.padding = padding
        self.stride = stride

        self.real_filter = torch.nn.init.normal_(torch.randn(out_channel, in_channel, kernel_size[0], kernel_size[1])).to(device)
        self.imag_filter = torch.nn.init.normal_(torch.randn(out_channel, in_channel, kernel_size[0], kernel_size[1])).to(device)

        self.weightr = nn.Parameter(self.real_filter, requires_grad=True)
        #self.biasr = nn.Parameter(torch.Tensor(1), requires_grad=True).to(device)
        self.weighti = nn.Parameter(self.imag_filter, requires_grad=True)
        #self.biasi = nn.Parameter(torch.Tensor(1), requires_grad=True).to(device)

        #self.conv_real = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        #self.conv_imag = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x):
        input_dim = x.shape[1] // 2
        x_real = x[:, :input_dim, :, :]
        x_imag = x[:, input_dim:, :, :]
        real_real = F.conv2d(x_real, self.weightr, stride=self.stride, padding=self.padding)
        imag_imag = F.conv2d(x_imag, self.weighti, stride=self.stride, padding=self.padding)
        real_imag = F.conv2d(x_imag, self.weightr, stride=self.stride, padding=self.padding)
        imag_real = F.conv2d(x_real, self.weighti, stride=self.stride, padding=self.padding)
        real = real_real - imag_imag
        imag = real_imag + imag_real
        conv_map = torch.cat((real, imag),1)
        return conv_map

def real_to_complex(x):
    a = random.uniform(0,1)
    a = torch.tensor([a]).to(device)
    b = random.uniform(0,1)
    b = torch.tensor([b]).to(device)
    alpha = random.uniform(0.0,np.pi)
    alpha = torch.tensor([alpha]).to(device)
    z = (torch.sin(a*x+(1j*(b*x))+alpha)).to(device)
    real = z.real
    imag = z.imag
    z = torch.cat((real,imag),1)
    return z



class ComplexImg(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True):
        super(ComplexImg,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv0 = nn.Conv2d(in_channel, out_channel, kernel_size, stride=1, padding='same')
        self.act = nn.ReLU()
        
    def forward(self, x):
        inter = self.bn(x)
        inter = self.act(inter)
        inter = self.conv0(inter)
        inter = self.bn(inter)
        inter = self.act(inter)
        inter = self.conv0(inter)
        img = torch.cat((x, inter),1)
        return img

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, shortcut):
        super(ResidualBlock,self).__init__()
        self.Cbn1 = ComplexBatchNorm2d(in_channel)
        self.Cbn2 = ComplexBatchNorm2d(out_channel)
        self.Cconv = ComplexConv(in_channel, out_channel, kernel_size, padding="same", bias=False)
        self.Cconv1 = ComplexConv(in_channel, out_channel, kernel_size=1, padding="same", bias=False)
        self.Cact = Cardioid()
        self.shortcut = shortcut

    def forward(self, I):
        #O = self.Cbn1(I)
        O = self.Cact(I)
        O = self.Cconv(O)

        #O = self.Cbn2(O)
        O = self.Cact(O)
        O = self.Cconv(O)

        if self.shortcut=='regular':
            O = O+I
        elif self.shortcut=='proj':
            X = self.Cconv1(I)
            input_dim_X = X.shape[1] // 2
            input_dim_O = O.shape[1] // 2
            X_real = X[:, :input_dim_X, :, :]
            X_imag = X[:, input_dim_X:, :, :]

            O_real = O[:, :input_dim_O, :, :]
            O_imag = O[:, input_dim_O:, :, :]

            O_real = torch.cat((X_real,O_real),1)
            O_imag = torch.cat((X_imag,O_imag),1)

            O = torch.cat((O_real,O_imag),1)

        return O

def normal_img(x):
    real = x
    imag = torch.zeros(x.shape).to(device)
    img = torch.cat((real, imag),1)
    return img

################################## Complex BatchNorm code is taken from the DCN implementation ######################################################
class _ComplexBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features,3))
            self.bias = Parameter(torch.Tensor(num_features,2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype = torch.complex64))
            self.register_buffer('running_covar', torch.zeros(num_features,3))
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:,:2],1.4142135623730951)
            init.zeros_(self.weight[:,2])
            init.zeros_(self.bias)

class ComplexBatchNorm2d(_ComplexBatchNorm):

    def forward(self, input):
        input_dim = input.shape[1] // 2
        x_real = input[:, :input_dim, :, :]
        x_imag = input[:, input_dim:, :, :]
        #print(input.shape)
        input = torch.complex(x_real, x_imag)
        #print(input.shape)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = input.real.mean([0, 2, 3]).type(torch.complex64)
            mean_i = input.imag.mean([0, 2, 3]).type(torch.complex64)
            mean = mean_r + 1j*mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, :, None, None]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = 1./n*input.real.pow(2).sum(dim=[0,2,3])+self.eps
            Cii = 1./n*input.imag.pow(2).sum(dim=[0,2,3])+self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=[0,2,3])
        else:
            Crr = self.running_covar[:,0]+self.eps
            Cii = self.running_covar[:,1]+self.eps
            Cri = self.running_covar[:,2]#+self.eps 
       
        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:,0] = exponential_average_factor * Crr * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,0]

                self.running_covar[:,1] = exponential_average_factor * Cii * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,1]

                self.running_covar[:,2] = exponential_average_factor * Cri * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,2]

        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (Rrr[None,:,None,None]*input.real+Rri[None,:,None,None]*input.imag).type(torch.complex64) \
                + 1j*(Rii[None,:,None,None]*input.imag+Rri[None,:,None,None]*input.real).type(torch.complex64)

        if self.affine:
            input = (self.weight[None,:,0,None,None]*input.real+self.weight[None,:,2,None,None]*input.imag+\
                    self.bias[None,:,0,None,None]).type(torch.complex64) \
                    +1j*(self.weight[None,:,2,None,None]*input.real+self.weight[None,:,1,None,None]*input.imag+\
                    self.bias[None,:,1,None,None]).type(torch.complex64)

        input_real = input.real
        input_imag = input.imag
        input = torch.cat((input_real,input_imag),1)

        return input

########################################################################################################################################

def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output

def magn_phase(x):
    magn_phase = abs(x+1/x)
    return magn_phase
        
def complex_max_pooling(x,pool_size):
    input_dim = x.shape[1] // 2
    x_real = x[:, :input_dim, :, :]
    x_imag = x[:, input_dim:, :, :]
    x = torch.complex(x_real, x_imag)
    complex_net = magn_phase(x)
    #maxpool = torch.nn.MaxPool2d(2,2, return_indices=True)
    maxpool = torch.nn.MaxPool2d(pool_size[0],pool_size[1], return_indices=True)
    complex_net, indices = maxpool(complex_net)
    y = retrieve_elements_from_indices(x,indices)
    y_real = y.real
    y_imag = y.imag
    y = torch.cat((y_real,y_imag),1)
    return y

def cardioid(x):
    input_dim = x.shape[1] // 2
    x_real = x[:, :input_dim, :, :]
    x_imag = x[:, input_dim:, :, :]
    x = torch.complex(x_real, x_imag)
    phase = x.angle()
    scale = 0.5*(1.0+torch.cos(phase))
    real = x.real*scale
    imag = x.imag*scale
    x = torch.cat((real,imag),1)
    return x

def cardioid_linear(x):
    input_dim = x.shape[1] // 2
    x_real = x[:, :input_dim]
    x_imag = x[:, input_dim:]
    x = torch.complex(x_real, x_imag)
    phase = x.angle()
    scale = 0.5*(1.0+torch.cos(phase))
    real = x.real*scale
    imag = x.imag*scale
    x = torch.cat((real,imag),1)
    return x

def complex_exp(x):
    x_real = x.real
    x_imag = x.imag
    exp_real = torch.exp(x_real)
    exp_imag = (torch.cos(x_imag))+(1j*(torch.sin(x_imag)))
    return (exp_real*exp_imag)

def apply_complex(fr, fi, input, dtype = torch.complex64):
    x =  (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)
    real = x.real
    imag = x.imag
    x = torch.cat((real, imag),1)
    return x

class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, input):
        input_dim = input.shape[1] // 2
        x_real = input[:, :input_dim]
        x_imag = input[:, input_dim:]
        input = torch.complex(x_real, x_imag)
        return apply_complex(self.fc_r, self.fc_i, input)


class Complex_ReLU(nn.Module):
    def forward(self,input):
         return ReLU1234(input)

class Cardioid(nn.Module):
    def forward(self,input):
         return cardioid(input)

class Cardioid_Linear(nn.Module):
    def forward(self,input):
         return cardioid_linear(input)

class Complex_ReLU14(nn.Module):
    def forward(self,input):
         return ReLU14(input)

class Exp_act(nn.Module):
    def forward(self,input):
         return complex_exp(input)

class complex_img(nn.Module):
    def forward(self,input):
         return fft_image(input)

class cRelu(nn.Module):
    def __init__(self):
        super(cRelu,self).__init__()
        self.relu = nn.ReLU()
    def forward(self,x):
        input_dim = x.shape[1]//2
        x_real = x[:, :input_dim, :, :]
        x_imag = x[:, input_dim:, :, :]
        real_relu = self.relu(x_real)
        imag_relu = self.relu(x_imag)
        crelu = torch.cat((real_relu, imag_relu),1)
        return crelu


def complex_dropout(input, p=0.5, training=True):
    input_dim = input.shape[1]//2
    x_real = input[:, :input_dim]
    x_imag = input[:, input_dim:]
    input = torch.complex(x_real, x_imag)
    mask = torch.ones(input.shape, dtype = torch.float32).to(device)
    mask = F.dropout(mask, p, training)*1/(1-p)
    mask.type(input.dtype)
    dropout_out = mask*input
    dropout_real = dropout_out.real
    dropout_imag = dropout_out.imag
    dropout_out = torch.cat((dropout_real, dropout_imag),1)
    return dropout_out

class dropout(nn.Module):
    def forward(self,input,p):
        return complex_dropout(input,p)

class max_pool(nn.Module):
    def forward(self, input, pool_size):
        return complex_max_pooling(input,pool_size)

def error_reg(out, g_t):
    out_real = out.real
    out_imag = out.imag
    g_t_real = g_t.real
    g_t_imag = g_t.imag
    real_mul = out_real*g_t_real
    g_o_indices = real_mul >= 0.0
    out_real[g_o_indices] == 0.0
    out_imag[g_o_indices] == 0.0
    output = torch.complex(out_real,out_imag)
    g_t_real[g_o_indices] == 0.0
    g_t_imag[g_o_indices] == 0.0
    gt = torch.complex(g_t_real,g_t_imag)
    error = gt-output
    return error

def regularizer(out,g_t,epoch):
    input_dim = out.shape[1] // 2
    out_real = out[:,:input_dim]
    out_imag = out[:,input_dim:]
    g_t_real = g_t[:,:input_dim]
    g_t_imag = g_t[:,input_dim:]
    out = torch.complex(out_real,out_imag)
    g_t = torch.complex(g_t_real, g_t_imag)
    err = torch.tensor([0.0]).type(torch.complex64).to(device)
    e_t_init = 0.5
    if (epoch%10==0):
        e_thresh = e_t_init*np.exp(-0.2)
        e_t_init = e_thresh
    else:
        e_thresh = e_t_init
    correct_prediction = torch.eq(torch.argmax(g_t_real, 1), torch.argmax(out_real, 1))
    for i in range(len(correct_prediction)):
        e = torch.max(abs(error_reg(out[i],g_t[i])))
        if((correct_prediction[i]==True) and (e < e_thresh)):
            zero_C = torch.tensor([0.0]).type(torch.complex64).to(device)
            loss_e = error_reg(zero_C,zero_C).to(device)
        else:
            loss_e = error_reg(out[i],g_t[i]).to(device)
        err = torch.cat((err,loss_e))
    
    error_real = err.real
    error_imag = err.imag
    error_imag_conj = torch.neg(error_imag)
    loss_val = ((error_real*error_real)-(error_imag*error_imag_conj))
    loss_val = torch.mean(loss_val)
    return loss_val

class Complex_Loss(nn.Module):
    def forward(self,out,g_t,epoch):
        return regularizer(out,g_t,epoch)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            ComplexLinear(gate_channels, gate_channels // reduction_ratio),
            Cardioid_Linear(),
            ComplexLinear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        input_dim = x.shape[1]//2
        x_real = x[:, :input_dim, :, :]
        x_imag = x[:, input_dim:, :, :]
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool_real = F.avg_pool2d(x_real,(x_real.size(2), x_real.size(3)), stride=(x_real.size(2), x_real.size(3)))
                avg_pool_imag = F.avg_pool2d(x_imag,(x_imag.size(2), x_imag.size(3)), stride=(x_imag.size(2), x_imag.size(3)))
                avg_pool = torch.cat((avg_pool_real,avg_pool_imag),1)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = complex_max_pooling(x,(x.size(2),x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        #scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return channel_att_sum


class ChannelPool(nn.Module):
    def forward(self, x):
        input_dim = x.shape[1]//2
        x_real = x[:, :input_dim]
        x_imag = x[:, input_dim:]
        max_pool_map = torch.max(x_real,1)[0]
        C_x = torch.complex(x_real,x_imag)
        avg_pool = torch.mean(C_x,1)
        input_dim = avg_pool.shape[1]//2
        avg_real = avg_pool[:, :input_dim]
        avg_imag = avg_pool[:, input_dim:]
        avg_pool_map = torch.cat((avg_real,avg_imag),1)

        return torch.cat((max_pool_map,avg_pool_map),1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = ComplexConv(2,1,kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        #scale = F.sigmoid(x_out)
        return x_out

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

##################################################################### Custom Scheduler ########################################
class CustomDCNScheduler():
    def __init__(self, optimizer):
        self._optimizer = optimizer
        
    def schedule(self, epoch):
        if   epoch >=   0 and epoch <  10:
            lrate = 0.01
            for g in self._optimizer.param_groups:            
                g['lr'] = lrate
        elif epoch >=  10 and epoch < 100:
            lrate = 0.1
            for g in self._optimizer.param_groups:            
                g['lr'] = lrate
        elif epoch >= 100 and epoch < 120:
            lrate = 0.01
            for g in self._optimizer.param_groups:            
                g['lr'] = lrate
        elif epoch >= 120 and epoch < 150:
            lrate = 0.001
            for g in self._optimizer.param_groups:            
                g['lr'] = lrate
        elif epoch >= 150:
            lrate = 0.0001
            for g in self._optimizer.param_groups:            
                g['lr'] = lrate
        return g

#############################################################################################################################################
class SpectralPooling2D(nn.Module):
    def __init__(self, **kwargs):
        super(SpectralPooling2D, self).__init__()
        if   "topf"  in kwargs:
            self.topf  = (int  (kwargs["topf" ][0]), int  (kwargs["topf" ][1]))
            self.topf  = (self.topf[0]//2, self.topf[1]//2)
        elif "gamma" in kwargs:
            self.gamma = (float(kwargs["gamma"][0]), float(kwargs["gamma"][1]))
            self.gamma = (self.gamma[0]/2, self.gamma[1]/2)
        else:
            raise RuntimeError("Must provide either topf= or gamma= !")
    def forward(self, x, mask=None):
        xshape = x.shape
        topf = (int(self.gamma[0]*xshape[2]), int(self.gamma[1]*xshape[3]))

        if(topf[0] > 0 and xshape[2] >= 2*topf[0]):
            mask = [1]*(topf[0]              ) +\
                    [0]*(xshape[2] - 2*topf[0]) +\
                    [1]*(topf[0]              )
            mask = [[[mask]]]
            mask = np.asarray(mask, dtype='float32').transpose((0,1,3,2))
            mask = torch.tensor(mask).to(device)
            x   *= mask
        if(topf[1] > 0 and xshape[3] >= 2*topf[1]):
            mask = [1]*(topf[1]              ) +\
                    [0]*(xshape[3] - 2*topf[1]) +\
                    [1]*(topf[1]              )
            mask = [[[mask]]]
            mask = np.asarray(mask, dtype='float32').transpose((0,1,2,3))
            mask = torch.tensor(mask).to(device)
            x   *= mask
		
        return x
