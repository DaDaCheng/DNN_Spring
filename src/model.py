import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
#from scipy import stats
import time

import torch.nn.init as init


def is_none(value):
    return value is None

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        out_dim=args.net_outdim
        depth=args.net_depth
        dim=args.net_dim
        act=args.net_act
        bias=args.net_bias
        alpha=args.net_alpha
        self.bias=bias
        self.alpha=alpha
        self.args=args
        self.depth=depth
        self.isbatchnorm=args.net_isbatchnorm
        self.dim=dim
        self.act=torch.nn.LeakyReLU
        self.fc_list=torch.nn.ModuleList([])
        self.ac_list=torch.nn.ModuleList([])
        self.do_list=torch.nn.ModuleList([])

        if self.isbatchnorm:
            self.bminput=nn.BatchNorm1d(dim)
            self.bmoutput=torch.nn.Identity()
            self.bm_list=torch.nn.ModuleList([])
        for i in range(self.depth-1):
            self.fc_list.append(nn.Linear(dim, dim,bias=bias))
            self.ac_list.append(self.act(alpha))
            if self.isbatchnorm:
                self.bm_list.append(nn.BatchNorm1d(dim))
            self.do_list.append(torch.nn.Dropout(args.net_p))
        self.fc_list.append(nn.Linear(dim,out_dim,bias=bias))
        self.init_scale=args.init_scale
        if args.init_scale=='default':
            pass
        else:
            self.reset_parameters()
    def reset_parameters(self) -> None:
        print('init_scale:',self.init_scale)
        for i in range(self.depth):
            if self.init_scale=='orth':
                init.orthogonal_(self.fc_list[i].weight)
            if self.init_scale=='kaiming':
                init.kaiming_uniform_(self.fc_list[i].weight, nonlinearity='leaky_relu',a=self.alpha)
            if self.init_scale=='normal':
                init.normal_(self.fc_list[i].weight)
            if self.init_scale=='NTK':
                init.normal_(self.fc_list[i].weight)
            if self.init_scale=='id':
                init.eye_(self.fc_list[i].weight)
            if self.args.init_constant==1:
                pass
            else:
                self.fc_list[i].weight.data = self.fc_list[i].weight.data*self.args.init_constant
            if self.bias:
                if self.args.init_zerobias:
                    init.zeros_(self.fc_list[i].bias)
                else:
                    if self.init_scale=='NTK':
                        init.uniform_(self.fc_list[i].bias, -1.0, 1.0)
                    else:
                        init.uniform_(self.fc_list[i].bias,-1/np.sqrt(self.dim),1/np.sqrt(self.dim))
#       
    def forward(self, x):
        if self.isbatchnorm:
            x=self.bminput(x)
        for i in range(self.depth-1):
            x=self.fc_list[i](x)
            if self.init_scale=='NTK':
                x=x/np.sqrt(self.dim)
            if self.isbatchnorm:
                x=self.bm_list[i](x)
            x=self.ac_list[i](x) 
            x=self.do_list[i](x)
        x=self.fc_list[-1](x)
        if self.init_scale=='NTK':
            x=x/np.sqrt(self.dim)
        if self.isbatchnorm:
            x=self.bmoutput(x)
        return x
    def get_mid_output(self,x):
        output_before_list=[]
        output_mid_list=[]
        output_after_list=[]
        output_before_list.append(x.clone().detach().cpu())
        if self.isbatchnorm:
            x=self.bminput(x)
        output_mid_list.append(x.clone().detach().cpu())
        output_after_list.append(x.clone().detach().cpu())
        for i in range(self.depth-1):
            x=self.fc_list[i](x)
            if self.init_scale=='NTK':
                x=x/np.sqrt(self.dim)
            output_before_list.append(x.clone().detach().cpu())
            if self.isbatchnorm:
                x=self.bm_list[i](x)
            output_mid_list.append(x.clone().detach().cpu())
            x=self.ac_list[i](x)
            output_after_list.append(x.clone().detach().cpu())
            x=self.do_list[i](x)
        x=self.fc_list[-1](x)
        if self.init_scale=='NTK':
            x=x/np.sqrt(self.dim)
        output_before_list.append(x.clone().detach().cpu())
        if self.isbatchnorm:
            x=self.bmoutput(x)
        output_mid_list.append(x.clone().detach().cpu())
        output_after_list.append(x.clone().detach().cpu())
        return output_before_list,output_mid_list,output_after_list,x






    
LeakyReLU=torch.nn.LeakyReLU
class CNN(nn.Module):
    def __init__(self,tau=0,width=40):
        super(CNN, self).__init__()
        self.width=width
        self.bn0 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, width, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(width)
        self.conv4 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(width)
        self.conv5 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(width)
        self.conv6 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(width)
        self.conv7 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(width)
        # self.conv8 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        # self.bn8 = nn.BatchNorm2d(16)
        # self.conv9 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        # self.bn9 = nn.BatchNorm2d(16)
        # self.conv10 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        # self.bn10 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2) 
        self.dropout = nn.Dropout(0)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.fc1 = nn.Linear(width*10*10, 10)
        self.nl=LeakyReLU(tau)

    def forward(self, x):
        x = x.reshape(x.shape[0],3,10,10)
        x = self.bn0(x)
        x = self.upsample(self.pool(self.nl(self.bn1(self.conv1(x)))))
        x = self.dropout(x)
        x = self.upsample(self.pool(self.nl(self.bn2(self.conv2(x)))))
        x = self.dropout(x)
        x = self.upsample(self.pool(self.nl(self.bn3(self.conv3(x)))))
        x = self.dropout(x)
        x = self.upsample(self.pool(self.nl(self.bn4(self.conv4(x)))))
        x = self.dropout(x)
        x = self.upsample(self.pool(self.nl(self.bn5(self.conv5(x)))))
        x = self.dropout(x)
        x = self.upsample(self.pool(self.nl(self.bn6(self.conv6(x)))))
        x = self.dropout(x)
        x = self.upsample(self.pool(self.nl(self.bn7(self.conv7(x)))))
        x = self.dropout(x)

        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x
    def get_mid_output(self,x):
        with torch.no_grad():
            x = self.bn0(x)
            x_list=[]
            x_list.append(x.detach().clone().cpu().reshape(x.shape[0],-1))
            x = x.reshape(x.shape[0],3,10,10)
            x=self.pool(self.nl(self.bn1(self.conv1(x))))
            x_list.append(x.detach().clone().cpu().reshape(x.shape[0],-1))
            x = self.upsample(x)

            x=self.pool(self.nl(self.bn2(self.conv2(x))))
            x_list.append(x.detach().clone().cpu().reshape(x.shape[0],-1))
            x = self.upsample(x)
            
            x=self.pool(self.nl(self.bn3(self.conv3(x))))
            x_list.append(x.detach().clone().cpu().reshape(x.shape[0],-1))
            x = self.upsample(x)
            
            x=self.pool(self.nl(self.bn4(self.conv4(x))))
            x_list.append(x.detach().clone().cpu().reshape(x.shape[0],-1))
            x = self.upsample(x)
            
            x=self.pool(self.nl(self.bn5(self.conv5(x))))
            x_list.append(x.detach().clone().cpu().reshape(x.shape[0],-1))
            x = self.upsample(x)
            
            x=self.pool(self.nl(self.bn6(self.conv6(x))))
            x_list.append(x.detach().clone().cpu().reshape(x.shape[0],-1))
            x = self.upsample(x)
            
            x=self.pool(self.nl(self.bn7(self.conv7(x))))
            x_list.append(x.detach().clone().cpu().reshape(x.shape[0],-1))
            x = self.upsample(x)
            x = x.reshape(x.shape[0],-1)
            x = self.fc1(x)
            x_list.append(x.detach().clone().cpu().reshape(x.shape[0],-1))
        return None,None, x_list,None