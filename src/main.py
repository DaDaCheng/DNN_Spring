import os
import torch
import torch.optim as optim
import argparse
import torch.nn as nn
from utils import *
from optimizer import SGLD,SGLD_RMS
from dataloader import load_real, load_GMM
from model import MLP
import copy
import argparse
import random
import torch.optim as optim
import matplotlib.pyplot as plt
from distutils.util import strtobool
from pyhessian import hessian


def str2bool(v):
    return bool(strtobool(v))
parser = argparse.ArgumentParser()
parser.add_argument('--net_outdim', type=int, default=10)
parser.add_argument('--net_depth', type=int, default=8)
parser.add_argument('--net_dim', type=int, default=100)
parser.add_argument('--net_bias', type=bool, default=True)
parser.add_argument('--net_isbatchnorm', type=str2bool, nargs='?', const=True, default=True)         
parser.add_argument('--net_p', type=float, default=0,help='drop out ratio')
parser.add_argument('--net_act', type=str, default='LeakyReLU')
parser.add_argument('--net_alpha', type=float, default=1,help='negative slope for LeakyReLU')

parser.add_argument('--init_scale', type=str, default='default') 
parser.add_argument('--init_zerobias', type=str2bool, nargs='?', const=True, default=True) 
parser.add_argument('--init_constant', type=float, default=1) 

parser.add_argument('--data_dataset', type=str, default='MNIST', help='MNIST,CIFAR10, FakeData, FashionMNIST')
parser.add_argument('--data_dim', type=int, default=100)
parser.add_argument('--data_len', type=int, default=2000)
parser.add_argument('--data_batchsize', type=int, default=100)
parser.add_argument('--data_gmmmu', type=float, default=1)
parser.add_argument('--data_maxlabel', type=int, default=10)


parser.add_argument('--train_lr', type=float, default=0.001)
parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--train_opt', type=str, default='Adam')
parser.add_argument('--train_losstype', type=str, default='CE',help='MSE, MSE_onehot, CE')
parser.add_argument('--train_noise', type=float, default=0)


parser.add_argument('--seed', type=int, default=0,help="random seed")
parser.add_argument('--group', type=str, default='default')
parser.add_argument('--test', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--tag', type=str, default='b')
#parser.add_argument('--device_id', type=int, default=0)
args = parser.parse_args()


print(args)
seed=args.seed
if seed is None:
    seed=random.randint(100,10000)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def add_label_noise(labels, noise_level=0.1):
    noisy_labels = labels.clone()
    num_samples = len(labels)
    num_noisy = int(noise_level * num_samples)
    noisy_indices = torch.randperm(num_samples)[:num_noisy]
    random_labels = torch.randint(0, 10, (num_noisy,))
    noisy_labels[noisy_indices] = random_labels
    return noisy_labels


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
losstype=args.train_losstype
dataset=args.data_dataset

if dataset in ['MNIST','CIFAR10','FakeData','FashionMNIST']:
    real_dataset=True
    train_loader, test_loader,val_loader=load_real(args)
        
# Define MLP    
model=MLP(args)
model_init=copy.deepcopy(model)
maxlabel=args.data_maxlabel
dim=args.data_dim
nCE=nn.CrossEntropyLoss()


def MSE_loss(y,y_label):
    return torch.mean((y-y_label)**2)

def MSE_onehot_loss(y,y_label):
    y_l=torch.nn.functional.one_hot(y_label,num_classes=maxlabel)
    return torch.mean((y-y_l)**2)

def CE_loss(y,y_label):
    y_label=y_label.long()
    return nCE(y,y_label)

if losstype=="MSE":
    loss_func=MSE_loss
if losstype=="MSE_onehot":
    loss_func=MSE_onehot_loss
if losstype=="CE":
    loss_func=CE_loss

X_cat=torch.tensor([])
y_cat=torch.tensor([])

for batch_idx, (X, y) in enumerate(train_loader, 1):
    X,y=X.reshape(-1,dim),y
    y_cat = torch.cat([y_cat,y.reshape(-1)],dim=0)
    X_cat = torch.cat([X_cat,X],dim=0)    
    
opt=args.train_opt
lr= args.train_lr
noise= args.train_noise
epoch=args.train_epoch

if opt=='SGD':
    optimizer = optim.SGD(model.parameters(), lr=lr)
if opt=='Adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)
if opt=='SGLD':
    from optimizer import SGLD
    optimizer = SGLD(model.parameters(), lr=lr,noise=noise)
if opt=='SGLD_RMS':
    from optimizer import SGLD_RMS
    optimizer = SGLD_RMS(model.parameters(), lr=lr,noise=noise)
record_loss=[]
record_metric=[]
model=model.to(device)




previous_loss=1000
Dmp_list_list=[]
val_acc=0
for epoch_id in range(epoch):
    loss_epoch=0
    model.train()
    for batch_idx, (X, y) in enumerate(train_loader, 1):
        if args.tag=='ns_nl':
            y = add_label_noise(y, noise)
            X=X+torch.randn_like(X)*(torch.std(X).item())*noise
        X,y=X.reshape(-1,dim).to(device),y.to(device)

        output = model(X)
        optimizer.zero_grad()
        loss=loss_func(output,y)
        loss.backward()
        optimizer.step()
        loss_epoch=loss_epoch+loss.item()*len(y)
    loss_epoch=loss_epoch/args.data_len
       

    with torch.no_grad():
        print(epoch_id,loss.item())
        if loss_epoch<1e-4:
            break
        if np.abs(previous_loss-loss_epoch)<1e-10:
            break
    previous_loss=loss_epoch
    
# compute the separation measure       
model.eval()
model_temp=copy.deepcopy(model)
model_temp=model_temp.to('cpu')
model_temp.eval()
#model.eval()
output_before_list,output_mid_list,output_after_list,x=model_temp.get_mid_output(X_cat)
Dmp_list=[]
Ds_list=[]
iDmp_list=[]
for xmid in output_after_list:
    logD,ilogD,logD_w,logD_b=smeasure_logD(xmid.cpu(),y_cat,detail=True)
    Dmp=logD
    iDmp=ilogD
    Ds=logD_w-logD_b
    Dmp_list.append(logD)
    Ds_list.append(Ds)
    iDmp_list.append(ilogD)

v_mp,c_mp=v_and_c(Dmp_list,remove_last_layer=True)
v_s,c_s=v_and_c(Ds_list,remove_last_layer=True)                                        
acc=get_acc(model, test_loader, dim,device)
print(epoch_id,loss.item(),acc.item(),v_mp,v_s)
model.eval()
hessian_comp = hessian(model, loss_func, data=(X_cat, y_cat), cuda=torch.cuda.is_available())
top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
print("The top Hessian eigenvalue of this model is %.4f"%top_eigenvalues[-1])
trace = hessian_comp.trace()
print("The trace of this model is: %.4f"%(np.mean(trace)))

import sys
log_file = open('save/PD/R_'+args.tag+'.txt', 'a')
sys.stdout = log_file
sys.setrecursionlimit(1000000)
out_list=[seed]+[lr]+[args.data_batchsize]+[args.net_alpha]+[args.net_p]+[acc.item()]+[v_mp,c_mp,v_s,c_s]+Dmp_list+Ds_list+iDmp_list+[noise]+[top_eigenvalues[-1]]+[np.mean(trace)]
print(out_list)
sys.stdout = sys.__stdout__
log_file.close()
