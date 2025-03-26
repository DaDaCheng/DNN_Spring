import torch
import torch.optim as optim
import argparse
import torch.nn as nn
import sys

sys.path.append('../../src')
sys.path.append('..')
from utils import *
from dataloader import load_real,load_GMM
import copy
import argparse
import random
import torch.optim as optim
from distutils.util import strtobool
import numpy as np
import torch
import torch.nn.functional as F
from model import CNN

def get_acc(model,test_gen,dim,device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images,labels in test_gen:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _, predicted = torch.max(output,1)
            correct += (predicted == labels).sum()
            total += labels.size(0)
    return ((100*correct)/(total+1))        
        
        
def str2bool(v):
    return bool(strtobool(v))
parser = argparse.ArgumentParser()
parser.add_argument('--net_dim', type=int, default=100)
parser.add_argument('--net_p', type=float, default=0)
parser.add_argument('--net_act', type=str, default='LeakyReLU')
parser.add_argument('--net_alpha', type=float, default=0)
parser.add_argument('--net_channel', type=int, default=20)
parser.add_argument('--data_dataset', type=str, default='CIFAR10', help='MNIST,CIFAR10, FakeData, FashionMNIST')
parser.add_argument('--data_dim', type=int, default=100)
parser.add_argument('--data_len', type=int, default=2560)
#parser.add_argument('--data_batchsize', type=int, default=2560)
parser.add_argument('--train_lr', type=float, default=0.0001)
parser.add_argument('--train_epoch', type=int, default=1000)
parser.add_argument('--train_opt', type=str, default='Adam')
parser.add_argument('--seed', type=int, default=1024,help="random seed")
parser.add_argument('--tag', type=str, default='b')
parser.add_argument('--data_maxlabel', type=int, default=10)
#parser.add_argument('--device_id', type=int, default=0)
args = parser.parse_args()
print(args)


seed=args.seed
if seed is None:
    seed=random.randint(100,10000)

#full batch
args.data_batchsize=args.data_len

random.seed(seed)
np.random.seed(seed)
_=torch.manual_seed(seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
dataset=args.data_dataset

if dataset in ['MNIST','CIFAR10','FakeData','FashionMNIST']:
    real_dataset=True
    train_loader,test_loader,val_loader=load_real(args)
elif dataset=='GMM':
    real_dataset=False
    args.data_maxlabel=2
    train_loader, test_loader=load_GMM(args)
print('dataset',dataset)
model=CNN(tau=args.net_alpha,width=args.net_channel).to(device)
opt=args.train_opt
lr= args.train_lr
epoch=args.train_epoch
dim=args.net_dim

for X_data,y_data in train_loader:
    X_data=X_data.to(device)
    y_data=y_data.to(device)
    break

if opt=='Adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)

nCE=nn.CrossEntropyLoss()
def CE_loss(y,y_label):
    y_label=y_label.long()
    return nCE(y,y_label)
loss_func=CE_loss

dp=args.net_p
model.dropout=nn.Dropout(dp)
print(X_data.shape)
for epoch_id in range(args.train_epoch+1):
    loss_epoch=0
    model.train()
    optimizer.zero_grad()
    output = model(X_data)
    loss=loss_func(output,y_data)
    lossiterm=loss.item()
    loss.backward()
    optimizer.step()
    
    print(epoch_id, lossiterm)
    if lossiterm<1e-15:
    #if lossiterm<0.99:
        print('break: loss=', lossiterm)
        break
    

with torch.no_grad():
    model.eval()
    model_temp=copy.deepcopy(model)
    model_temp=model_temp.to('cpu')
    model_temp.eval()
    output_before_list,output_mid_list,output_after_list,x=model_temp.get_mid_output((X_data).cpu())
    
    Dmp_list=[]
    Ds_list=[]
    iDmp_list=[]
    for xmid in output_after_list:
        logD,ilogD,logD_w,logD_b=smeasure_logD(xmid.cpu(),y_data.cpu(),detail=True)
        Dmp=logD
        iDmp=ilogD
        Ds=logD_w-logD_b
        Dmp_list.append(logD)
        Ds_list.append(Ds)
        iDmp_list.append(ilogD)
    
acc=get_acc(model, test_loader, dim,device)
            
import sys
log_file = open('save/PD/R_'+args.tag+'.txt', 'a')
sys.stdout = log_file
sys.setrecursionlimit(1000000)
out_list=[seed]+[dp]+[acc.item()]+Ds_list
out_list = [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in out_list]
print(out_list)
sys.stdout = sys.__stdout__
log_file.close()