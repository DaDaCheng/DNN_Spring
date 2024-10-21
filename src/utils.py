from scipy import stats

import torch
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import log_loss

import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import log_loss

from torch.utils.data import Dataset
import torch.nn.functional as F


def smeasure_logD(X,labels,detail=False):
    device=X.device
    number_label=int(labels.max()+1)
    numb_feature=X.shape[1]
    X_mean=torch.mean(X,dim=0)
    SSB=torch.zeros((numb_feature,numb_feature)).to(device)
    SSW=torch.zeros((numb_feature,numb_feature)).to(device)
    for i in range(number_label):        
        X_per=X[labels==i,:]
        X_Data_per_label_mean=torch.mean(X_per,dim=0)
        dx=(X_Data_per_label_mean-X_mean).reshape(-1,1)
        SSB=SSB+X_per.shape[0]*dx@dx.T
        X_Data_per_label_mean_keepdim=torch.mean(X_per,dim=0,keepdim=True)
        ddx=X_per-X_Data_per_label_mean_keepdim
        SSW=SSW+ddx.T@ddx
    SSB=SSB/X.shape[0]
    SSW=SSW/X.shape[0]
    D=torch.trace(SSW@torch.linalg.pinv(SSB))    
    if detail:
        invD=torch.trace(SSB@torch.linalg.pinv(SSW))
        return np.log(D.cpu().numpy()),np.log(invD.cpu().numpy()),np.log(torch.trace(SSW).cpu().numpy()),np.log(torch.trace(SSB).cpu().numpy())
    else:
        return np.log(D.cpu().numpy())
    

def smeasure_sqrtLinR(X,y):
    X=np.array(X)
    y=np.array(y)
    reg = LinearRegression().fit(X, y)
    yp=reg.predict(X).reshape(-1)
    return np.sqrt(np.mean((yp-y.reshape(-1))**2)/np.mean(y.reshape(-1)**2))

def smeasure_sqrtLinR_onehot(X,y):
    X=np.array(X)
    y =F.one_hot(y.to(torch.int64).reshape(-1), num_classes=10).float()
    y=np.array(y)
    reg = LinearRegression().fit(X, y)
    yp=reg.predict(X).reshape(-1)
    return np.sqrt(np.mean((yp-y.reshape(-1))**2))

def smeasure_LogicR(X,y):
    X=np.array(X)
    y=np.array(y)
    reg = LogisticRegression(max_iter=10000).fit(X, y)
    #reg = LogisticRegression().fit(X, y)
    y_pred_proba=reg.predict_proba(X)
    return log_loss(y, y_pred_proba)



def get_full_data(train_gen,size,device):
    ycat_list=torch.tensor([])
    images_cat=torch.tensor([]).to(device)
    for i ,(images,labels) in enumerate(train_gen):
        images = images.view(-1,size**2).to(device)
        labels = labels.detach().clone().cpu()
        ycat_list = torch.cat([ycat_list,labels.reshape(-1)],dim=0)
        images_cat = torch.cat([images_cat,images],dim=0)
    return images_cat, ycat_list

def get_seperate_data_two_input(model, images_cat, ycat_list):
    Xcat_list=[]
    depth=model.depth
    with torch.no_grad():
        model.eval()
        Xcat_list=model(images_cat)
        for k in range(len(Xcat_list)):
            Xcat_list[k]=Xcat_list[k].detach().clone().cpu()
    return Xcat_list,ycat_list

def get_seperate_data(model,train_gen,size,device,full_batch=True):
    Xcat_list=[]
    depth=model.depth
    ycat_list=torch.tensor([])
    with torch.no_grad():
        if full_batch:
            model.eval()
            images_cat=torch.tensor([]).to(device)
            for i ,(images,labels) in enumerate(train_gen):
                images = images.view(-1,size**2).to(device)
                labels = labels.detach().clone().cpu()
                ycat_list = torch.cat([ycat_list,labels.reshape(-1)],dim=0)
                images_cat = torch.cat([images_cat,images],dim=0)        
                #print(images_cat.shape,images.shape)
            Xcat_list=model(images_cat)
            for k in range(len(Xcat_list)):
                Xcat_list[k]=Xcat_list[k].detach().clone().cpu()
        else:
            Xcat_list=[]
            for iii in range(depth+1):
                Xcat_list.append(torch.tensor([]))
            for i ,(images,labels) in enumerate(train_gen):
                model.eval()
                images = images.view(-1,size**2).to(device)
                labels = labels.detach().clone().cpu()
                X_= model(images)
                for iii in range(len(X_)):
                    X_[iii]=X_[iii].detach().clone().cpu()
                ycat_list=torch.cat([ycat_list,labels.reshape(-1)],dim=0)
                for k in range(len(X_)):
                    Xcat_list[k]=torch.cat([Xcat_list[k],X_[k]],dim=0)
    return Xcat_list,ycat_list

def get_acc(model,test_gen,dim,device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images,labels in test_gen:
            images = images.view(-1,dim).to(device)
            labels = labels.to(device)
            output = model(images)
            _, predicted = torch.max(output,1)
            correct += (predicted == labels).sum()
            total += labels.size(0)
    return ((100*correct)/(total+1))







class GaussianMixtureDataset(Dataset):
    def __init__(self, num_classes, samples_per_class, num_features):
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.num_features = num_features
        self.size=int(np.sqrt(num_features))
        self.means = torch.randn(num_classes, num_features)
        self.std_devs = torch.abs(torch.randn(num_classes, num_features))

        self.labels = torch.zeros(num_classes * samples_per_class, dtype=torch.long)
        self.dataset = torch.zeros(num_classes * samples_per_class, num_features)

        self.generate_data()
        self.shuffle_data()

    def generate_data(self):
        for c in range(self.num_classes):
            start = c * self.samples_per_class
            end = (c + 1) * self.samples_per_class
            self.labels[start:end] = c
            for n in range(self.samples_per_class):
                self.dataset[start + n, :] = self.means[c, :] + self.std_devs[c, :] * torch.randn(1, self.num_features)

    def shuffle_data(self):
        indices = torch.randperm(self.dataset.size(0))
        self.dataset = self.dataset[indices]
        self.labels = self.labels[indices]

    def __len__(self):
        return self.dataset.size(0)

    def __getitem__(self, idx):
        return self.dataset[idx].reshape(1,self.size,self.size), self.labels[idx]
    

def v_and_c(data,remove_last_layer=True):
    #compute variance and curvature 
    if remove_last_layer:
        data=data[:-1]
    data_len=len(data)
    d_data=[]
    for i in range(data_len-1):
        d_data.append(data[i]-data[i+1])
    dd_data=[]
    for i in range(data_len-2):
        dd_data.append(d_data[i]-d_data[i+1])
    d_data=np.array(d_data)
    dd_data=np.array(dd_data)
    return np.std(d_data),np.sum(dd_data)
    