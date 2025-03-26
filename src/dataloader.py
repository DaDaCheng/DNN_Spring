
import torchvision.datasets as dsets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import torch

def load_real(args):
    size = int(np.sqrt(args.data_dim))
    data_len=args.data_len
    data_dataset=args.data_dataset
    batch_size=args.data_batchsize
    maxlabel=args.data_maxlabel
    if data_dataset=='MNIST':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
            transforms.Resize(size=size),
        ])
    
        train_data = dsets.MNIST(root = './data', train = True, transform = transform, download = True)
        train_data, val_data = torch.utils.data.random_split(train_data, [data_len, 60000-data_len], generator=torch.Generator().manual_seed(1))
        
        val_data,_ = torch.utils.data.random_split(val_data, [data_len, 60000-data_len-data_len], generator=torch.Generator().manual_seed(1))
        
        test_data = dsets.MNIST(root = './data', train = False, transform = transform)
        
        if maxlabel==2:
            indices = [i for i, (_, label) in enumerate(train_data) if label in [0, 1]]
            train_data = Subset(train_data, indices)
            indices = [i for i, (_, label) in enumerate(test_data) if label in [0, 1]]
            test_data = Subset(test_data, indices)
            
    if data_dataset=='FashionMNIST':
        transform=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Resize(size=size,antialias=True),
            transforms.Normalize((0,), (1,)),
            transforms.Resize(size=size)
        ])
    
        train_data = dsets.FashionMNIST(root = './data', train = True, transform = transform, download = True)
        train_data, val_data = torch.utils.data.random_split(train_data, [data_len, 60000-data_len], generator=torch.Generator().manual_seed(1))
        val_data,_ = torch.utils.data.random_split(val_data, [data_len, 60000-data_len-data_len], generator=torch.Generator().manual_seed(1))
        
        test_data = dsets.FashionMNIST(root = './data', train = False, transform = transform)
        
        if maxlabel==2:
            indices = [i for i, (_, label) in enumerate(train_data) if label in [0, 1]]
            train_data = Subset(train_data, indices)
            indices = [i for i, (_, label) in enumerate(test_data) if label in [0, 1]]
            test_data = Subset(test_data, indices)
        
    if data_dataset =='CIFAR10':
            
            class FlattenTransform:
                def __call__(self, image):
                    return image.view(-1) 
            transform=transforms.Compose([
                            transforms.ToTensor(),
                            #transforms.Grayscale(),
                            #transforms.Grayscale(num_output_channels=1),
                            transforms.Normalize(0,1),
                            transforms.Resize(size=size,antialias=True),
                            #FlattenTransform()
                        ])
            train_data = dsets.CIFAR10(root = './data', train = True,
                                    transform = transform, download = True)
            print('cifar10')
            train_data, val_data = torch.utils.data.random_split(train_data, [data_len, 50000-data_len],generator=torch.Generator().manual_seed(1))
            val_data,_ = torch.utils.data.random_split(val_data, [data_len, 50000-data_len-data_len], generator=torch.Generator().manual_seed(1))
            test_data = dsets.CIFAR10(root = './data', train = False, transform = transform)
            
            if maxlabel==2:
                indices = [i for i, (_, label) in enumerate(train_data) if label in [0, 1]]
                train_data = Subset(train_data, indices)
                indices = [i for i, (_, label) in enumerate(test_data) if label in [0, 1]]
                test_data = Subset(test_data, indices)



    if data_dataset =='CIFAR10_flat':
        class FlattenTransform:
            def __call__(self, image):
                return image.view(-1)
        print(size)
        transform = transforms.Compose([
            transforms.Resize((size, size)),  # Rescale the image to NxN
            transforms.ToTensor(),      # Convert image to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize the image
            FlattenTransform(),  # Flatten to a vector of size 3*N^2
        ])
        
        train_data = dsets.CIFAR10(root = './data', train = True,
                                transform = transform, download = True)
        print('cifar10_flat')
        train_data, val_data = torch.utils.data.random_split(train_data, [data_len, 50000-data_len],generator=torch.Generator().manual_seed(1))
        val_data,_ = torch.utils.data.random_split(val_data, [data_len, 50000-data_len-data_len], generator=torch.Generator().manual_seed(1))
        test_data = dsets.CIFAR10(root = './data', train = False, transform = transform)
        
        if maxlabel==2:
            indices = [i for i, (_, label) in enumerate(train_data) if label in [0, 1]]
            train_data = Subset(train_data, indices)
            indices = [i for i, (_, label) in enumerate(test_data) if label in [0, 1]]
            test_data = Subset(test_data, indices)

    if data_dataset =='FakeData':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0,1),
        ])
        train_data = dsets.FakeData(size=data_len,image_size=[size,size],num_classes=maxlabel,transform = transform)
        test_data = train_data
        
    train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                                batch_size = batch_size,
                                                shuffle = True)
    
    test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                            batch_size = batch_size,
                                            shuffle = False)
    
    val_loader = torch.utils.data.DataLoader(dataset = val_data,
                                            batch_size = batch_size,
                                            shuffle = False)
    return train_loader, test_loader, val_loader



class VectorDataset(Dataset):
    def __init__(self, X, y):
        """
        Initialize the dataset with vector data and labels.
        
        Parameters:
        - X: A 2D list or array containing the vector data, shape (n_samples, n_features)
        - y: A list or array containing the labels, shape (n_samples,)
        """
        # Convert inputs to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Return a single sample from the dataset at the specified index.
        
        Parameters:
        - idx: Integer, index of the sample to return.
        """
        return self.X[idx], self.y[idx]





def load_GMM(args):

    batch_size=args.data_batchsize
    N=args.data_len
    hN=int(N/2)
    F=args.data_dim
    mu=args.data_gmmmu
    y=np.ones(hN)*1.0
    y=np.concatenate([y,-y],axis=0).reshape(N,1)
    u=np.random.randn(F,1)/np.sqrt(F)
    #u=np.ones((F,1))/np.sqrt(F)
    Omega=np.random.randn(N,F)
    X=np.sqrt(mu/N)*y@u.T+Omega/np.sqrt(F)
    X=torch.tensor(X,dtype=torch.float32)
    y=torch.tensor(y,dtype=torch.float32)
    u=torch.tensor(u,dtype=torch.float32)

    yy=((torch.tensor(y).reshape(-1)+1)/2).to(int)
    train_data=VectorDataset(X.numpy(),yy.numpy())
    train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                            batch_size = batch_size,
                                            shuffle = False)
    
    Omega=np.random.randn(N,F)
    X_=np.sqrt(mu/N)*y@u.T+Omega/np.sqrt(F)
    test_data=VectorDataset(X_.numpy(),yy.numpy())
    test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                        batch_size = batch_size,
                                        shuffle = False)
    return train_loader, test_loader