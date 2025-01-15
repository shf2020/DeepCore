import h5py
import torch
from torch.utils.data import DataLoader,Dataset
import numpy as np
from torchvision import transforms
import cv2 
import matplotlib.pyplot as plt
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from PIL import Image
import random

# random.seed(5)

def normalize(image):
    image_data = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
    image_data = np.array(image_data)
    img_copy = torch.zeros(image.shape)
    for i in range(3):
        img_copy[ i, :, :] = (image[ i, :, :] - image_data[0, i])/image_data[1,i]

    return img_copy


transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
# cifar100定义数据转换
transform_cifar100 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

transform_CIFAR100 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.53561753,0.48983628,0.42546818), (0.26656017,0.26091456,0.27394977))
    ])

transform_CIFAR10C_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4645897160947712,0.6514782475490196,0.5637088950163399), (0.18422159112571024, 0.3151505122530825, 0.26127269383599344))
    ])

transform_CIFAR10C_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4645897160947712,0.6514782475490196,0.5637088950163399), (0.18422159112571024, 0.3151505122530825, 0.26127269383599344))
    ])


class dataset(Dataset):
    def __init__(self, name1, name2, rate, train=False):
        super(dataset, self).__init__()
        self.name1 = name1
        self.name2 = name2
        self.data1 = h5py.File(os.path.join("data",name1), 'r')
        self.data2 = h5py.File(os.path.join("data",name2), 'r')
        # print(self.data1)
        
        random.seed(5)
        List1 = random.sample(range(0,20000),int(20000*rate))
        random.seed(1)
        List2 = random.sample(range(0,20000),int(20000-20000*rate))
        # print(List1[0:10],List2[0:10])
        a = np.array(self.data1['/data'])[0:20000]
        b = np.array(self.data2['/data'])[0:20000]
        # self.data1.close()
        # self.data2.close()
        # print(a[0].shape)
        A = []
        B = []
        if len(List1) != 0:
            for i in List1:
                A.append(a[i])
        if len(List2) != 0:       
            for i in List2:
                B.append(b[i])
        print("训练集从一个数据集随机选择图片个数:", len(A), "从另一个互不相交的数据集随机选择图片个数:", len(B), "所以与受害者模型训练数据集重合占比为", rate )    
        if len(A) == 0:
            self.images = np.array(B)
        elif len(B) == 0:
            self.images = np.array(A)
        else:
            self.images = np.concatenate((A, B),axis=0)
        
        x = np.array(self.data1['/label'])[0:20000]
        y = np.array(self.data2['/label'])[0:20000]
        # print(x)
        X = []
        Y = []
        if len(List1) != 0:
            for i in List1:
                X.append(x[i])
        if len(List2) != 0:
            for i in List2:
                Y.append(y[i])
        # print(len(X),len(Y)) 
        if len(X) == 0:
            self.labels =  np.array(Y)
        elif len(Y) == 0:
            self.labels =  np.array(X)
        else:
            self.labels = np.concatenate((X, Y),axis=0) 

    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, item):

        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :]*255,dtype='uint8')
        image = transform_train(image)
        return [image,label]

class dataset_cifar100(Dataset):
    def __init__(self, name1, name2, rate, train=False):
        super(dataset, self).__init__()
        self.name1 = name1
        self.name2 = name2
        self.data1 = h5py.File(os.path.join("data",name1), 'r')
        self.data2 = h5py.File(os.path.join("data",name2), 'r')
        # print(self.data1)
        
        random.seed(5)
        List1 = random.sample(range(0,20000),int(20000*rate))
        random.seed(1)
        List2 = random.sample(range(0,20000),int(20000-20000*rate))
        # print(List1[0:10],List2[0:10])
        a = np.array(self.data1['/data'])[0:20000]
        b = np.array(self.data2['/data'])[0:20000]
      
        A = []
        B = []
        if len(List1) != 0:
            for i in List1:
                A.append(a[i])
        if len(List2) != 0:       
            for i in List2:
                B.append(b[i])
        print("训练集从一个数据集随机选择图片个数:", len(A), "从另一个互不相交的数据集随机选择图片个数:", len(B), "所以与受害者模型训练数据集重合占比为", rate )    
        if len(A) == 0:
            self.images = np.array(B)
        elif len(B) == 0:
            self.images = np.array(A)
        else:
            self.images = np.concatenate((A, B),axis=0)
        
        x = np.array(self.data1['/label'])[0:20000]
        y = np.array(self.data2['/label'])[0:20000]
        # print(x)
        X = []
        Y = []
        if len(List1) != 0:
            for i in List1:
                X.append(x[i])
        if len(List2) != 0:
            for i in List2:
                Y.append(y[i])
        # print(len(X),len(Y)) 
        if len(X) == 0:
            self.labels =  np.array(Y)
        elif len(Y) == 0:
            self.labels =  np.array(X)
        else:
            self.labels = np.concatenate((X, Y),axis=0) 

    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, item):

        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :]*255,dtype='uint8')
        image = transform_cifar100(image)
        return [image,label]

class dataset_attack(Dataset):
    def __init__(self, name1, name2, train=False):
        super(dataset_attack, self).__init__()
        self.name1 = name1
        self.name2 = name2
        self.data1 = h5py.File(os.path.join("data",name1), 'r')
        self.data2 = h5py.File(os.path.join("data",name2), 'r')
        print(self.data1)
        
        
        # print(len(List1),len(List2),List1[0],List2[0])
        a = np.array(self.data1['/data'])[20000:25000]
        b = np.array(self.data2['/data'])[20000:25000]
        
        
        self.images = np.concatenate((a, b),axis=0)
        
        x = np.array(self.data1['/label'])[20000:25000]
        y = np.array(self.data2['/label'])[20000:25000]
        print("与训练集不重叠的攻击者数据集数量：",len(x)+len(y))
        self.labels = np.concatenate((x, y),axis=0) 

    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, item):

        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :]*255,dtype='uint8')
        image = transform_train(image)
        return [image,label]

class dataset_attack1(Dataset):
    def __init__(self, name1, name2, train=False):
        super(dataset_attack1, self).__init__()
        self.name1 = name1
        self.name2 = name2
        self.data1 = h5py.File(os.path.join("data",name1), 'r')
        self.data2 = h5py.File(os.path.join("data",name2), 'r')
        print(self.data1)
        
        
        # print(len(List1),len(List2),List1[0],List2[0])
        a = np.array(self.data1['/data'])[15000:25000]
        b = np.array(self.data2['/data'])[15000:25000]
        
        
        self.images = np.concatenate((a, b),axis=0)
        
        x = np.array(self.data1['/label'])[15000:25000]
        y = np.array(self.data2['/label'])[15000:25000]
        print("与训练集不重叠的攻击者数据集数量：",len(x)+len(y))
        self.labels = np.concatenate((x, y),axis=0) 

    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, item):

        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :]*255,dtype='uint8')
        image = transform_train(image)
        return [image,label]

class dataset1(Dataset):
    def __init__(self, name,train=False):
        super(dataset1, self).__init__()
        self.name = name
        self.data = h5py.File(os.path.join("data",name), 'r')
        self.images = np.array(self.data['/data'])
        self.labels = np.array(self.data['/label'])

    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, item):

        label = torch.tensor(self.labels[item])
        image = np.array(self.images[item, :, :, :]*255,dtype='uint8')
        image = transform_test(image)

        return [image,label]