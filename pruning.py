import argparse
import sys
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
import torch.nn as nn
from dataset import dataset,dataset_attack
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import random
import h5py
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.prune as prune
import torch.utils.data as Data
from copy import deepcopy
from models import vgg, ResidualBlock, ResNet
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import random
from torch.utils.data import DataLoader,Dataset, SubsetRandomSampler
from torchvision.datasets import ImageFolder
torch.cuda.set_device(0)
dir = 'pruning_models_tinyImageNet'

BATCH_SIZE = 512
LR = 0.001
prune_ratio = 0.90
import numpy as np
import random
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
def attackdataset_cifar100():
    # 定义数据转换
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载 CIFAR-100 训练集
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    # 获取训练集大小和类别数量
    num_samples = len(train_dataset)
    num_classes = len(set(train_dataset.targets))

    # 定义要分割的训练集数量
    print(num_samples,num_classes)
    num_splits = 2

    # 计算每个部分的数据数量
    subset_sizes = [num_samples // num_splits]*num_splits

    # 将训练集等分为 num_splits 个部分
    random.seed(1)
    subsets = random_split(train_dataset, subset_sizes)
    # 创建 HDF5 文件并保存等分后的训练集
    images_data = []
    labels_data = []
    for i, subset in enumerate(subsets):
        images_data.append(torch.stack([sample[0] for sample in subset]))
        labels_data.append(torch.tensor([sample[1] for sample in subset], dtype=torch.long))
       
   

    a = images_data[0][20000:]
    b = images_data[1][20000:]
    
    images = np.concatenate((a, b),axis=0)
    
    x = labels_data[0][20000:]
    y = labels_data[1][20000:]
    # print(x)
    labels = np.concatenate((x, y),axis=0) 
    return images,labels

def attackdataset_tinyImageNet():
    # 定义数据转换
    transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小
    transforms.ToTensor(),         # 将图像转为Tensor
    ])
    data_dir = './data/tiny-imagenet-200'
    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    # 获取训练集大小和类别数量
    num_samples = len(train_dataset)
    num_classes = len(set(train_dataset.targets))

    # 定义要分割的训练集数量
    print(num_samples,num_classes)
    torch.manual_seed(1)  # 设置随机种子以确保每次运行结果一致
    indices = torch.randperm(num_samples).tolist()
    random.seed(1)
    val_sampler = SubsetRandomSampler(indices[80000:90000])
    
    val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
    
    random.seed(1)
    att_sampler = SubsetRandomSampler(indices[90000:])
    
    att_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=att_sampler)
    print(len(att_sampler))
    return att_loader, val_loader

def idx_change(idx, neuron_num):
    total = 0
    for i in range(neuron_num.shape[0]):
        total += neuron_num[i]
        if idx < total:
            layer_num = i
            layer_idx = idx - (total - neuron_num[i])
            break
    return layer_num, layer_idx

def prune_neuron(mask_list, idx, neuron_num):
    layer_num, layer_idx = idx_change(idx, neuron_num)
    mask_list[layer_num].weight_mask[layer_idx] = 0

class FeatureHook():

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        self.output = output

    def close(self):
        self.hook.remove()

def find_smallest_neuron(hook_list,prune_list):
    activation_list = []
    for j in range(len(hook_list)):
        activation = hook_list[j].output
        for i in range(activation.shape[1]):
            activation_channel = torch.mean(torch.abs(activation[:,i,:,:]))
            activation_list.append(activation_channel)

    activation_list1 = []
    activation_list2 = []

    for n, data in enumerate(activation_list):
        if n in prune_list:
            pass
        else:
            activation_list1.append(n)
            activation_list2.append(data)

    activation_list2 = torch.tensor(activation_list2)
    prune_num = torch.argmin(activation_list2)
    prune_idx = activation_list1[prune_num]

    return prune_idx

def finetune_step(model, dataloader, test_loader,init_val,remove_num, criterion):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=2e-3)
    best_val=value(model, test_loader)
    for ep in range(1):
        
        for j,(inputs,labels) in enumerate(dataloader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            labels=labels.long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            new_val = value(model, test_loader)
            print(j)
            if   new_val>best_val:
                torch.save(model, os.path.join(dir, "pruning_{}_remove_neuron_{}.pth".format(model_name, remove_num )))
                best_val = new_val 
            # print( new_val,best_val)
    return best_val


def value(model, dataloader):
    model.eval()
    num = 0
    total_num = 0
    for i, (x, y) in enumerate(dataloader):
        y = y.long()
        b_x, b_y = x.cuda(), y.cuda()
        output = model(b_x)
        pred = torch.max(output, 1)[1].data.squeeze()
        num += (pred == b_y).sum().item()
        total_num += pred.shape[0]

    accu = num / total_num
    return accu

def run_model(model, dataloader):
    model.eval()
    for i, (x, y) in enumerate(dataloader):
        y = y.long()
        b_x, b_y = x.cuda(), y.cuda()
        output = model(b_x)
        pred = torch.max(output, 1)[1].data.squeeze()

def idx_change(idx, neuron_num):
    total = 0
    for i in range(neuron_num.shape[0]):
        total += neuron_num[i]
        if idx < total:
            layer_num = i
            layer_idx = idx - (total - neuron_num[i])
            break
    return layer_num, layer_idx

def prune_neuron(mask_list, idx, neuron_num):
    layer_num, layer_idx = idx_change(idx, neuron_num)
    mask_list[layer_num].weight_mask[layer_idx] = 0

def plot_figure(mem,length, model_name):
    plt.figure(1)
    acc = np.squeeze(mem)
    plt.plot(np.squeeze(np.array(acc)[:, 0])/length, np.squeeze(np.array(acc)[:, 1]), 'b',label='Clean Classification Accuracy')
    plt.xlabel("Ratio of Neurons Pruned")
    plt.ylabel("Rate")
    plt.legend()
    plt.savefig('./pruning/pruning_{}.png'.format(model_name))
    plt.show()


def fine_pruning(model, train_loader, test_loader, model_name):
    model = model.cuda()
    module_list = []
    neuron_num = []
    hook_list = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module_list.append(module)
            neuron_num.append(module.out_channels)
            hook_list.append(FeatureHook(module))

    neuron_num = np.array(neuron_num)
    max_id = np.sum(neuron_num)

    neuron_list = []
    mask_list = []
    for i in range(neuron_num.shape[0]):
        neurons = list(range(neuron_num[i]))
        neuron_list.append(neurons)
        prune_filter = prune.identity(module_list[i], 'weight')
        mask_list.append(prune_filter)

    prune_list = []
    init_val = value(model, test_loader)
    print(init_val)
    acc = []
    length = deepcopy(len(neuron_list))
    total_length = 0
    for i in range(length):
        total_length += len(neuron_list[i])
    print("Total number of neurons is",total_length)
    for i in range(int(np.floor(0.5*total_length))):
        # if i % 20 == 0:
        #     run_model(model, train_loader)
        idx = find_smallest_neuron(hook_list, prune_list)
        prune_list.append(idx)
        prune_neuron(mask_list, idx, neuron_num)
        if i % 50 == 0:
            best_val = finetune_step(model, train_loader,test_loader,init_val,i, criterion=torch.nn.CrossEntropyLoss())
            acc.append([i, best_val])
            
            print("remove_num_{}_best_val:{}".format(i,best_val))

    mem = np.array([acc])
    if os.path.exists("./pruning") == 0:
        os.mkdir("./pruning")
    np.save("./pruning/{}_acc".format(model_name), mem)
    return mem,length


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    if os.path.exists(dir) == 0:
        os.mkdir(dir)
        print("Making directory!")


    # transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
#     #加载cifar100
#     transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
# ])
#     testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
# #加载tinyimagenet
 
    train_loader,test_loader = attackdataset_tinyImageNet()
    
    
    save_path = "model_tinyImageNet/resnet_0.pth"
    print(save_path)
    teacher = torchvision.models.resnet18(pretrained=False)
    in_feature = teacher.fc.in_features
    teacher.fc = torch.nn.Linear(in_feature, 200)
    teacher.load_state_dict(torch.load(save_path)) 
    teacher.eval()
    teacher = teacher.cuda()#to(device)
    model_name=save_path.split("/")[-1].split(".")[0]
    mem,length = fine_pruning(teacher, train_loader, test_loader, model_name)
    plot_figure(mem, length ,model_name)
        
