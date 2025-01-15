from dataset import dataset,dataset_attack
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
from math import isnan
from models import vgg, ResidualBlock, ResNet
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import random
from torch.utils.data import DataLoader,Dataset, SubsetRandomSampler
from torchvision.datasets import ImageFolder
torch.cuda.set_device(1)
BATCH_SIZE = 512
EPOCH = 30
dir ='finetune_models_tinyImageNet'

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
    
    return att_loader, val_loader


def finetune_model(iter,model_name,teacher,cls):

    teacher = teacher.cuda()
    teacher.train()
    accu_best = 0

    # train_data = dataset_attack('dataset_defend_1.h5', 'dataset_attack_1.h5', train=False)
    # images,labels = attackdataset_cifar100()
    # images,labels = attackdataset_tinyImageNet()
    # print(images.shape,labels.shape)
    # train_data = torch.utils.data.TensorDataset(torch.tensor(images), torch.tensor(labels))
    # train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    # transform_test = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # testset = torchvision.datasets.CIFAR10(
    #     root='./data', train=False, download=True, transform=transform_test)
    # #加载cifar100
    # transform = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    # ])
    # testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    # #加载tinyimagenet
  
    train_loader,test_loader = attackdataset_tinyImageNet()

    loss_func = torch.nn.CrossEntropyLoss()

    if cls == 'all':
        optimizer = optim.SGD(teacher.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
    elif cls == 'last':
        optimizer = optim.SGD(teacher.fc.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)

    for epoch in range(EPOCH):

        teacher.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, b_y = x.cuda(), y.cuda()
            teacher_output = teacher(b_x)
            loss = loss_func(teacher_output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i % 20 == 0):
                print("Epoch:", epoch + 1, "iteration:", i, "loss:", loss.data.item())
        teacher.eval()
        num = 0
        total_num = 0
        for i, (x, y) in enumerate(test_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, b_y = x.cuda(), y.cuda()
            output = teacher(b_x)
            pred = torch.max(output, 1)[1].data.squeeze()
            num += (pred == b_y).sum().item()
            total_num += pred.shape[0]


        accu1 = num / total_num

        print("Epoch:", epoch + 1, "accuracy:", accu1)

        if accu1 > accu_best:
            accu_best = accu1
            torch.save(teacher.state_dict(), os.path.join(dir, "finetune_{}_{}_{}.pth".format(model_name,cls,iter)))
            


    return accu_best

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    if os.path.exists(dir) == 0:
        os.mkdir(dir)
        
        
    # model = "vgg16"
    # teacher = vgg()
    # teacher = teacher.cuda()
    # in_feature = teacher.classifier[-1].in_features
    # teacher.classifier[-1] = torch.nn.Linear(in_feature, 10)
    # save_path = './model/{}Net_0.pth'.format(model)
    # teacher.load_state_dict(torch.load(save_path))
    # teacher.eval()
    # model_name=save_path.split("/")[-1].split(".")[0]
    # print(model_name)
    model_name = 'resnet'
    teacher = torchvision.models.resnet18(pretrained=False)
    in_feature = teacher.fc.in_features
    teacher.fc = torch.nn.Linear(in_feature, 200)
    save_path = "model_tinyImageNet/resnet_0.pth"
    teacher.load_state_dict(torch.load(save_path)) 
    teacher.eval()
    teacher = teacher.cuda()#to(device)

    for iter in range(20):
        iters = iter

        if iters<10:
            cls = 'all'
        elif 10<=iters<20:
            cls = 'last'

        print("Beigin training model:", iters,"Model:",cls)
        accu = finetune_model(iter,model_name,teacher,cls)
        teacher.load_state_dict(torch.load(save_path))
        print("{}_{} has been finetuned and the accuracy is {}".format(model_name, cls ,  accu))
