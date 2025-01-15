from dataset import dataset_attack
from torch.utils.data import DataLoader,Dataset,random_split
import torch
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from model_load import load_model
from copy import deepcopy
from torch.autograd import Variable
import h5py
import time
from models import vgg, ResidualBlock, ResNet

import numpy as np
import random
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

print(torch.cuda.is_available())
print(torch.cuda.device_count())
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import random
from torch.utils.data import DataLoader,Dataset, SubsetRandomSampler
from torchvision.datasets import ImageFolder
torch.cuda.set_device(2)
# dir = "adv_models_cifar100"
dir = 'adv_models_tinyImageNet'
BATCH_SIZE = 512


def denormalize(image):
    image_data = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
    image_data = np.array(image_data)

    img_copy = torch.zeros(image.shape).cuda()
    for i in range(3):
        img_copy[:,i,:,:] = image[:,i,:,:]*image_data[1,i] + image_data[0,i]

    return img_copy

def normalize(image):
    image_data = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
    image_data = np.array(image_data)
    img_copy = torch.zeros(image.shape).cuda()
    for i in range(3):
        img_copy[:, i, :, :] = (image[:, i, :, :] - image_data[0, i])/image_data[1,i]

    return img_copy

def PGD(model,image,label):
    label = label.cuda()
    loss_func1 = torch.nn.CrossEntropyLoss()
    image_de = denormalize(deepcopy(image))
    image_attack = deepcopy(image)
    image_attack = image_attack.cuda()
    image_attack = Variable(image_attack, requires_grad=True)
    alpha = 1/256
    epsilon = 4/256

    for iter in range(30):
        image_attack = Variable(image_attack, requires_grad=True)
        output = model(image_attack)
        loss = -loss_func1(output,label)
        loss.backward()
        grad = image_attack.grad.detach().sign()
        image_attack = image_attack.detach()
        image_attack = denormalize(image_attack)
        image_attack -= alpha*grad
        eta = torch.clamp(image_attack-image_de,min=-epsilon,max=epsilon)
        image_attack = torch.clamp(image_de+eta,min=0,max=1)
        image_attack = normalize(image_attack)
    pred_prob = output.detach()
    pred = torch.argmax(pred_prob, dim=-1)
    acc_num = torch.sum(label == pred)
    num = label.shape[0]
    acc = acc_num/num
    acc = acc.data.item()

    return image_attack.detach(), acc

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



def adv_train(model,teacher,train_loader,test_loader,cls,iter):
    model = model.cuda()
    model.train()
    teacher.eval()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
    for epoch in range(2):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, _ = x.cuda(), y.cuda()
            teacher_output = teacher(b_x)
            pred = torch.max(teacher_output, 1)[1].data.squeeze().detach()
            x_adv,acc = PGD(model,b_x,pred)
            output = model(b_x)
            output_adv = model(x_adv)
            loss = loss_func(output, pred) + loss_func(output_adv, pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i % 20 == 0):
                print("Epoch:", epoch + 1, "iteration:", i, "loss:", loss.data.item(),"ASR:",1-acc)


        model.eval()
        num = 0
        total_num = 0

        for i, (x, y) in enumerate(test_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, b_y = x.cuda(), y.cuda()
            output = model(b_x)
            pred = torch.max(output, 1)[1].data.squeeze()
            num += (pred == b_y).sum().item()
            total_num += pred.shape[0]

        accu1 = num / total_num
        print("Epoch:", epoch + 1, "accuracy:", accu1)
    torch.save(model.state_dict(), os.path.join(dir, "model_adv_" + str(cls) + "_" + str(iter) + ".pth"))
    model = model.cpu()

    return accu1


if __name__ == "__main__":

    if os.path.exists(dir)==0:
        os.mkdir(dir)

    # model = "vgg16"
    # teacher = vgg()
    
    # model = "resnet50"
    # teacher =  ResNet(ResidualBlock, [2, 2, 2])
    
    model = 'resnet'
    teacher = torchvision.models.resnet18(pretrained=False)
    in_feature = teacher.fc.in_features
    teacher.fc = torch.nn.Linear(in_feature, 200)
    teacher.load_state_dict(torch.load("model_tinyImageNet/resnet_0.pth")) 
    teacher.eval()
    teacher = teacher.cuda()#to(device)
    models = []
    accus = []
    
    # transform_test = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # testset = torchvision.datasets.CIFAR10(
    #     root='./data', train=False, download=True, transform=transform_test)
#     #加载cifar100
#     transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
# ])
#     testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
#      #加载tinyimagenet

    train_loader,test_loader = attackdataset_tinyImageNet()
    for i in range(0,20):
        iters = i
        # if iters < 5:
        #     cls = 'vgg'
        # elif 5<=iters<10:
        #     cls = 'resnet'
        # elif 10 <= iters < 15:
        #     cls = 'dense'
        # elif 15 <= iters:
        #     cls = 'mobile'
        if iters < 10:
            cls = 'resnet'
        elif 10<=iters<20:
            cls = 'dense'
            
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 200)
        model.load_state_dict(torch.load("model_tinyImageNet/resnet_0.pth")) 

        # model = load_model(cls,iters, "extract_l_models").cuda()
        # models.append(globals()['student' + str(iters)])
    
        accu = adv_train(model, teacher, train_loader, test_loader,cls, i)
        accus.append(accu)
        
    # f = open(os.path.join(dir, "extract_model_adv_accus.txt"),mode='w')
    f = open(os.path.join(dir, "model_adv_accus.txt"),mode='w')
    f.write("acc:"+" \n".join(map(str,accus)))
    f.close()
    print(accus)