import torch
import torchvision
from dataset import dataset
from models import vgg, ResidualBlock, ResNet
from model_load import load_model
from torch.utils.data import DataLoader,Dataset,random_split, SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import os
import numpy as np
from tqdm import tqdm
import random
import torch.optim as optim
print(torch.cuda.is_available())
print(torch.cuda.device_count())
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
# 限制每个进程使用的GPU内存比例
# torch.cuda.set_per_process_memory_fraction(0.7)  # 例如，限制每个进程使用50%的GPU内存
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
torch.cuda.set_device(3)
import torch.nn as nn
from tqdm import tqdm
BATCH_SIZE=512

def dataset_cifar100(rate):
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
       
    random.seed(5)
    List1 = random.sample(range(0,20000),int(20000*rate))
    random.seed(1)
    List2 = random.sample(range(0,20000),int(20000-20000*rate))

    a = images_data[0][0:20000]
    b = images_data[1][0:20000]
    print(a.size())
    A = []
    B = []
    if len(List1) != 0:
        for i in List1:
            A.append(a[i].numpy())
    if len(List2) != 0:       
        for i in List2:
            B.append(b[i].numpy())
    print("训练集从一个数据集随机选择图片个数:", len(A), "从另一个互不相交的数据集随机选择图片个数:", len(B), "所以与受害者模型训练数据集重合占比为", rate )    
    if len(A) == 0:
        images = np.array(B)
    elif len(B) == 0:
        images = np.array(A)
    else:
        images = np.concatenate((A, B),axis=0)
    
    x = labels_data[0][0:20000]
    y = labels_data[1][0:20000]
    # print(x)
    X = []
    Y = []
    if len(List1) != 0:
        for i in List1:
            X.append(x[i].numpy())
    if len(List2) != 0:
        for i in List2:
            Y.append(y[i].numpy())
    # print(len(X),len(Y)) 
    if len(X) == 0:
        labels =  np.array(Y)
    elif len(Y) == 0:
        labels =  np.array(X)
    else:
        labels = np.concatenate((X, Y),axis=0) 
    return images,labels

def dataset_tinyImageNet(rate):
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
    # print(indices)
    random.seed(1)
    List1 = random.sample(indices[:40000],int(40000*rate))
    List2 = random.sample(indices[40000:80000],int(40000-40000*rate))
    
    List = List1 + List2
    print("训练集从一个数据集随机选择图片个数:", len(List1), "从另一个互不相交的数据集随机选择图片个数:", len(List2), "所以与受害者模型训练数据集重合占比为", rate )
    train_sampler = SubsetRandomSampler(List)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    
    random.seed(1)
    val_sampler = SubsetRandomSampler(indices[80000:90000])
    
    val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
    # print(len(val_sampler))
    
    
    return train_loader,val_loader, len(val_sampler)

def train_model(rate,model_name):
    
    # train_data = dataset('dataset_defend_1.h5', 'dataset_attack_1.h5', rate, train=False)
    # images,labels = dataset_cifar100(rate)

    # train_data = torch.utils.data.TensorDataset(torch.tensor(images), torch.tensor(labels))
    # train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
   
    # transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # testset = torchvision.datasets.CIFAR10(
    #     root='./data', train=False, download=True, transform=transform_test)
    
    # 加载cifar100
    # transform = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    #加载tinyimagenet
    train_loader,test_loader, val_num = dataset_tinyImageNet(rate)
    
    print(val_num)
    # transform = transforms.Compose([
    # transforms.Resize((64, 64)),  # 调整图像大小
    # transforms.ToTensor(),         # 将图像转为Tensor
    # ])
    # data_dir = './data/tiny-imagenet-200'
    # testset = ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    # test_loader = torch.utils.data.DataLoader(
    #     testset, batch_size=BATCH_SIZE, shuffle=False)
    # val_num = len(testset)
    
    # model_name = "vgg_test"
    # net = vgg(model_name="vgg16", num_classes=10, init_weights=True)
    # model_name = "vgg_test"
    # model_name = "resnet50"
    # net = ResNet(ResidualBlock, [2, 2, 2]).cuda()
    # if model_name == "vgg":
    #     net = torchvision.models.vgg16(pretrained=False)
    #     in_feature = net.classifier[-1].in_features
    #     net.classifier[-1] = torch.nn.Linear(in_feature, 10)
    
    if model_name == "resnet":
        net = torchvision.models.resnet18(pretrained=False)
        in_feature = net.fc.in_features
        net.fc = torch.nn.Linear(in_feature, 200)
        
    if model_name == "densenet":
        net = torchvision.models.densenet121(pretrained=False)
        in_feature = net.classifier.in_features
        net.classifier = torch.nn.Linear(in_feature, 200)
        
    net.cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 100
    best_acc = 0.0
    save_path = dir + '/{}_{}.pth'.format(model_name,int(rate*10))
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar,0):
            images, labels = data
            # print(images.shape)
            optimizer.zero_grad()
            outputs = net(images.cuda())
            loss = loss_function(outputs, labels.cuda())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            #val_bar = tqdm(validate_loader, colour = 'green')
            for val_data in test_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.cuda())
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.cuda()).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')
    
if __name__ == '__main__':
    # dir='model_cifar100'
    dir='model_tinyImageNet'
    if os.path.exists(dir) == 0:
        os.mkdir(dir)
        print("Making directory!")
    for j in range(2,3):
        # if j==0:
        #     model_name = "vgg"
        if j==1:
            model_name = "resnet"
        if j==2:
            model_name = "densenet"
        # if j==3:
        #     model_name = "mobilenet"
        for i in range(11):
            train_model(float(i/10),model_name)

    # model_name = "vgg"
    # # train_model(float(0/10),model_name)
    # for i in range(11):
    #     train_model(float(i/10),model_name)
    
    