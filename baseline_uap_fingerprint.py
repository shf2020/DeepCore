import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


# 构造通用对抗扰动
def generate_uap(model, data_loader, epsilon=0.01, num_epochs=2, sample_num=100, target_label=torch.tensor(1)):
    '''使用 Fast Gradient Sign Method (FGSM) 进行对抗攻击，对测试集每个样本进行多轮对抗生成uap
    
    model：构造uap指纹的目标模型
    data_loader：构造uap指纹的测试集
    epsilon：控制加噪幅度
    num_epochs：遍历测试集轮次
    sample_num：控制构造UAP的测试样本数量
    target_label：通用对抗扰动的目标域标签
    
    '''
    criterion = nn.CrossEntropyLoss()
    perturbation = torch.zeros_like(next(iter(data_loader))[0])
    perturbation.requires_grad = True
    optimizer = optim.SGD([perturbation], lr=epsilon)

    

    for epoch in range(num_epochs):
        i=0
        for inputs, labels in data_loader:
            outputs = model(inputs.cuda()+perturbation.cuda())
            loss = criterion(outputs, target_label.repeat(1).cuda())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            i+=1
            if i >sample_num:
                break
            
        
    return perturbation

if __name__ == "__main__":
    #存储指纹文件夹
    dir = './fingerprint'
    #加载自定义模型,例如resnet
    save_path = 'model/resnet_0.pth' 
    model = torchvision.models.resnet18(pretrained=False)
    in_feature = model.fc.in_features
    model.fc = torch.nn.Linear(in_feature, 10)
    model.load_state_dict(torch.load(save_path))
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    
    # 加载数据集去构造uap，例如CIFAR-10数据集
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
        (0.2023, 0.1994, 0.2010))])

    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 构造uap模型指纹
    uap = generate_uap(model, test_loader)    
    torch.save(uap, os.path.join(dir,"uap_fingerprint"+".pth"))
    print('模型指纹生成完毕！')
