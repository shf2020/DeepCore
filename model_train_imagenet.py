import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import random
import torch
import os
import requests
import tarfile
import random
import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

data_dir = './data/tiny-imagenet-200'
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小
    transforms.ToTensor(),         # 将图像转为Tensor
    ])

train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=transform)

# 选择前十个类别的数据和对应标签
selected_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 假设这是您要选择的类别的标签

selected_data = [(image, label) for image, label in train_dataset if label in selected_classes]

# 创建新的数据集
from torch.utils.data import Dataset

class SubsetDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        return image, label

new_dataset = SubsetDataset(selected_data)

from torch.utils.data import DataLoader, random_split

# 定义要分配给训练集和测试集的比例
train_ratio = 0.8
test_ratio = 1 - train_ratio

# 计算分配的样本数量
train_size = int(train_ratio * len(new_dataset))
test_size = len(new_dataset) - train_size

# 划分数据集
train_dataset, test_dataset = random_split(new_dataset, [train_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(train_size,test_size) 

# 使用VGG
vgg_model = models.vgg16(pretrained=True)
vgg_model.classifier[6] = nn.Linear(4096, len(selected_classes))  # 修改最后一层全连接层

# 使用GoogLeNet
googlenet_model = models.googlenet(pretrained=True)
googlenet_model.fc = nn.Linear(1024, len(selected_classes))  # 修改最后一层全连接层

# 训练模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg_model.parameters(), lr=0.001, momentum=0.9)

from tqdm import tqdm
#训练模型
for epoch in range(5): # 举例，实际根据需要调整
    for inputs, labels in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = vgg_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

#测试模型精度
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = vgg_model(inputs)
        _, predicted = torch.max(outputs, 1)
        