from dataset import dataset1,dataset
from torch.utils.data import DataLoader,Dataset
import torch
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from model_load import load_model
from sklearn.metrics import roc_curve,auc
from models import vgg, ResidualBlock, ResNet
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

uap = []
for i in range(102):
    # path = 'UAPs'
    # save_path = "./UAPs/uap_" + str(i) +".pth"
    # path = 'UAPs_cifar100'
    # save_path = './UAPs_cifar100/uap_' + str(i) +".pth"
    path = 'UAPs_tinyImageNet'
    save_path = './UAPs_tinyImageNet/uap_' + str(i) +".pth"
    x = torch.load(save_path)
    uap.append(x)
cos_sim = []   
for i in range(1,102):
    cos_sim.append(F.cosine_similarity(uap[0].view(-1), uap[i].view(-1), dim=0))
    
print(cos_sim) 
diff = cos_sim
list1 = diff[:10]
list2 = diff[10:21]
list3 = diff[21:31]
list4 = diff[31:41]
list5 = diff[41:51]
list6 = diff[51:61]
list7 = diff[61:71]
list8 = diff[71:81]
list9 = diff[81:91]
list10 = diff[91:101]
plt.figure( figsize=(10, 6))
plt.xlabel('Models',fontsize=20)  # x轴标题
plt.ylabel('Cosine Similarity',fontsize=20)  # y轴标题
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.plot(list1, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
plt.plot(list2, marker='o', markersize=3) 
plt.plot(list3, marker='o', markersize=3) 
plt.plot(list4, marker='o', markersize=3) 
plt.plot(list5, marker='o', markersize=3) 
plt.plot(list6, marker='o', markersize=3) 
plt.plot(list7, marker='o', markersize=3) 
plt.plot(list8, marker='o', markersize=3) 
plt.plot(list9, marker='o', markersize=3) 
plt.plot(list10, marker='o', markersize=3) 
plt.legend(['HM_SA', 'HM_DA', 'PM_P', 'PM_FL','PM_FA','PM_Adv','EM_SA_L','EM_DA_L','EM_SA_Pr','EM_DA_Pr'],fontsize=20)  # 设置折线名称   

plt.savefig( path + "/uap.png", bbox_inches='tight')

plt.close()
rate = []    
for d in np.linspace(0, 1, 50):
    d1 = 0
    d2 = 0
    for i in diff[:21]:
        d1+=i
    for i in diff[21:]:
        d2+=i
    h=0
    for i in diff[:21]:
        if i< d:
            h+=1
    p=0
    for i in diff[21:]:
        if i>= d:
            p+=1
    rate.append(float(h/21)+ float(p/len(diff[21:])))
    print("FIR:",1-float(h/21))  
    print("MIR:",1-float(p/len(diff[21:]))) 
    
d = rate.index(max(rate))/50
d1 = 0
d2 = 0
for i in diff[:21]:
    d1+=i
for i in diff[21:]:
    d2+=i

h=0
for i in diff[:21]:
    if i< d:
        h+=1
p=0
for i in diff[21:]:
    if i>= d:
        p+=1

print('最佳阈值:',d)
print("误检率：",1-float(h/21))  
print("漏检率：",1-float(p/len(diff[21:]))) 
        