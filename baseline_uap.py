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
# dir='UAPs_demo'
# dir='UAPs_cifar100'
dir='UAPs_tinyImageNet'
import os
if os.path.exists(dir) == 0:
    os.mkdir(dir)
    print("Making directory!")
# 加载CIFAR-10数据集
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
    (0.2023, 0.1994, 0.2010))])

test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
# #加载cifar100
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
# ])
# test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 定义通用对抗扰动生成函数
def generate_uap(model, data_loader, epsilon=0.01, num_epochs=2,target_label=torch.tensor(1)):
    # 使用 Fast Gradient Sign Method (FGSM) 进行对抗攻击
    criterion = nn.CrossEntropyLoss()
    perturbation = torch.zeros_like(next(iter(data_loader))[0])
    perturbation.requires_grad = True
    optimizer = optim.SGD([perturbation], lr=epsilon)
    # print(perturbation.size(),target_label.repeat(1).size())
    

    for epoch in range(num_epochs):
        i=0
        for inputs, labels in data_loader:
            # inputs.requires_grad = True
            model = model.to('cuda:0')
            outputs = model(inputs.to('cuda:0')+perturbation.to('cuda:0'))
            loss = criterion(outputs, target_label.repeat(1).cuda())
            model.zero_grad()
            loss.backward()
            # perturbation = epsilon * torch.sign(inputs.grad.data)
            optimizer.step()
            # print(perturbation)
            i+=1
            if i >100:
                break
            
            

    return perturbation

#加载模型
class FeatureHook():
    
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        self.output = output

    def close(self):
        self.hook.remove()

def loadmodels():
    
    models = []
    
    for i in range(11):
        save_path = 'model/resnet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        
    for i in range(11):
        save_path = 'model/densenet_{}.pth'.format(i)  
        # print(save_path) 
        model = torchvision.models.densenet121(pretrained=False)
        in_feature = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        
    current_address = "pruning_models1"
    file_list = os.listdir(current_address)
    i=0
    for file_address in file_list:
        save_path = os.path.join(current_address, file_address)
        # print(save_path)
        i+=1
        # print(save_path) 
        model = torch.load(save_path)
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        if i==10:
            break

    
    for i in range(10):
        save_path = 'finetune_models/finetune_resnet_last_{}.pth'.format(i+10)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        
    for i in range(10):
        save_path = 'finetune_models/finetune_resnet_all_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
      
    for i in range(10):
        save_path = 'adv_models/model_adv_resnet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
         
    for i in range(10):
        save_path = 'extract_l_models/extract_model_l_resnet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
    
    for i in range(10):
        save_path = 'extract_l_models/extract_model_l_dense_{}.pth'.format(i+10)  
        #print(save_path) 
        model = torchvision.models.densenet121(pretrained=False)
        in_feature = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        
    for i in range(10):
        save_path = 'extract_p_models/extract_model_p_resnet{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
    
    for i in range(10):
        save_path = 'extract_p_models/extract_model_p_dense{}.pth'.format(i+10)  
        #print(save_path) 
        model = torchvision.models.densenet121(pretrained=False)
        in_feature = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        
    return models

def loadmodels_cifar100():
    
    models = []
    
    for i in range(11):
        save_path = 'model_cifar100/resnet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 100)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        
    for i in range(11):
        save_path = 'model_cifar100/densenet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.densenet121(pretrained=False)
        in_feature = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feature, 100)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        
    current_address = "pruning_models_cifar100"
    file_list = os.listdir(current_address)
    i=0
    for file_address in file_list:
        save_path = os.path.join(current_address, file_address)
        # #print(save_path)
        i+=1
        # print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 100)
        model = torch.load(save_path)
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        if i==10:
            break

    
    for i in range(10):
        save_path = 'finetune_models_cifar100/finetune_resnet_last_{}.pth'.format(i+10)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 100)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        
    for i in range(10):
        save_path = 'finetune_models_cifar100/finetune_resnet_all_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 100)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
      
    for i in range(10):
        save_path = 'adv_models_cifar100/model_adv_resnet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 100)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
         
    for i in range(10):
        save_path = 'extract_l_models_cifar100/extract_model_l_resnet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 100)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
    
    for i in range(10):
        save_path = 'extract_l_models_cifar100/extract_model_l_dense_{}.pth'.format(i+10)  
        #print(save_path) 
        model = torchvision.models.densenet121(pretrained=False)
        in_feature = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feature, 100)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        
    for i in range(10):
        save_path = 'extract_p_models_cifar100/extract_model_p_resnet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 100)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
    
    for i in range(10):
        save_path = 'extract_p_models_cifar100/extract_model_p_dense_{}.pth'.format(i+10)  
        #print(save_path) 
        model = torchvision.models.densenet121(pretrained=False)
        in_feature = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feature, 100)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        
    return models

def loadmodels_tinyImageNet():
    
    models = []
    
    for i in range(11):
        save_path = 'model_tinyImageNet/resnet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 200)
        model.load_state_dict(torch.load(save_path, map_location='cuda:1'))
        # model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        
    for i in range(11):
        save_path = 'model_tinyImageNet/densenet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.densenet121(pretrained=False)
        in_feature = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feature, 200)
        model.load_state_dict(torch.load(save_path, map_location='cuda:1'))
        # model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        
    current_address = "pruning_models_tinyImageNet"
    file_list = os.listdir(current_address)
    i=0
    for file_address in file_list:
        save_path = os.path.join(current_address, file_address)
        # #print(save_path)
        i+=1
        # print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 200)
        model = torch.load(save_path, map_location='cuda:1')
        # model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        if i==10:
            break

    
    for i in range(10):
        save_path = 'finetune_models_tinyImageNet/finetune_resnet_last_{}.pth'.format(i+10)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 200)
        model.load_state_dict(torch.load(save_path, map_location='cuda:1'))
        # model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        
    for i in range(10):
        save_path = 'finetune_models_tinyImageNet/finetune_resnet_all_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 200)
        model.load_state_dict(torch.load(save_path, map_location='cuda:1'))
        # model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
      
    for i in range(10):
        save_path = 'adv_models_tinyImageNet/model_adv_resnet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 200)
        model.load_state_dict(torch.load(save_path, map_location='cuda:1'))
        # model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
         
    for i in range(10):
        save_path = 'extract_l_models_tinyImageNet/extract_model_l_resnet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 200)
        model.load_state_dict(torch.load(save_path, map_location='cuda:1'))
        # model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
    
    for i in range(10):
        save_path = 'extract_l_models_tinyImageNet/extract_model_l_dense_{}.pth'.format(i+10)  
        #print(save_path) 
        model = torchvision.models.densenet121(pretrained=False)
        in_feature = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feature, 200)
        model.load_state_dict(torch.load(save_path, map_location='cuda:1'))
        # model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        
    for i in range(10):
        save_path = 'extract_p_models_tinyImageNet/extract_model_p_resnet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 200)
        model.load_state_dict(torch.load(save_path, map_location='cuda:1'))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
    
    for i in range(10):
        save_path = 'extract_p_models_tinyImageNet/extract_model_p_dense_{}.pth'.format(i+10)  
        #print(save_path) 
        model = torchvision.models.densenet121(pretrained=False)
        in_feature = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feature, 200)
        model.load_state_dict(torch.load(save_path, map_location='cuda:1'))
        # model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        
    return models



# models=loadmodels()
# models=loadmodels_cifar100()
models = loadmodels_tinyImageNet()
print( "len(models):", len(models))


# 生成通用对抗扰动
from tqdm import tqdm
i=0
for model in tqdm(models):
    i+=1
    if i>78:
        uap = generate_uap(model, test_loader)  
        torch.save(uap, os.path.join(dir,"uap_"+str(i-1)+".pth"))


