import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import time
import h5py
from dataset import  dataset1
from models import vgg, ResidualBlock, ResNet


os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
BATCH_SIZE = 128


def load_model(cls,num,mode):
    if mode == 'teacher':
        # teacher = vgg()
        model = torchvision.models.vgg16_bn(pretrained=False)
        in_feature = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(os.path.join("model", "vgg_model.pth")))


    elif mode == 'extract_l_models':
        if num<5:
            model = torchvision.models.vgg16_bn(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)

        elif 5<=num<10:
            model = torchvision.models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 10)
        
        elif 15>num>=10:
            model = torchvision.models.densenet121(pretrained=False)
            in_feature = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_feature, 10)

        elif 20>num>=15:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)

        model.load_state_dict(torch.load(os.path.join("extract_l_models", "extract_model_l_"+ str(cls) + str(num) + ".pth")))


    elif mode == "extract_adv_models":

        if 5<=num<10:
            model = torchvision.models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 10)
        elif num<5:
            model = torchvision.models.vgg13(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)

        elif 15>num>=10:
            model = torchvision.models.densenet121(pretrained=False)
            in_feature = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_feature, 10)

        elif 20>num>=15:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)

        model.load_state_dict(torch.load(os.path.join("extract_adv_models", "extract_model_adv_"+ str(cls)  + str(num) + ".pth")))



    elif mode == "original_model":
        
        save_path= os.path.join("model", str(cls) + "_" + str(num) + ".pth")

        if num<=10:
            model = vgg()
            model.load_state_dict(torch.load(save_path))
            model.eval()
            
        else:
            model = ResNet(ResidualBlock, [2, 2, 2])
            model.load_state_dict(torch.load(os.path.join("model", str(cls) + "_" + str(num-11) + ".pth")))
            model.eval()
     

    elif mode == "adv_models":
        
        if num<=10:
            model = vgg()
            model.load_state_dict(torch.load( os.path.join("adv_models", str(cls) + "_" + str(num) + ".pth")))
            model.eval()
            
        else:
            model = ResNet(ResidualBlock, [2, 2, 2])
            model.load_state_dict(torch.load(os.path.join("adv_models", str(cls) + "_" + str(num) + ".pth")))
            model.eval()


    
    model = torch.nn.DataParallel(model)
    model.cuda()
    return model


