from model_load import load_model
from dataset import dataset, normalize
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import utils
from models import vgg, ResidualBlock, ResNet
import os
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
from torchvision import transforms
import torchvision
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage
import copy
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
import random
from torch.utils.data import DataLoader,Dataset, SubsetRandomSampler
from torchvision.datasets import ImageFolder
torch.cuda.set_device(3)
BATCH_SIZE=512
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
    
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler)
    
    random.seed(1)
    val_sampler = SubsetRandomSampler(indices[80000:90000])
    
    val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
    # print(len(val_sampler))
    
    
    return train_loader,val_loader, len(val_sampler)

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
a = max(-0.4914/0.2023, -0.4822/ 0.1994, -0.4465/0.2010) 
b = min((1-0.4914)/0.2023, (1-0.4822)/ 0.1994, (1-0.4465)/0.2010)
 
def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=100):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        # print("Using GPU")
        image = image.cuda()
        net = net.cuda()
        
    f_image = net.forward(Variable(image, requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.detach().cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
    # print(input_shape)
    loop_i = 0

    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            if x.grad is not None:
              x.grad.zero_()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1
    r_tot = (1+overshoot)*r_tot
    return r_tot, loop_i, label, k_i, pert_image
    
def show_save_img(img, path, name):
    #展示并保存图片
    # classes_label = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    classes_label = []
    # print(img.shape)
    mean = np.array([0.4914, 0.4822, 0.4465])[:, np.newaxis, np.newaxis]
    std = np.array([0.2023, 0.1994, 0.2010])[:, np.newaxis, np.newaxis]
    img = img.cpu().detach().numpy() * std + mean
    img = img.transpose(1, 2, 0).astype(float) #(H x W x C) 在 [0.0, 1.0]
    # print(img.shape)
    # plt.figure(figsize=(6,4))
    img = np.clip(img, a_min=0, a_max=1)
    plt.imshow(img,vmin=0, vmax=1)
    # plt.title(Target)
    plt.savefig(os.path.join(path, name), format="png") 
    plt.close()

def test_rf_acc(model, x, y, testloader):
    correct = 0
    num = 0  
    for i,(data,target) in enumerate(testloader):
        data, target = data.cuda(), target.cuda()
        batch_size = data.size()[0]
        
        perturbed_data1 = x.repeat(batch_size, 1, 1, 1).detach().cuda()
        y1 = y.repeat(batch_size)
        # print(perturbed_data.size(),data.size())
        output1 = model(perturbed_data1 + data)
        # output1 = output1.detach()
        pred = torch.argmax(output1,dim=-1)
        correct += torch.sum(y1==pred)
        num += y1.shape[-1]
        
    acc = float(correct/num) 
    return acc

def attack(rate,Target,epoch,link,convergence):
     
    # load model and dataset
   
    print(save_path)
    # model = torchvision.models.vgg16(pretrained=False)
    model = torchvision.models.resnet18(pretrained=False)
    in_feature = model.fc.in_features
    model.fc = torch.nn.Linear(in_feature, 200)
    model.load_state_dict(torch.load(save_path))
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    model.eval()

    # data = dataset('dataset_defend_1.h5', 'dataset_attack_1.h5', rate=rate, train=False)
    
    # trainloader = DataLoader(data, shuffle=True, batch_size=256)
    
    #cifar10
    # transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # testset = torchvision.datasets.CIFAR10(
    #     root='./data', train=False, download=True, transform=transform_test)
    # 加载cifar100
    # transform = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    # ])
    # testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    # 加载tinyimagenet
    # transform = transforms.Compose([
    # transforms.Resize((64, 64)),  # 调整图像大小
    # transforms.ToTensor(),         # 将图像转为Tensor
    # ])
    # data_dir = './data/tiny-imagenet-200'
    # testset = ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
    
    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=256, shuffle=False)
    
    # data = dataset('dataset_defend_1.h5', 'dataset_attack_1.h5', rate=rate, train=False)
    # dataloader = DataLoader(testset, shuffle=False, batch_size=1)
    dataloader,testloader, _ = dataset_tinyImageNet(rate)
    #中心点初始化为已有得分最高的图片
    value=0
    for i, (im,target) in enumerate(dataloader):

        im, target = im.cuda(), target.cuda()
        x = im
        original_rf=im.cpu().detach()
        if target==Target: 
            # 随机选取
            if  model(im).max(1, keepdim=True)[1].item() == target.item():
                x = im
                
                with torch.no_grad():
                    original_rf=im.cpu().detach()
                    print("选取第",i,"个数据作为初始中心点")
                    break
                
    #中心点初始化为0
    y = torch.tensor(Target)
    # print(x.shape,y.shape)torch.Size([1, 3, 32, 32]) torch.Size([])
    x, y = x.cuda(), y.cuda()
    # 通过模型前向传递数据
    output = model(x)
    print("预测标签：", output.max(1, keepdim=True)[1].item() , "原始标签：", y.item())
     
    #寻找rf
    optimizer = torch.optim.Adam( [x] , lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    Loss=[]
    scores = []
    Confidence_x=[] 
    r_adv = []
    # 测试初始rf精度    
    acc = []   
    best_score = 0      
    x.requires_grad = True
    # r_tot = 0
    for j in tqdm(range(epoch)):
          
        output = model(x)
        # 计算总损失
        loss_rf = F.nll_loss(output,  y.repeat(1))
        Loss.append(loss_rf)
        # 将所有现有的渐变归零
        optimizer.zero_grad()
        # 计算总的损失梯度
        loss_rf.backward(retain_graph=True)
        # 梯度下降
        optimizer.step() 
        with torch.no_grad():
            # #限制像素点范围
            x[0][0].clamp_(min=-0.4914/0.2023, max=(1-0.4914)/0.2023)
            x[0][1].clamp_(min=-0.4822/ 0.1994, max=(1-0.4822)/ 0.1994)
            x[0][2].clamp_(min=-0.4465/0.2010, max=(1-0.4465)/0.2010)
            # mean = np.array([0.4914, 0.4822, 0.4465])[:, np.newaxis, np.newaxis]#(3,1,1)
            # std = np.array([0.2023, 0.1994, 0.2010])[:, np.newaxis, np.newaxis]#(3,1,1)
            # x = x[0].cpu().detach().numpy() * std + mean
            # x = np.clip(x, a_min=0, a_max=1)
            # x = torch.tensor(x).view(1,3,32,32).to(torch.float32)
            #目标得分
            score = model(x)[0][Target].item()
            confidence_x = link(model(x))[0][Target].item()
            # 测试rf精度           
            acc_rf = test_rf_acc(model, x, y, testloader) 
            acc.append(acc_rf) 
         
        if j % 1 == 0:
            r_tot, loop_i, label, k_i, pert_image = deepfool(x,model)
            r_adv.append(np.linalg.norm(r_tot.flatten()))
        print("rf精度:",acc_rf,"原始标签:", y.item(),  "预测标签:",model(x).max(1, keepdim=True)[1].item(), "原始标签得分:", score, "置信度:",confidence_x,"最小扰动:",np.linalg.norm(r_tot.flatten())) 
        Confidence_x.append(confidence_x)
        scores.append(score)  
            
        # 收敛则提前终止
        if convergence:
            if scores[-1] > best_score:
                best_score = scores[-1]
                best_epoch = j
            
            if j - best_epoch == 10: 
                print('Stopping early at epoch = {}'.format(j))
                break 
               
    return acc, x, Loss, scores, Confidence_x, original_rf, j, r_adv

if __name__ == '__main__':
    epoch = 300
    convergence = True
    link=nn.Softmax(dim=-1)
    for j in range(1):
        j=0
        #加载模型
        # save_path = './model/resnet_{}.pth'.format(j)
        # save_path = './model_cifar100/resnet_{}.pth'.format(j)
        save_path = './model_tinyImageNet/resnet_{}.pth'.format(j)
        
        model_name=save_path.split("/")[-1].split(".")[0]
        # path = "./show_rf/{}_data-free_{}".format(model_name,epoch)
        # path = "./show_rf_cifar100/{}_data-free_{}".format(model_name,epoch)
        path = "./show_rf_tinyImageNet/{}_data-free_{}".format(model_name,epoch)
        if os.path.exists(path) == False:
            os.makedirs(path)
         
      #分别对每个类别都去找rf
        train_epoch = []
        for i in tqdm(range(0,10)):
            Target=i
            acc, rf, Loss, scores, Confidence_x, original_rf, stop_epoch,r_adv = attack(float(j/10),Target,epoch,link,convergence)
            train_epoch.append(stop_epoch)
            #保存rf
            torch.save(rf, os.path.join(path,"rf_"+str(Target)+".pth"))
            #保存original_rf
            torch.save(original_rf, os.path.join(path,"original_rf_"+str(Target)+".pth"))
            #rf图 
            show_save_img(rf[0],path,"rf_"+str(Target)+".png")
            #原图 
            show_save_img(original_rf[0],path,"original_rf_"+str(Target)+".png")
            
            #保存confidence
            fileObject = open(os.path.join(path,"rf_"+str(Target)+"_Confidence.txt"), 'w')
            for l in Confidence_x:  
                fileObject.write(str(l))  
                fileObject.write('\n') 
            fileObject.close()
            #confidence图 
            plt.plot(Confidence_x, linewidth=2) 
            plt.title("rf Confidence", fontsize=24) 
            plt.xlabel("Epoch", fontsize=14)
            plt.ylabel("Confidence", fontsize=14) 
            plt.savefig(os.path.join(path,"rf_"+str(Target)+"_Confidence.png"), bbox_inches='tight')
            plt.close()
            
            #保存acc
            fileObject = open(os.path.join(path,"rf_"+str(Target)+"_acc.txt"), 'w')
            for l in acc:  
                fileObject.write(str(l))  
                fileObject.write('\n') 
            fileObject.close()
            #acc图 
            plt.plot(acc, linewidth=2) 
            plt.title("rf robustness", fontsize=24) 
            plt.xlabel("Epoch", fontsize=14)
            plt.ylabel("rf acc", fontsize=14) 
            plt.savefig(os.path.join(path,"rf_"+str(Target)+"_acc.png"), bbox_inches='tight')
            plt.close()
            
            #保存scores
            fileObject = open(os.path.join(path,"rf_"+str(Target)+"_scores.txt"), 'w')
            for l in scores:  
                fileObject.write(str(l))  
                fileObject.write('\n') 
            fileObject.close()
            #scores图 
            plt.plot(scores, linewidth=2) 
            plt.title("rf Confidence", fontsize=24) 
            plt.xlabel("Epoch", fontsize=14)
            plt.ylabel("score", fontsize=14) 
            plt.savefig(os.path.join(path,"rf_"+str(Target)+"_scores.png"), bbox_inches='tight')
            plt.close()
            
            #保存r_adv
            fileObject = open(os.path.join(path,"rf_"+str(Target)+"_radv.txt"), 'w')
            for l in r_adv:  
                fileObject.write(str(l))  
                fileObject.write('\n') 
            fileObject.close()
            #r_adv图 
            plt.plot(r_adv, linewidth=2) 
            plt.title("r_adv", fontsize=24) 
            plt.xlabel("Epoch", fontsize=14)
            plt.ylabel("r_adv", fontsize=14) 
            plt.savefig(os.path.join(path,"rf_"+str(Target)+"_radv.png"), bbox_inches='tight')
            plt.close()
    
        #保存epoch
        fileObject = open(os.path.join(path,"rf_train_epoch.txt"), 'w')
        for l in train_epoch:  
            fileObject.write(str(l))  
            fileObject.write('\n') 
        fileObject.close()
        
        