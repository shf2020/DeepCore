from model_load import load_model
from dataset import dataset, normalize
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import utils
from models import vgg, ResidualBlock, ResNet
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
from torchvision import transforms
import torchvision
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage
import copy
from torch.autograd import Variable
# mean = torch.tensor([[[0.4914]*32]*32, [[0.4822]*32]*32, [[0.4465]*32]*32]).cuda()
# std = torch.tensor([[[0.2023]*32]*32, [[0.1994]*32]*32, [[0.2010]*32]*32]).cuda()
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
a = max(-0.4914/0.2023, -0.4822/ 0.1994, -0.4465/0.2010) 
b = min((1-0.4914)/0.2023, (1-0.4822)/ 0.1994, (1-0.4465)/0.2010)
       
def show_save_img(img, path, name):
    """展示并保存图片

    Args:
    img (_type_): 当前图片
    path (_type_): 图片保存路径
    name (_type_): 图片名称
    """ 
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    img = img.cpu().detach().numpy().transpose(1, 2, 0) #(C x H x W) 在 [0.0, 1.0]
    img = (img * std + mean) * 255
    # img = Image.fromarray(np.uint8(img))
    plt.figure(figsize=(6,4))
    plt.imshow(img.astype('uint8'))
    plt.title("label "+str(Target))
    plt.savefig(os.path.join(path, name), format="png", dpi=600) 
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

def attack(rate,Target):
     
    # load model and dataset
    
    # model = vgg()
    # model = ResNet(ResidualBlock, [2, 2, 2])
    print(save_path)
    # model = torchvision.models.vgg16(pretrained=False)
    model = torchvision.models.resnet18(pretrained=False)
    in_feature = model.fc.in_features
    model.fc = torch.nn.Linear(in_feature, 10)
    model.load_state_dict(torch.load(save_path))
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    data = dataset('dataset_defend_1.h5', 'dataset_attack_1.h5', rate=rate, train=False)
    
    trainloader = DataLoader(data, shuffle=True, batch_size=256)
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False)
    
    # data = dataset('dataset_defend_1.h5', 'dataset_attack_1.h5', rate=rate, train=False)
    dataloader = DataLoader(data, shuffle=False, batch_size=1)
    #中心点初始化为已有得分最高的图片
    value=0
    for i, (data,target) in enumerate(dataloader):

        data, target = data.cuda(), target.cuda()
        
        if target==Target:
            # 选取起始置信度最高的
            # if  model(data.repeat(1, 1, 1, 1)).max(1, keepdim=True)[1].item() == target.item():
            #     output_value=model(data).max().item()
            #     output_value>value
            #     # x=data.repeat(1, 1, 1, 1)
            #     x=data
            #     value = output_value
            
            # 随机选取
            if  model(data.repeat(1, 1, 1, 1)).max(1, keepdim=True)[1].item() == target.item():
                x=data
                print("选取第",i,"个数据作为初始中心点")
                break
    
    #中心点初始化为0
    # x = torch.tensor(np.zeros((1,3,32,32)),dtype=torch.float32)  
    # print(x)
    y = torch.tensor(Target)
    # print(x,x.shape,y)
    x, y = x.cuda(), y.cuda()
    
    # 通过模型前向传递数据
    output = model(x)
    # print(x)
    print("预测标签：", output.max(1, keepdim=True)[1].item() , "原始标签：", y.item())
    # 测试初始rf精度    
    acc = test_rf_acc(model, x, y, testloader) 
    
    # print(x.shape,y.shape)
    print("初始rf精度：",acc)  
    #寻找rf
    optimizer = torch.optim.Adam( [x] , lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # optimizer1 = torch.optim.Adam( [x] , lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    Loss=[]
    Confidence_x=[]
    done = False
    # if output.max(1, keepdim=True)[1].item() == y.item(): 
    
    epsilon1=0.001
    epsilon2=0.01
    epoch1 = 10
    epoch2 = 1000    
    
    # perturbation = torch.zeros_like(x)   
    # perturbation.requires_grad = True
    # adv = x+perturbation
    # for i in tqdm(range( epoch1)):
    #     rf=x#.detach()
    #     y2 = y.repeat(1)
    #     # print(x.shape,y2.shape)  
    #     perturbation.requires_grad = True 
    #     output_x = model(adv)  
    #     loss_x = F.nll_loss(output_x, y2)
    #     # print(loss_x)
    #     optimizer.zero_grad()                    
    #     loss_x.backward(retain_graph=True)                   
    #     # optimizer.step()
    #     data_grad = perturbation.grad.data
    #     # 收集数据梯度的元素符号
    #     sign_data_grad = data_grad.sign()
    #     # print(data_grad,sign_data_grad)
    #     # 通过调整输入图像的每个像素来创建扰动
    #     adv = (adv + epsilon1*sign_data_grad)
    #     #限制像素点范围
    #     # print(a,b)
    #     adv[0][0].clamp_(min=-0.4914/0.2023, max=(1-0.4914)/0.2023)
    #     adv[0][1].clamp_(min=-0.4822/ 0.1994, max=(1-0.4822)/ 0.1994)
    #     adv[0][2].clamp_(min=-0.4465/0.2010, max=(1-0.4465)/0.2010)
    #     # adv.clamp_(min=a, max=b)
    #     #mean = (0.4914, 0.4822, 0.4465)std = (0.2023, 0.1994, 0.2010)
    #     # adv = torch.clamp(adv, min(-0.4914/0.2023, -0.4822/ 0.1994, -0.4465/0.2010), max(1-0.4914/0.2023, 1-0.4822/ 0.1994, 1-0.4465/0.2010))  
    #     # print(adv)
    #     acc = test_rf_acc(model, adv, y, testloader)
    #     print( "对抗扰动后rf精度：",acc, "对抗扰动后标签:",model(adv).max(1, keepdim=True)[1].item(), "原始标签得分:", model(adv)[0][Target].item(), "对抗扰动后最高得分:",model(adv).max().item())
 
    #     pert = (adv-rf).data
    #     r[Target] = pert.pow(2).sum().sqrt().item()
    #     print("当前核半径:", pert.pow(2).sum().sqrt().item())
        
    #     if acc<=0.95:
          
    x.requires_grad = True
    # adv_rf = x + pert

    for j in tqdm(range(epoch2)):
        
        
        y3 = y.repeat(1)
        output = model(adv_rf)
        # 计算总损失
        loss_rf = F.nll_loss(output, y3)
        # print(loss)
        # 将所有现有的渐变归零
        optimizer.zero_grad()
        # 计算总的损失梯度
        loss_rf.backward(retain_graph=True)
        # 梯度下降
        # optimizer.step() 
        data_grad = x.grad.data
        # 收集数据梯度的元素符号
        sign_data_grad = data_grad.sign()
    
        adv_rf = adv_rf - epsilon2*sign_data_grad
        
        #限制像素点范围
        adv_rf[0][0].clamp_(min=-0.4914/0.2023, max=(1-0.4914)/0.2023)
        adv_rf[0][1].clamp_(min=-0.4822/ 0.1994, max=(1-0.4822)/ 0.1994)
        adv_rf[0][2].clamp_(min=-0.4465/0.2010, max=(1-0.4465)/0.2010)
        # adv_rf.clamp_(min=a, max=b)
        #目标得分
        confidence_x = model(adv_rf)[0][Target].item()
        print(test_rf_acc(model, adv_rf, y, testloader))    
        # 测试rf精度           
        acc_rf = test_rf_acc(model, adv_rf, y, testloader)
        
        print("rf精度:",acc_rf, "标签:",model(adv_rf).max(1, keepdim=True)[1].item(), "原始标签得分:", confidence_x, "最高得分:",model(adv_rf).max().item()) 
        Confidence_x.append(confidence_x)
        # print(test_rf_acc(model, adv_rf, y, testloader))
        # print(adv_rf)
        # if acc==1:
        # if j==epoch2-1:
            
        #     x = adv_rf.detach()  #.detach()
            
        #     print("break--------------------------------")
                    # break 
                
                # if j==epoch2-1:
                #     done = True
            
        # print("核半径集合:",r)
        # if done == True or i == epoch1-1:
            
        #     break  
    
        # if done:
        #     break             

           
    return acc, rf, Loss, Confidence_x

if __name__ == '__main__':
    # iters = 1000
    for j in range(1):
        j=0
        save_path = './model/resnet_'+str(j)+'.pth'
        # save_path = './model/vgg_0.pth'
        model_name=save_path.split("/")[-1].split(".")[0]
        path = "./show_rf/"+str(model_name)+"_data-free_v1000"
        if os.path.exists(path) == False:
            os.makedirs(path)
        
        r=[0]*10 #定义初始核半径
        
        #分别对每个类别都去找rf
        for i in range(0,10):
            Target=i
            final_acc, rf, Loss, Confidence_x = attack(float(j/10),Target)
            print("final_acc:", final_acc)

            #保存rf
            torch.save(rf, os.path.join(path,"rf_"+str(Target)+".pth"))
            #rf图 
            show_save_img(rf[0],path,"rf_"+str(Target)+".png")
            
            #保存confidence
            fileObject = open(os.path.join(path,"rf_"+str(Target)+"_Confidence.txt"), 'w')
            for l in Confidence_x:  
                fileObject.write(str(l))  
                fileObject.write('\n') 
            fileObject.close()
            #confidence图 
            plt.plot(Confidence_x, linewidth=2) 
            plt.title("rf Confidence", fontsize=24) 
            plt.xlabel("iters", fontsize=14)
            plt.ylabel("Confidence", fontsize=14) 
            plt.savefig(os.path.join(path,"rf_"+str(Target)+"_Confidence.png"), bbox_inches='tight')
            plt.close()
                
        #保存核半径
        fileObject = open(os.path.join(path,"rf"+"_nuclear_radius.txt"), 'w')
        for l in r:  
            fileObject.write(str(l))  
            fileObject.write('\n') 
        fileObject.close()