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

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3' #'0,1,2,3'
BATCH_SIZE = 100
torch.cuda.set_device(1)

def calculate_auc(list_a, list_b):
    l1,l2 = len(list_a),len(list_b)
    y_true,y_score = [],[]
    for i in range(l1):
        y_true.append(0)
    for i in range(l2):
        y_true.append(1)
    y_score.extend(list_a)
    y_score.extend(list_b)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return auc(fpr, tpr)

class FeatureHook():

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.output = output
    def close(self):
        self.hook.remove()

def correlation(m,n):
    m = F.normalize(m,dim=-1)
    n = F.normalize(n,dim=-1).transpose(0,1)
    cose = torch.mm(m,n)
    matrix = 1-cose
    matrix = matrix/2
    return matrix

def pairwise_euclid_distance(A):
    sqr_norm_A = torch.unsqueeze(torch.sum(torch.pow(A, 2),dim=1),dim=0)
    sqr_norm_B = torch.unsqueeze(torch.sum(torch.pow(A, 2), dim=1), dim=1)
    inner_prod = torch.matmul(A, A.transpose(0,1))
    tile1 = torch.reshape(sqr_norm_A,[A.shape[0],1])
    tile2 = torch.reshape(sqr_norm_B,[1,A.shape[0]])
    return tile1+tile2 - 2*inner_prod

def correlation_dist(A):
    A = F.normalize(A,dim=-1)
    cor = pairwise_euclid_distance(A)
    cor = torch.exp(-cor)

    return cor

def cal_cor1(model,dataloader):
    model.eval()
    model = model.cuda()
    outputs = []
    for i, (x,y) in enumerate(dataloader):
        x = x.cuda()
        output = model(x)
        # print(x.shape)
        outputs.append(output.cuda().detach())
        if len(outputs)==10:
            break
    # print(len(outputs))
    output = torch.cat(outputs,dim=0)
    
    # print("-"*50)
    # print(output.shape)
    # print("-"*50)
    
    cor_mat = correlation(output,output)

    # cor_mat = correlation_dist(output)

    model = model.cuda()
    return cor_mat, output

def cal_cor(model,dataloader):
    # model.eval()
    model = model.to('cuda:0')
    outputs = []
    for x in dataloader:
        x = x.to('cuda:0')
        # print(model,x)
        output = model(x)
        # print(x.shape)
        # print(output)
        outputs.append(output.cuda().detach())
    # print(len(outputs))
    output = torch.cat(outputs,dim=0)
    
    # print("-"*50)
    # print(output.shape)
    # print("-"*50)
    
    cor_mat = correlation(output,output)

    # cor_mat = correlation_dist(output)

    model = model.cuda()
    return cor_mat, output


def cal_correlation(models):
    
    print("{}个模型加载完毕!".format(len(models)))
    cor_mats = []
    outputs = []

    # name = "SAC-w"
    # train_data = dataset('dataset_defend_1.h5', 'dataset_attack_1.h5', rate=0, train=False)
    # print(train_data.images.shape,train_data.labels.shape,train_data)
    # train_loader = DataLoader(train_data, shuffle=False, batch_size=1)
    # i=0
    # rf=[]
    # for j,(x,y) in enumerate(train_loader):
    #     x = x.cuda()
    #     y = y.cuda()
    #     output0 = models[0](x)
    #     output1 = models[31](x)
    #     output2 = models[41](x)
    #     output3 = models[51](x)
    #     output4 = models[61](x)
    #     output5 = models[71](x)
    #     output6 = models[81](x)
    #     output7 = models[91](x)
    #     output8 = models[101](x)
    #     if y != np.argmax(output0.cpu().detach().numpy()) and y != np.argmax(output1.cpu().detach().numpy()) and y != np.argmax(output2.cpu().detach().numpy()) and y != np.argmax(output3.cpu().detach().numpy()) and y != np.argmax(output4.cpu().detach().numpy()) and y != np.argmax(output5.cpu().detach().numpy()) and y != np.argmax(output6.cpu().detach().numpy()) and y != np.argmax(output7.cpu().detach().numpy()) and y != np.argmax(output8.cpu().detach().numpy()):
    #         i+=1
    #         rf.append(x)
            
    #         print(y,np.argmax(output0.cpu().detach().numpy()),np.argmax(output4.cpu().detach().numpy()))
    #         if i==10:
    #             break 
    # m=0
    # for i in range(len(models)):
    #     model = models[i]
    #     cor_mat, output = cal_cor(model, rf)
    #     cor_mats.append(cor_mat)
    #     outputs.append(output)
    # print("len(cor_mats):",len(cor_mats), "cor_mat.shape:",cor_mat.shape,"len(outputs):",len(outputs),"output.shape:",output.shape)
    # diff = torch.zeros(len(models))
    # diff_outputs = []

    # name = "SAC-m"
    # data = dataset1('cut_mix_final.h5', train=False)
    # rf = DataLoader(data, shuffle=True, batch_size=BATCH_SIZE)
    # print("rf个数:", len(data.images))
    # m=1
    # for i in range(len(models)):
    #     model = models[i]
    #     cor_mat, output = cal_cor1(model, rf)
    #     cor_mats.append(cor_mat)
    #     outputs.append(output)
    # print("len(cor_mats):",len(cor_mats), "cor_mat.shape:",cor_mat.shape,"len(outputs):",len(outputs),"output.shape:",output.shape)
    # diff = torch.zeros(len(models))
    # diff_outputs = []

    #决策空间中心点
    name = 'deepcore'
    rf=[]
    # combination = [0,2,3]
    combination = range(10)
    for i in combination:
        # save_path = "./show_rf/resnet_0_final_epoch100/rf_" + str(i) +" .pth"
        # save_path = './show_rf_cifar100/resnet_0_data-free_300/rf_' + str(i) +".pth"
        save_path = dir + '/resnet_0_data-free_300/rf_' + str(i) +".pth"
        x = torch.load(save_path, map_location='cuda:1')
        # print(x)
        rf.append(x)
    print("rf个数:", len(rf))
    m=2
    for i in range(len(models)):
        # print(models[i])
        model = models[i]
        # print(model, rf[0])
        cor_mat, output = cal_cor(model, rf) 
        cor_mats.append(cor_mat)
        outputs.append(output)
    print("len(cor_mats):",len(cor_mats), "cor_mat.shape:",cor_mat.shape,"len(outputs):",len(outputs),"output.shape:",output.shape)
    diff = torch.zeros(len(models))
    diff_outputs = [] 
    
    # # 消融实验
    # combination = range(10)
    # transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    # train_data = dataset('dataset_defend_1.h5', 'dataset_attack_1.h5', rate=0, train=False)
    # print(train_data.images.shape,train_data.labels.shape,train_data)
    # train_loader = DataLoader(train_data, shuffle=False, batch_size=1)
    # i=0
    # rf=[]
    # for label in range(10): 
    #     for j,(x,y) in enumerate(train_loader):
    #         x = x.cuda()
    #         y = y.cuda()
    #         output = models[0](x)
            
    #         if y == np.argmax(output.cpu().detach().numpy()) and y == label:
    #             rf.append(x)            
    #             print(y,np.argmax(output.cpu().detach().numpy()), label)
    #             break 
    # m=3
    # for i in range(len(models)):
    #     model = models[i]
    #     cor_mat, output = cal_cor(model, rf)
    #     cor_mats.append(cor_mat)
    #     outputs.append(output)
    # print("len(cor_mats):",len(cor_mats), "cor_mat.shape:",cor_mat.shape,"len(outputs):",len(outputs),"output.shape:",output.shape)
    # diff = torch.zeros(len(models))
    # diff_outputs = []
    


    for i in range(len(models) - 1):
        iter = i + 1
        diff[i] = torch.sum(torch.abs(cor_mats[iter] - cor_mats[0])) / (cor_mat.shape[0] * cor_mat.shape[1])
    if m==0 or m==1:
        for i in range(len(models) - 1):
            iter = i + 1
            diff_outputs.append( np.linalg.norm((outputs[iter]-outputs[0]).cpu().numpy(),ord=None))
            # print((outputs[iter]-outputs[0]).size()) 
    if m==2 or m==3:    
        for i in range(1,len(models)):
            num = 0
            s = 0
            for j in range(len(combination)):
                # for j in combination:
                    # print(outputs)
                s += (outputs[0][j][combination[j]]-outputs[i][j][combination[j]])
                    
            diff_outputs.append(s)

    print("resnet同源模型 Correlation difference is:", diff[:10], "Outputs difference is:", diff_outputs[:10])
    print("densenet同源模型 Correlation difference is:", diff[10:21], "Outputs difference is:", diff_outputs[10:21])
    print("盗版剪枝模型 Correlation difference is:", diff[21:31], "Outputs difference is:", diff_outputs[21:31])
    print("last微调盗版模型 Correlation difference is:", diff[31:41], "Outputs difference is:", diff_outputs[31:41])
    print("all微调盗版模型 Correlation difference is:", diff[41:51], "Outputs difference is:", diff_outputs[41:51])
    print("对抗训练模型 Correlation difference is:", diff[51:61], "Outputs difference is:", diff_outputs[51:61])
    print("resnet窃取模型 Correlation difference is:", diff[61:71], "Outputs difference is:", diff_outputs[61:71])
    print("densenet窃取模型 Correlation difference is:", diff[71:81], "Outputs difference is:", diff_outputs[71:81])
    list1 = diff[:10]
    list11 = diff_outputs[:10]
    list2 = diff[10:21]
    list21 = diff_outputs[10:21]
    list3 = diff[21:31]
    list31 = diff_outputs[21:31]
    list4 = diff[31:41]
    list41 = diff_outputs[31:41]
    list5 = diff[41:51]
    list51 = diff_outputs[41:51]
    list6 = diff[51:61]
    list61 = diff_outputs[51:61]
    list7 = diff[61:71]
    list71 = diff_outputs[61:71]
    list8 = diff[71:81]
    list81 = diff_outputs[71:81]
    list9 = diff[81:91]
    list91 = diff_outputs[81:91]
    list10 = diff[91:101]
    list101 = diff_outputs[91:101]
            
    plt.figure( figsize=(10, 6))
    # if m==0:
    #     plt.title('Effect of homologous/piracy model differentiation (SAC-w)',fontsize=20)
    # if m==1:
    #     plt.title('Effect of homologous/piracy model differentiation (SAC-c)',fontsize=20)  # 折线图标题
    # else:
    #     plt.title('Effect of homologous/piracy model differentiation (rf)',fontsize=20) 
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
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
    if m==0:
        plt.savefig(dir + "/Correlation_coefficient_eval(SAC-w).png", bbox_inches='tight')
    elif m==1:
        plt.savefig(dir + "/Correlation_coefficient_eval(SAC-c).png", bbox_inches='tight')
    elif m==2:
        plt.savefig(dir + "/Correlation_coefficient_eval(3rf).png", bbox_inches='tight')
    elif m==3:
        plt.savefig(dir + "/Correlation_coefficient_eval(normal).png", bbox_inches='tight')
    
    plt.close()
    
    
    if 1: 
        plt.figure( figsize=(10, 6))
        # if m==0:
        #     plt.title('Effect of homologous/piracy model differentiation (SAC-w)',fontsize=20)
        # elif m==1:
        #     plt.title('Effect of homologous/piracy model differentiation (SAC-c)',fontsize=20)  # 折线图标题
        # else:
        #     plt.title('Effect of homologous/piracy model differentiation (rf)',fontsize=20) 
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
        plt.xlabel('Models',fontsize=20)  # x轴标题
        # plt.ylabel('Label Score Gap',fontsize=20)  # y轴标题
        plt.ylabel('Euclidean Distance',fontsize=20)  
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.plot(list11, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
        plt.plot(list21, marker='o', markersize=3) 
        plt.plot(list31, marker='o', markersize=3) 
        plt.plot(list41, marker='o', markersize=3) 
        plt.plot(list51, marker='o', markersize=3) 
        plt.plot(list61, marker='o', markersize=3) 
        plt.plot(list71, marker='o', markersize=3) 
        plt.plot(list81, marker='o', markersize=3)
        plt.plot(list91, marker='o', markersize=3) 
        plt.plot(list101, marker='o', markersize=3)  
        plt.legend(['HM_SA', 'HM_DA', 'PM_P', 'PM_FL','PM_FA','PM_Adv','EM_SA_L','EM_DA_L','EM_SA_Pr','EM_DA_Pr'],fontsize=20)  # 设置折线名称   
        if m==0:
            plt.savefig(dir + "/Outputs_difference_eval(SAC-w).png", bbox_inches='tight')
        elif m==1:
            plt.savefig(dir + "/Outputs_difference_eval(SAC-c).png", bbox_inches='tight')
        elif m==2:    
            plt.savefig(dir + "/Outputs_difference_eval(3rf).png", bbox_inches='tight')
        elif m==3:    
            plt.savefig(dir + "/Outputs_difference_eval(normal).png", bbox_inches='tight')
        
        plt.close()
        
    diff = diff_outputs
    #阈值判断
    rate = []  
    for d in np.linspace(0, 500, 500):
        d1 = 0
        d2 = 0
        for i in diff[:21]:
            d1+=i
        for i in diff[21:]:
            d2+=i
        h=0
        for i in diff[:21]:
            if i> d:
                h+=1
        p=0
        for i in diff[21:]:
            if i<= d:
                p+=1
        rate.append(float(h/21)+ float(p/len(diff[21:])))
        print("误检率：",1-float(h/21))  
        print("漏检率：",1-float(p/len(diff[21:])))  
    
    d = rate.index(max(rate))/50
    d1 = 0
    d2 = 0
    for i in diff[:21]:
        d1+=i
    for i in diff[21:]:
        d2+=i

    h=0
    for i in diff[:21]:
        if i> d:
            h+=1
    p=0
    for i in diff[21:]:
        if i<= d:
            p+=1

    print('最佳阈值:',d)
    print("误检率：",1-float(h/21))  
    print("漏检率：",1-float(p/len(diff[21:])))       
        
        

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
        #print(save_path) 
        model = torchvision.models.densenet121(pretrained=False)
        in_feature = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models.append(model)
        
    current_address = "pruning_models"
    file_list = os.listdir(current_address)
    i=0
    for file_address in file_list:
        save_path = os.path.join(current_address, file_address)
        # #print(save_path)
        i+=1
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 10)
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

def loadmodels_test():
    
    models_test = []
    
    for i in range(11):
        save_path = 'model_test/resnet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models_test.append(model)
        
    for i in range(11):
        save_path = 'model_test/densenet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.densenet121(pretrained=False)
        in_feature = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models_test.append(model)
        
    current_address = "pruning_models_test"
    file_list = os.listdir(current_address)
    i=0
    for file_address in file_list:
        save_path = os.path.join(current_address, file_address)
        # #print(save_path)
        i+=1
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 10)
        model = torch.load(save_path)
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models_test.append(model)
        if i==10:
            break
    
    for i in range(10):
        save_path = 'finetune_models_test/finetune_resnet_last_{}.pth'.format(i+10)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models_test.append(model)
        
    for i in range(10):
        save_path = 'finetune_models_test/finetune_resnet_all_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models_test.append(model)
    
    
    for i in range(10):
        save_path = 'adv_models_test/model_adv_resnet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models_test.append(model)
     
        
    for i in range(10):
        save_path = 'extract_l_models_test/extract_model_l_resnet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models_test.append(model)
    
    for i in range(10):
        save_path = 'extract_l_models_test/extract_model_l_dense_{}.pth'.format(i+10)  
        #print(save_path) 
        model = torchvision.models.densenet121(pretrained=False)
        in_feature = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models_test.append(model)
    
    for i in range(10):
        save_path = 'extract_p_models_test/extract_model_p_resnet_{}.pth'.format(i)  
        #print(save_path) 
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models_test.append(model)
    
    for i in range(10):
        save_path = 'extract_p_models_test/extract_model_p_dense_{}.pth'.format(i+10)  
        #print(save_path) 
        model = torchvision.models.densenet121(pretrained=False)
        in_feature = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(save_path))
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        models_test.append(model)
       
        
    return models_test

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


if __name__ == '__main__':

    #加载模型 
    
    # models = loadmodels()
    # dir = './show_rf_cifar100' 
    # models = loadmodels_cifar100()
    
    dir = './show_rf_tinyImageNet' 
    models = loadmodels_tinyImageNet()
    # models_test = loadmodels_test()
    #计算相关性
    cal_correlation(models)
    





