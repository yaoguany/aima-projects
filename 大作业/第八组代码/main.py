import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import argparse
import random
from sklearn.model_selection import KFold
from grid import *
from model import ResNet_18
import pdb

def train_epoch(model, device, dataloader, enpoch, loss_fn, optimizer, scheduler):
    train_loss, train_correct = 0.0, 0
    model.train()
    for images, labels in dataloader:
        #pdb.set_trace()
        images, labels = images.to(device), labels.to(device)
        #images = train_transform(images)
        #数据增强
        images = grid(images)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        #采用top-1准确率
        #对网络输出的归一化概率选取概率最大的位置作为网络预测输出与标签进行对比
        scores, predictions = torch.max(output, 1)
        _, labels = torch.max(labels, 1)
        #如果相同则预测正确
        train_correct += (predictions == labels).sum().item()
    #随训练轮次调整学习率
    #初始学习率设置为0.1，在训练到第20/50/75轮时学习率乘以0.1
    scheduler.step(enpoch)
    return train_loss, train_correct

def valid_epoch(model, device, dataloader, loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output, labels)
        valid_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        _, labels=torch.max(labels, 1)
        val_correct += (predictions == labels).sum().item()

    return valid_loss, val_correct

if __name__ == '__main__':
    device = torch.device('cuda'if torch.cuda.is_available() else 'mps')
    #定义参数，便于在命令行运行
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", type = str, default = "/Volumes/移动硬盘/aima/人工智能实验课材料/Adience数据集/Folds/original_txt_files",
                        help = "Path of label data")
    parser.add_argument("--data_path", type = str, default = "/Volumes/移动硬盘/aima/人工智能实验课材料/Adience数据集/aligned",
                        help = "Path of img data")
    parser.add_argument("--epoch", type = int, default = 100)
    parser.add_argument("--n_classes", type = int, default = 8)
    parser.add_argument("--batchsize", type = int, default = 32)
    parser.add_argument('--d1', type = int, default = 96, help = 'd1')
    parser.add_argument('--d2', type = int, default = 224, help = 'd2')
    parser.add_argument('--rotate', type = int, default = 360, help = 'rotate the mask')
    parser.add_argument('--ratio', type = float, default = 0.6, help = 'ratio')
    parser.add_argument("--lr", type = float, default = 0.1)
    
    parser.add_argument('--grid', action = 'store_true', default = True,
                        help = 'apply grid')
    parser.add_argument('--mode', type = int, default = 1,
                        help = 'GridMask (1) or revised GridMask (0)')
    parser.add_argument('--prob', type = float, default = 0.8,
                        help = 'max prob')
    parser.add_argument('--st_epochs', type = float, default = 240,
                        help='epoch when archive max prob')
    
    parser.add_argument('--lr_adjust_step', default = [20, 50, 75], type = int, nargs = '+',
                        help = 'initial learning rate')
    parser.add_argument('--lr_adjust_type', default = 'step', type = str,
                        help = 'lr adjust type')
    parser.add_argument('--gamma', default = '0.99', type = float,
                        help = 'exp lr gamma')
    parser.add_argument('--momentum', default = 0.9, type = float, metavar = 'M',
                        help = 'momentum')

    parser.add_argument('--weight-decay', '--wd', default = 1e-4, type = float,
                        metavar = 'W', help = 'weight decay (default: 1e-4)')
    parser.add_argument('--k', default = 5, type = int,)
    args = parser.parse_args()
    
    #是否数据增强
    #根据原始论文中的结果选择:
    #d1 = 96, d2 = 224, ratio = 0.6
    if args.grid:
        grid = GridMask(args.d1, args.d2, args.rotate, args.ratio, args.mode, args.prob)
    #读入数据
    data_0 = pd.read_csv(os.path.join(args.label_path+'/fold_0_data.txt'), delimiter='\t')
    data_1 = pd.read_csv(os.path.join(args.label_path+'/fold_1_data.txt'), delimiter='\t')
    data_2 = pd.read_csv(os.path.join(args.label_path+'/fold_2_data.txt'), delimiter='\t')
    data_3 = pd.read_csv(os.path.join(args.label_path+'/fold_3_data.txt'), delimiter='\t')
    data_4 = pd.read_csv(os.path.join(args.label_path+'/fold_4_data.txt'), delimiter='\t')
    data = [data_0, data_1, data_2, data_3, data_4]
    #数据合并
    data = pd.concat(data)
    #丢弃'age'这列中有缺失值的行
    data = data.dropna(axis = 0, subset=['age'])
    #train_df, valid_df = train_test_split(data, test_size=0.2, random_state=42)
    classes = ('0-2', '4-6', '8-13', '15-20', '25-32', '38-43', '48-53', '60-')

    class GenderDataset(Dataset):
        def __init__(self, root_dir, df, transform = None):
            self.root_dir = root_dir
            self.subdir = df['user_id'].tolist()
            self.face_id = df['face_id'].tolist()
            self.filename = df['original_image'].tolist()
            self.label = df['age'].tolist()
            self.transform = transform
            
        #返回第(idx + 1)个图片及其label    
        def __getitem__(self, idx):
            filename = 'landmark_aligned_face.' + \
                str(self.face_id[idx]) + '.' + self.filename[idx]
            img_path = os.path.join(self.root_dir, self.subdir[idx], filename)
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            label = self.label[idx]
            try:
                label=label[1:-1]
                label=tuple(map(int, label.split(', ')))
                label= int(label[-1])
            except:
                age=random.randint(0, 7)
                agelist=[0, 0, 0, 0, 0, 0, 0, 0]
                agelist[age]=1
                return img, torch.Tensor(agelist)

            if label <= 2:
                age=[1, 0, 0, 0, 0, 0, 0, 0]
            elif label <= 6:
                age=[0, 1, 0, 0, 0, 0, 0, 0]
            elif label <= 13:
                age=[0, 0, 1, 0, 0, 0, 0, 0]
            elif label <= 20:
                age=[0, 0, 0, 1, 0, 0, 0, 0]
            elif label <= 32:
                age=[0, 0, 0, 0, 1, 0, 0, 0]
            elif label <= 43:
                age=[0, 0, 0, 0, 0, 1, 0, 0]
            elif label <= 53:
                age=[0, 0, 0, 0, 0, 0, 1, 0]
            else:
                age=[0, 0, 0, 0, 0, 0, 0, 1]
            return img, torch.Tensor(age)
        
        #返回数据集元素个数
        def __len__(self):
            return len(self.label)

    def accuracy(prediction, ground_truth):
        num_correct = (np.array(prediction) == np.array(ground_truth)).sum()
        return num_correct / len(prediction)

    #数据的预处理
    train_transform = transforms.Compose([
        #将数据集中的图片变换为224 x 224分辨率
        transforms.Resize((224, 224)),
        #随机水平翻转
        transforms.RandomHorizontalFlip(),
        #随机旋转20度以内
        transforms.RandomRotation(20),
        #亮度随机改变原图0.1
        #对比度随机改变原图0.1
        #饱和度随机改变原图0.1
        #色调随机改变原图0.02
        #尽可能排除了光照和角度等因素的影响
        transforms.ColorJitter(brightness = 0.1, contrast = 0.1,
                            saturation = 0.1, hue = 0.02),
        #转换数据类型
        transforms.ToTensor(),
        #通过查阅先前论文
        #选择将rgb三个通道的
        #均值变为[0.485, 0.456, 0.406]
        #方差变为[0.229, 0.224, 0.225]
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225])
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225])
    ])
    dataset = GenderDataset(
        args.data_path, data, transform = valid_transform)
    #实例化
    model = ResNet_18().to(device)
    #用SGD(随机梯度下降)优化
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum = args.momentum, weight_decay = args.weight_decay)
    #调整学习率
    if args.lr_adjust_type == 'step':
        scheduler = MultiStepLR(optimizer, milestones = args.lr_adjust_step, gamma = 0.1)
    elif args.lr_adjust_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, args.epochs)
    elif args.lr_adjust_type == 'exp':
        scheduler = ExponentialLR(optimizer, args.gamma)

    #k折交叉验证
    #k = 5
    #4折作为训练集，1折为测试集
    splits = KFold(n_splits = args.k, shuffle = True, random_state = 42)
    foldperf={}
    #损失函数选择交叉熵损失函数
    loss_fun = nn.CrossEntropyLoss()
    #train(model, train_loader,loss_fun,args.epoch , optimizer, device,scheduler)
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        #初始化训练集和验证集
        #每一折均需初始化
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        model=ResNet_18().to(device)
        train_loader = DataLoader(dataset, batch_size = args.batchsize, sampler = train_sampler)
        test_loader = DataLoader(dataset, batch_size = args.batchsize, sampler = test_sampler)
        print('Fold {}'.format(fold + 1))
        history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
        if args.lr_adjust_type == 'step':
            scheduler = MultiStepLR(optimizer, milestones = args.lr_adjust_step, gamma=0.1)
        elif args.lr_adjust_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, args.epochs)
        elif args.lr_adjust_type == 'exp':
            scheduler = ExponentialLR(optimizer, args.gamma)
        for epoch in range(args.epoch):
            #训练并进行测试
            train_loss, train_correct = train_epoch(model,device, train_loader, epoch, loss_fun, optimizer, scheduler)
            test_loss, test_correct = valid_epoch(model,device, test_loader, loss_fun)
            #计算正确率
            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                                args.epoch,
                                                                                                                train_loss,
                                                                                                                test_loss,
                                                                                                                train_acc,
                                                                                                                test_acc))
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
        #存储每折的模型
        foldperf['fold{}'.format(fold+1)] = history  
        torch.save(model.state_dict(), os.path.join('net_fold{}.pth'.format(fold)))
