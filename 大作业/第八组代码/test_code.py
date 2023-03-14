import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import argparse
import random
from grid import *
from model import ResNet_18
import pdb
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
unloader = transforms.ToPILImage()

#制图，画出参数的分布
#x-axis: weight
#y-axis: density
def plot_weight_distribution(model, bins = 256, count_nonzero_only = False):
    fig, axes = plt.subplots(3,7, figsize = (8, 6))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            if count_nonzero_only:
                param_cpu = param.detach().view(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0].view(-1)
                ax.hist(param_cpu, bins = bins, density = True, 
                        color = 'blue', alpha = 0.5)
            else:
                ax.hist(param.detach().view(-1).cpu(), bins = bins, density = True, 
                        color = 'blue', alpha = 0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()



def get_num_parameters(model: nn.Module, count_nonzero_only = False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width = 32, count_nonzero_only = False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

#格式转换
Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

def test(net, test_loader, device, classes, showerrors = False):
    correct_pred = {classname:0 for classname in classes}
    total_pred = {classname:0 for classname in classes}
    with torch.no_grad():
        current_time=time.time()
        total = 0
        score = 0
        one_off_score=0
        for i, (image, target) in enumerate(test_loader):
            image = image.to(device)
            target = target.to(device)
            _,target=torch.max(target,1)
            result = net(image)
            _, prediction = torch.max(result.data,dim=1)
            #pdb.set_trace()
            #showerrors默认为False
            #当showerrors == True时，可获取分类错误图片
            if showerrors:
                errorplace=torch.where(prediction!=target)
                #进行拷贝
                result = result.cpu().clone()
                image = image.cpu().clone()
                image = image.squeeze(0)
                #进行保存
                for j in errorplace[0]:
                    print(result[j])
                    errorimage=image[j]
                    errorimage = unloader(errorimage)
                    errorimage.save('/Users/yaoguanyu/Desktop/code/error/'+str(i)+'_'+str(j)+'.jpg')
                break
            one_off_score+=torch.sum(torch.abs(prediction.cpu()-target.cpu())<=1).item()
            score += torch.sum(torch.argmax(result,dim=1)==target).item()
            total += len(target)
            for tar,predict in zip(target,prediction):
                if tar == predict:
                    correct_pred[classes[tar]] += 1
                total_pred[classes[tar]] += 1
        print("inference time:",time.time()-current_time)
        print("evalue accuracy:{:.2f}%".format(100*score/total))
        print("one-off accuracy:{:.2f}%".format(100*one_off_score/total))


if __name__ == '__main__':
    #下列部分代码在main.py中重复出现
    #故不再重复注释
    device = torch.device('cuda'if torch.cuda.is_available() else 'mps')
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", type=str, default="/Volumes/移动硬盘/aima/人工智能实验课材料/Adience数据集/Folds/original_txt_files",
                        help="Path of label data")
    parser.add_argument("--data_path", type=str, default="/Volumes/移动硬盘/aima/人工智能实验课材料/Adience数据集/aligned",
                        help="Path of img data")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--n_classes", type=int, default=8)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument('--d1', type=int, default=96,help='d1')
    parser.add_argument('--d2', type=int, default=224,help='d2')
    parser.add_argument('--rotate', type=int, default=360,help='rotate the mask')
    parser.add_argument('--ratio', type=float, default=0.6,help='ratio')
    parser.add_argument("--lr", type=float, default=0.1)
    
    parser.add_argument('--grid', action='store_true', default=True,
                        help='apply grid')
    parser.add_argument('--mode', type=int, default=1,
                        help='GridMask (1) or revised GridMask (0)')
    parser.add_argument('--prob', type=float, default=0.8,
                        help='max prob')
    parser.add_argument('--st_epochs', type=float, default=240,
        help='epoch when archive max prob')
    
    parser.add_argument('--lr_adjust_step', default=[100,200,265], type=int, nargs='+',
                        help='initial learning rate')
    parser.add_argument('--lr_adjust_type', default='step', type=str,
                        help='lr adjust type')
    parser.add_argument('--gamma', default='0.99', type=float,
                        help='exp lr gamma')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')

    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--k', default=5, type=int,)
    args = parser.parse_args()
    

    
    data_0 = pd.read_csv(os.path.join(args.label_path+'/fold_0_data.txt'), delimiter='\t')
    data_1 = pd.read_csv(os.path.join(args.label_path+'/fold_1_data.txt'), delimiter='\t')
    data_2 = pd.read_csv(os.path.join(args.label_path+'/fold_2_data.txt'), delimiter='\t')
    data_3 = pd.read_csv(os.path.join(args.label_path+'/fold_3_data.txt'), delimiter='\t')
    data_4 = pd.read_csv(os.path.join(args.label_path+'/fold_4_data.txt'), delimiter='\t')
    data = [data_0, data_1, data_2, data_3, data_4]
    data = pd.concat(data)
    data = data.dropna(axis = 0,subset=['age'])
    classes = ('0-2','4-6','8-13','15-20','25-32','38-43','48-53','60-')

    class GenderDataset(Dataset):
        def __init__(self, root_dir, df, transform=None):
            self.root_dir = root_dir
            self.subdir = df['user_id'].tolist()
            self.face_id = df['face_id'].tolist()
            self.filename = df['original_image'].tolist()
            self.label = df['age'].tolist()
            self.transform = transform
            
            
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
                agelist=[0,0,0,0,0,0,0,0]
                agelist[age]=1
                return img, torch.Tensor(agelist)

            if label<=2:
                age=[1,0,0,0,0,0,0,0]
            elif label<=6:
                age=[0,1,0,0,0,0,0,0]
            elif label<=13:
                age=[0,0,1,0,0,0,0,0]
            elif label<=20:
                age=[0,0,0,1,0,0,0,0]
            elif label<=32:
                age=[0,0,0,0,1,0,0,0]
            elif label<=43:
                age=[0,0,0,0,0,1,0,0]
            elif label<=53:
                age=[0,0,0,0,0,0,1,0]
            else:
                age=[0,0,0,0,0,0,0,1]
            return img, torch.Tensor(age)

        def __len__(self):
            return len(self.label)

    def accuracy(prediction, ground_truth):
        num_correct = (np.array(prediction) == np.array(ground_truth)).sum()
        return num_correct / len(prediction)


    valid_transform = transforms.Compose([
        #进行预处理
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    dataset=GenderDataset(
        args.data_path, data, transform=valid_transform)
    model=ResNet_18()
    state_dict=torch.load('/Volumes/移动硬盘/aima/人工智能实验课材料/大作业/第八组代码/net_fold0.pth',map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.to(device)
 
    test_loader = DataLoader(dataset, batch_size=240, shuffle=True)
    modelsize=get_model_size(model)
    print(f"our model has size={modelsize/MiB:.2f} MiB")
    #pdb.set_trace()
    test(model,test_loader,  device,classes,showerrors=False)
    plot_weight_distribution(model)