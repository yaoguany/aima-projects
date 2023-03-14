# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy import stats
import scipy.io as scio
#import pdb
# EM算法的实现
def em(iter):
    '''
    h为输入的数据
    mu1，sigma1,w1分别为A的初始均值，方差，权重
    mu2，sigma2,w2分别为B的初始均值，方差，权重
    '''
    #E-step,将中间值计算出来加速计算
    temp1=w1m[iter-1]*stats.norm(mu1m[iter-1],np.sqrt(sigma1m[iter-1])).pdf(h)
    temp2=w2m[iter-1]*stats.norm(mu2m[iter-1],np.sqrt(sigma2m[iter-1])).pdf(h)
    temp3=temp1+temp2
    r1=temp1/temp3
    r2=temp2/temp3
    #更新均值
    mu1m[iter]=np.sum(r1*h)/sum(r1)
    mu2m[iter]=np.sum(r2*h)/sum(r2)
    #更新方差
    #pdb.set_trace()
    sigma1m[iter]=np.sum(r1*(h-mu1m[iter])**2)/sum(r1)
    sigma2m[iter]=np.sum(r2*(h-mu2m[iter])**2)/sum(r2)
    #更新权重
    w1m[iter]=sum(r1)/lenth
    w2m[iter]=sum(r2)/lenth
    
    
    
# 加载数据
dataFile = '/Volumes/移动硬盘/aima/EM算法/DATA.mat'
h = scio.loadmat(dataFile)
h=h['C']
h=h.flatten()

# GMM的构造
# Step 1.首先根据经验来分别对A,B班级的均值、方差和权值进行初始化,如：
'''
mu1 = ???
sigma1 = ???
w1 = ???  # A
mu2 = ???
sigma2 = ???
w2 = ???  # B
'''

lenth = len(h)  # 样本长度
print(sum(stats.norm(75,30).pdf(h))/lenth)
w1=0.3
w2=0.7
sigma1=15
sigma2=30
m1=60
m2=100
mu1m = np.zeros(50)
mu2m = np.zeros(50)
w1m = np.zeros(50)
w2m = np.zeros(50)
sigma1m = np.zeros(50)
sigma2m = np.zeros(50)
#权重，均值，方差初始化
w1m[0] = w1
w2m[0] = w2
mu1m[0]=m1
mu2m[0]=m2
sigma1m[0] = sigma1
sigma2m[0] = sigma2
# 开始EM算法的主循环
for i in range(1, 50):
    em(i)

# 作图
data=[i for i in range(60,101)]
h1=stats.norm(mu1m[49],sigma1m[49]).pdf(data)
h2=stats.norm(mu2m[49],sigma2m[49]).pdf(data)
plt.figure(1)
plt.hist(h)
plt.figure(2)
plt.subplot(121)
plt.plot(mu1m)
plt.subplot(122)
plt.plot(mu2m)
plt.figure(3)
plt.subplot(121)
plt.plot(sigma1m)
plt.subplot(122)
plt.plot(sigma2m)
plt.figure(4)
plt.subplot(121)
plt.plot(w1m)
plt.subplot(122)
plt.plot(w2m)
plt.figure(5)
plt.subplot(131)
plt.plot(h1)
plt.subplot(132)
plt.plot(h2)
plt.subplot(133)
plt.plot(w1m[49]*h1+w2m[49]*h2)
plt.show()