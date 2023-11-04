import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import estimaggdparam
import mapminmax
from math import gamma
def gaussain_kernel(size,sigma):
        l=size//2
        x=np.arange(-l,l+1,1)
        #print(x)
        x,y=np.meshgrid(x,x)
        g_k=np.exp(-1*(x**2+y**2)/(2*sigma**2))/(2*np.pi*sigma**2)
        g_k=torch.tensor(g_k).float()
        return g_k
#g_k=gaussain_kernel()
#print(g_k[3,3])
def D_AGGD(D):
        #print(D.shape)
        k_num=0.01
        c1=(k_num*255)**2
        window=gaussain_kernel(size=7,sigma=7/6)
        window=window/(torch.sum(window))
        #print(window)
        mu=F.conv2d(D[None, None], window[None, None], padding='same').squeeze()
        #print(mu.shape)
        D_2=D*D
        sigma=torch.sqrt(torch.abs(F.conv2d(D_2[None, None], window[None, None], padding='same').squeeze())-mu**2)
        structdis=(D-mu)/(sigma+c1)
        structdis=structdis.flatten().view(-1)
       # structdis=mapminmax.fun(tensor=structdis,min=0,max=1)
        #print(structdis.shape)
        alpha,leftstd,righstd=estimaggdparam.estparam(tensor=structdis)
        #print(alpha)
        #print(leftstd)
        #print(righstd)
        const=(np.sqrt(gamma(1/alpha)))/(np.sqrt(gamma(3/alpha)))
        #print(const)
        meanparam=(righstd-leftstd)*(gamma(2/alpha)/(gamma(1/alpha))*const)
        #print(meanparam)
         #Difference map
        D_feat=torch.tensor([alpha,meanparam,leftstd**2,righstd**2])
        return D_feat

t_img=Image.open('/home/govind/Benchmark/Enhanced/group_1/1/1_BL-TM.png')
t_img=torch.tensor(np.array(t_img)).permute(2,1,0)
tensor=t_img.float()

r_ch,g_ch,b_ch=tensor[0,:,:],tensor[1,:,:],tensor[2,:,:]
O1=(r_ch-g_ch)/torch.sqrt(torch.tensor(2))
O2=(r_ch+g_ch-2*b_ch)/torch.sqrt(torch.tensor(6))
O3=((r_ch+g_ch+b_ch))/torch.sqrt(torch.tensor(3))
x=D_AGGD(torch.abs(O1-O2))
print(x)