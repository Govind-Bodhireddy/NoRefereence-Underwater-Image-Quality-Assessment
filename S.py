from PIL import Image
import torch
import numpy as np
import estimaggdparam
from math import gamma
def S_AGGD(O_1,O_2):
    S=torch.sqrt(O_1**2+O_2**2)
    Shifts=torch.tensor([[0,1],[1,0],[1,1],[-1,1]])
    shifted_S=torch.roll(S,shifts=(0,1),dims=(0,1))
    diff=S-shifted_S
    m,n=diff.shape
    diff=diff[1:m-1,1:n-1]
    diff=diff.flatten().view(-1)
    alpha,leftstd,rightstd=estimaggdparam.estparam(diff)
    const=(np.sqrt(gamma(1/alpha)))/(np.sqrt(gamma(3/alpha)))
    meanparam=(rightstd-leftstd)*(gamma(2/alpha)/gamma(1/alpha))*const
    S_feat=torch.tensor([alpha,meanparam,leftstd**2,rightstd**2])
    return S_feat

'''t_img=Image.open('/home/govind/Benchmark/Enhanced/group_1/1/1_BL-TM.png')
t_img=torch.tensor(np.array(t_img)).permute(2,1,0)
tensor=t_img.float()
r_ch,g_ch,b_ch=tensor[0,:,:],tensor[1,:,:],tensor[2,:,:]
O1=(r_ch-g_ch)/torch.sqrt(torch.tensor(2))
O2=(r_ch+g_ch-2*b_ch)/torch.sqrt(torch.tensor(6))
O3=((r_ch+g_ch+b_ch))/torch.sqrt(torch.tensor(3))
print(S_AGGD(O1,O1))'''