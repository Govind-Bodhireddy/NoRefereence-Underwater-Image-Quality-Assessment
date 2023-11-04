import torch 
import numpy as np
from PIL import Image
import torch.nn.functional as F
import getmapping
import lbp

from Difference_Map import gaussain_kernel
def MSCNM(tensor):
    RGB_IMAGE=tensor
     # 0.2989 * R + 0.5870 * G + 0.1140 * B 
    gray_IMAGE=0.2989*RGB_IMAGE[0,:,:]+0.587*RGB_IMAGE[1,:,:]+0.114*RGB_IMAGE[2,:,:]
    gray_IMAGE=torch.round(gray_IMAGE)
    scale_num=2
    k_num=0.01
    C1=(k_num*255)**2
    window=gaussain_kernel(size=7,sigma=7/6)
    window/=torch.sum(window)
    mu =F.conv2d(gray_IMAGE[None,None],window[None,None],padding='same').squeeze()
    mu=torch.round(mu)
    mu_sq=mu*mu
    gray_IMAGE_2=gray_IMAGE*gray_IMAGE
    sigma=torch.sqrt(torch.abs(F.conv2d(gray_IMAGE_2[None,None],window[None,None],padding='same'))-mu_sq).squeeze()
    sigma=torch.round(sigma)
    structdis = (gray_IMAGE-mu)/(sigma+C1)
    structdis=torch.round(structdis)
    structdis=structdis.numpy()
    mapping=getmapping.getmapping(8,'riu2')
    LBP_feat_O3=lbp.lbp(structdis,1,8,mapping,'nh')
    O3_LBP=LBP_feat_O3

    return O3_LBP
    

    
    #print(sigma.shape)
'''t_img=Image.open('/home/govind/Benchmark/Enhanced/group_1/1/1_BL-TM.png')
t_img=torch.tensor(np.array(t_img)).permute(2,1,0)
tensor=t_img.float()
MSCNM(tensor=tensor)'''