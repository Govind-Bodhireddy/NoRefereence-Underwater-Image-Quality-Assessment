import numpy as np
import torch 
import torch.nn.functional as F
from PIL import Image
import getmapping
import lbp
def GM(tensor):
    RGB_IMAGE=tensor
     # 0.2989 * R + 0.5870 * G + 0.1140 * B 
    gray_IMAGE=0.2989*RGB_IMAGE[0,:,:]+0.587*RGB_IMAGE[1,:,:]+0.114*RGB_IMAGE[2,:,:]
    gray_IMAGE=torch.round(gray_IMAGE)
    sigma=0.5
    x=torch.tensor([[-1,0,1],[-1,0,1],[-1,0,1]])
    y=torch.tensor([[-1,-1,-1],[0,0,0],[1,1,1]])
    gdx=-(x/(2*torch.pi*sigma**4))*torch.exp(-(x**2+y**2)/(2*sigma**2))
    gdy=-(y/(2*torch.pi*sigma**4))*torch.exp(-(x**2+y**2)/(2*sigma**2))
    im_dx=F.conv2d(gray_IMAGE[None,None],gdx[None,None],padding='same').squeeze()
    im_dy=F.conv2d(gray_IMAGE[None,None],gdy[None,None],padding='same').squeeze()
    GM=torch.sqrt(im_dx**2+im_dy**2)
    GM=((GM-torch.min(GM))/(torch.max(GM)-torch.min(GM)))*255
    GM=torch.round(GM).int()
    mapping=getmapping.getmapping(samples=8,mappingtype='riu2')
    GM=GM.numpy()
    LBP_feat_GM=lbp.lbp(GM,1,8,mapping,'nh')
    return LBP_feat_GM


t_img=Image.open('/home/govind/Benchmark/Enhanced/group_1/1/1_BL-TM.png')
t_img=torch.tensor(np.array(t_img)).permute(2,1,0)
tensor=t_img.float()
GM(tensor=tensor)