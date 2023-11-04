import torch
import torch.nn as nn
import torch.nn.functional as F
def si(img):
    color_img=img
    img_gray=torch.mean(color_img,dim=0)
    m,_=img_gray.shape
    img_gray=img_gray[None,None,:].float()
    #print(img_gray.shape)
    h_sobel_kernel=torch.tensor([[[[1,0,-1],[2,0,-2],[1,0,-1]]]]).float()
    #print(h_sobel_kernel.shape)
    v_sobel_kernel=torch.tensor([[[[1,2,1],[0,0,0],[-1,-2,-1]]]]).float()
    #print(v_sobel_kernel.shape)
    s_h=F.conv2d(img_gray,h_sobel_kernel).squeeze()
    s_v=F.conv2d(img_gray,v_sobel_kernel).squeeze()
    s_r=torch.sqrt(s_h**2+s_v**2)
    SI=torch.sqrt((m/1080)*torch.mean(s_r**2))
    return SI.item()
def cf(img):
    color_img=img
    rg=color_img[0,:,:]-color_img[1,:,:]
    by=0.5*(color_img[0,:,:]+color_img[1,:,:])-color_img[2,:,:]
    mu_rg=torch.mean(rg)
    mu_by=torch.mean(by)
    var_rg=torch.mean(rg**2)-mu_rg**2
    var_by=torch.mean(by**2)-mu_by**2
    CF=torch.sqrt(var_rg+var_by)+0.3*torch.sqrt((mu_rg**2+mu_by**2))
    return CF.item()

    
'''out=cf(torch.randn([3,230,310]))
print(out)'''
