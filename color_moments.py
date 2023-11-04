from skimage import color
import torch
import numpy as np
from PIL import Image
def c_m(img):
    img=img.permute(2,1,0).float()
    lab=color.rgb2lab(img)
    L_Img=lab[:,:,0]
    A_img=lab[:,:,1]
    B_img=lab[:,:,2]
    m,n=L_Img.shape
    N=m*n
    h_mean=np.mean(L_Img)
    s_mean=np.mean(A_img)
    v_mean=np.mean(B_img)
    h_sig=np.sqrt(np.var(L_Img))
    s_sig=np.sqrt(np.var(A_img))
    v_sig=np.sqrt(np.var(B_img))
    h3=np.sum(np.sum(L_Img-h_mean,axis=1)**3)
    hSke=np.cbrt((h3/N))
    s3=np.sum(np.sum(A_img-s_mean,axis=1)**3)
    sSke=np.cbrt((s3/N))
    v3=np.sum(np.sum(B_img-v_mean,axis=1)**3)
    vSke=np.cbrt((v3/N))
    vectors=[h_mean,h_sig,hSke,s_mean,s_sig,sSke,v_mean,v_sig,vSke]
    return vectors



'''v=c_m(torch.randn(3,230,320))
print(len(v))
print(v)'''