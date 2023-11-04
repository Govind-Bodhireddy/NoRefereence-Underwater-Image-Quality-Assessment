import torch
import numpy as np
from color_moments import c_m
import torch.nn.functional as F
import torchvision.transforms as T
import mapminmax
import estimaggdparam
from PIL import Image
from skimage.color import rgb2gray
import Difference_Map
import S
import A
import A_x
import A_y
import MSCNM
import GM_LBP
def feat_extract(tensor):
    tensor=tensor.float()
    r_ch,g_ch,b_ch=tensor[0,:,:],tensor[1,:,:],tensor[2,:,:]
    #  Oponent-color space
    O1=(r_ch-g_ch)/torch.sqrt(torch.tensor(2))
    O2=(r_ch+g_ch-2*b_ch)/torch.sqrt(torch.tensor(6))
    O3=((r_ch+g_ch+b_ch))/torch.sqrt(torch.tensor(3))
    
    vector=c_m(tensor)
    moment=torch.abs(torch.tensor(vector))
    #print(moment.shape)
    # chromatic features extraction
    D=torch.abs(O1-O2)
    D_AGGD=Difference_Map.D_AGGD(D)
    #print(D_AGGD.shape)
    #S_AGGD shift
    S_AGGD=S.S_AGGD(O_1=O1,O_2=O2)
    #print(S_AGGD.shape)
    
    #A_AGGD
    A_AGGD=A.A_AGGD(O_1=O1,O_2=O2)
    #print(A_AGGD.shape)
    #Ax_AGGD
    Ax_AGGD=A_x.Ax_AGGD(O_1=O1)
    #print(Ax_AGGD.shape)
    #Ay_AGGD
    Ay_AGGd=A_y.Ay_AGGD(O_2=O2)
    #print(Ay_AGGd.shape)
    #MSCNM
    O3_LBP=torch.tensor(MSCNM.MSCNM(tensor=tensor))
    #print(O3_LBP.shape)
    #GM_LBP and GO_LBP
    GM_LBP_feat,GO_LBP_feat=GM_LBP.GMO(tensor=tensor)
    #print(GM_LBP_feat.shape)
    #print(GO_LBP_feat.shape)
    #final_feature
    final_feature=torch.hstack([D_AGGD,S_AGGD,A_AGGD,Ax_AGGD,Ay_AGGd,moment,O3_LBP,GM_LBP_feat,GO_LBP_feat])
    #print(final_feature.shape)
    return final_feature





t_img=Image.open('/home/govind/Benchmark/Enhanced/group_1/1/1_BL-TM.png')
t_img=torch.tensor(np.array(t_img)).permute(2,1,0)
v=feat_extract(t_img)
print(v.shape)