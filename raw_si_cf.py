import os
import torch
import numpy as np
import SI_CF
from PIL import Image
import pickle
fldr_path='/home/govind/Benchmark/Raw'
lis_dir=os.listdir('/home/govind/Benchmark/Raw')
print(lis_dir)
img_path=[]
for path in lis_dir:
    i_p=os.path.join(fldr_path,path)
    Images_path=os.listdir(i_p)
    for path2 in Images_path:
        img_path.append(os.path.join(i_p,path2))
SI=[]
CF=[]
for img in img_path:
    raw_image=Image.open(img)
    raw_img=torch.tensor(np.array(raw_image)).squeeze().permute(2,1,0).float()
    #print(raw_img.shape)
    SI.append(SI_CF.si(raw_img))
    CF.append(SI_CF.cf(raw_img))
SI_CF=SI+CF
print(len(SI_CF))
with open("/home/govind/QA/sivscf/si.pkl",'wb') as f:
    pickle.dump(SI_CF,f)

