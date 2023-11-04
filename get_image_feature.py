import os
import numpy as np
from PIL import Image
import torch
import feat_ext
import pickle
group1='/home/govind/Benchmark/Enhanced/group_1'
group2='/home/govind/Benchmark/Enhanced/group_2'
group3='/home/govind/Benchmark/Enhanced/group_3'
group4='/home/govind/Benchmark/Enhanced/group_4'
group5='/home/govind/Benchmark/Enhanced/group_5'
group=[group1,group2,group3,group4,group5]
types=np.arange(1,21,1)
types=list(map(str,types))
list1=['_BL-TM.png','_GL-net.png','_histogram prior.png','_RayleighDistribution.png','_retinex-based.png',
       '_RGHS.png','_two-step-based.png','_UDCP.png','_UWCNN.png','_Water-net.png']
data_path=[]
features=np.zeros([1000,59])
for i in group:
    for j in range(len(types)):
        for k in range(len(list1)):
            data_path.append(os.path.join(i,os.path.join(types[j],types[j]+list1[k])))
for i in range(len(data_path)):
    print(f"Exctarting features of the Image :{i}")
    img=torch.tensor(np.array(Image.open(data_path[i]))).permute(2,1,0)
    features[i,:]=feat_ext.feat_extract(img)
    print(features[i,:])
print(features.shape)
with open('/home/govind/QA/features1.pkl','wb') as f:
    pickle.dump(features,f)