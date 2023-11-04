import torch
import numpy as np
import gamma
def estparam(tensor):
    gam=torch.arange(0.2,10.001,0.001)
    r_gam=(gamma.gamma(2/gam)**2)/(gamma.gamma(1/gam)*gamma.gamma(3/gam))
    #print(r_gam[0])
    leftstd=torch.sqrt(torch.mean(tensor[tensor<0]**2))
    rightstd=torch.sqrt(torch.mean(tensor[tensor>0]**2))
    gammahat=leftstd/rightstd
    rhat=(torch.mean(torch.abs(tensor))**2)/(torch.mean(tensor**2))
    rhatnorm=((rhat*((gammahat**3)+1))*(gammahat+1))/(((gammahat**2)+1)**2)
    diff=(r_gam-rhatnorm)**2
    array_position=torch.argmin(diff)
    #print(array_position)
    alpha=gam[array_position]
    #features=torch.tensor([alpha,leftstd,rightstd])
    return alpha,leftstd,rightstd                                  
#x=estparam(torch.arange(-10,100,1).float())
#print(x)