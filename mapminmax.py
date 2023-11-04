import torch
def fun(tensor,max,min):
    x=(tensor-torch.min(tensor))/(torch.max(tensor)-torch.min(tensor))
    x=x*(max-min)+min
    return x
#x=fun(torch.tensor([1,2,3,3,4]),max=1,min=-1)
#print(x)