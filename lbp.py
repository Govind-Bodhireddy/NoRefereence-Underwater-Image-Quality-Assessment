import torch
import numpy as np

def lbp(*args):
    Image=args[0]
    raduis=args[1]
    neigbhour=args[2]
    mapping=args[3]
    mode=args[4]
    table=mapping[0]
    samples=mapping[1]
    newMax=mapping[2]
    ysize,xsize=np.shape(Image)
    s_points=np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]])
    miny=min(s_points[:,0])
    maxy=max(s_points[:,0])
    minx=min(s_points[:,1])
    maxx=max(s_points[:,1])

    bsizey=np.ceil(max([maxy,0]))-np.floor(min([miny,0]))+1
    bsizex=np.ceil(max([maxx,0]))-np.floor(min([minx,0]))+1

    origy=1-np.floor(min([miny,0]))
    origx=1-np.floor(min([minx,0]))

    dx=xsize-bsizex
    dy=ysize-bsizey
    nexty=int(origy+dy)
    nextx=int(origx+dx)
    C=Image[int(origy):nexty,int(origx):nextx]
    bins= 2**neigbhour

    result=np.zeros([int(dy),int(dx)])
    for i in range(neigbhour):
        y=s_points[i][0]+origy
        x=s_points[i][1]+origx

        fy=np.floor(y) 
        cy=np.ceil(y)
        ry=np.round(y)

        fx=np.floor(x)
        cx=np.ceil(x)
        rx=np.round(x)

        if (abs(x-rx)<10**-6 and abs(y-ry)<10**-6):
            N=Image[int(ry):int(ry+dy),int(rx):int(rx+dx)]
            D=N>C
        else:
            ty= y-fy
            tx=x-fx
            w1=(1-tx)*(1-ty)
            w2=tx*(1-ty)
            w3=(1-tx)*ty
            w4=tx*ty

            a1=Image[int(fy):int(fy+dy),int(fx):int(fx+dx)]
            a2=Image[int(fy):int(fy+fy),int(cx):int(cx+dx)]
            a3=Image[int(cy):int(cy+dy),int(fx):int(fx+dx)]
            a4=Image[int(cy):int(cy+dy),int(cx):int(cx+dx)]
            N=w1*a1+w2*a2+w3*a3+w4*a4
            D=N>C
        v=2**(i-1)
        result=result+v*D
    bins=newMax
    m,n=result.shape
    for i in range(m):
        for j in range(n):
            result[i][j]=table[int(np.round(result[i][j]))]
    result_vect=result.flatten()
    ma=max(result_vect)
    mi=min(result_vect)
    keys=np.arange(mi,ma+1,1)
    hist=dict()
    for i in keys :
        hist[i]=0
    for i in result_vect:
        hist[i]+=1
    result_hist_count=np.array(list(hist.values()))
    result_hist_count=result_hist_count /sum(result_hist_count)
    return result_hist_count




