import numpy as np
def bitshift_left(number,bitposition):
    number=np.uint16(number)
    return number<<bitposition
def bitget(number,position):
    return number>>position-1 & 1
def bitset(number,positon,bit):
    number=np.uint16(number)
    mask=np.uint16(1)
    mask=mask<<positon-1
    if (bit==0):
        mask=~mask
        return number & mask
    else:
        return number | mask
def list_bit_get(number,samples):
    l=[]
    for i in samples:
        l.append(bitget(number,i))
    return l
#print(list_bit_get(9,np.arange(0,16,1)))
##print(bitset(11,3))
##print(bitget(11,1))
##print(bitshift_left(2,2))
def  getmapping(samples,mappingtype):
    tables=np.arange(0,2**samples,1)
    #print(tables)
    newMAx=0
    index=0
    #uniform :less than two transitions in binary patterns
    if (mappingtype=='u2'):
        newMAx=samples*(samples-1)+3
        for i in tables:
            if (i==196):
                l=1
            a=bitshift_left(i,1)
            b=bitget(i,samples)
            j=bitset(bitshift_left(i,1),1,bitget(i,samples))
            c=list_bit_get(i^j,np.arange(1,samples+1,1))
            numt=np.sum(c)

            if numt <=2:
                tables[i]= index
                index=index+1
            else:
                tables[i]=newMAx-1
    # rotation invariant 
    elif(mappingtype=='ri'):
        tmpMap=np.zeros_like(tables)-1
        for i in tables:
            rm=i
            r=i
            for j in range(1,16):
                r= bitset(bitshift_left(r,1),1,bitget(r,samples))
                if r<rm:
                    rm=r
            if tmpMap[rm] <0:
                tmpMap[rm] = newMAx
                newMAx = newMAx+1
            tables[i]=tmpMap[rm]
    else:
        newMAx= samples+2
        for i in tables:
            j=bitset(bitshift_left(i,1),1,bitget(i,samples))
            numt=np.sum(list_bit_get(i^j,np.arange(1,samples+1,1)))
            if numt<=2:
                tables[i]= np.sum(list_bit_get(i,np.arange(1,samples+1,1)))
            else:
                tables[i]=samples+1
    return tables,samples,newMAx
#v=getmapping(8,'riu2')
#print(v)
'''print(bitget(10,4))
print(10<<1)
print(list_bit_get(6,16))'''