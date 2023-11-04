import math
def gamma(vector):
    for i in range(len(vector)):
        vector[i]=math.gamma(vector[i])
        return vector
    