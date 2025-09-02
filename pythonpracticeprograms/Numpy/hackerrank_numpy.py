import numpy as np
def arrays(arr):
    # complete this function
    # use numpy.array
    return np.array(arr[::-1], float)
    
    
arr =[1,2,3,4,-8,-10]
result = arrays(arr)
print(result)