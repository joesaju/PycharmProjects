import numpy as np

arr=np.ones((3,3,4),dtype=int)# basic array creation
print(arr)

matrix = np.arange(10,25,5)
print(matrix)

matrix1 = np.full((2,2),7)
print(matrix1)

m2= np.eye(2)
print(m2)

m3 = np.empty((3,2))
print(m3)


m4=np.arange(1, 20 , 2, 
          dtype = np.float32)
print(m4)

m5=np.array([3,6,10], dtype=np.int32)
print(m5)

m6=np.linspace(2,20,num=4,retstep=True)
print(m6)

m7=np.arange(2,20,2)
print(m7)

a = np.array([1,2,3,4])
x=a.copy()
a[0]=23

print(a)
print(x)
