import math

def areaofCircl(r):
    area = math.pi*r*r
    return area
r=6
res=areaofCircl(r)
print("area of circle :", res)

def volumeofsphere(r):
    volume= 4/3*math.pi*r*r*r
    return volume

r=3
res = volumeofsphere(r)
print("volume of sphere: ",res)

