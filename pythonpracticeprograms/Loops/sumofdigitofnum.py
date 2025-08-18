num=int(input("enter the number: "))
sum=0
while num>0:
    sum += num %10
    num = num//10
print("sum of the digits of the given number: ",sum)