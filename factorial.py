def fact(num,temp):
    for i in range(1,num+1):
        temp=temp*i
    print(temp)

num=6
temp=1
fact(num,temp)