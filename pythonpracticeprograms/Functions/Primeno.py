num=int(input("enter the number:")) 
count=2
for i in range(2,num):
    if(num%i==0):
        count+=1
        break;
if(count==2):
    print("prime number")
else:
    print("not a prime number")
