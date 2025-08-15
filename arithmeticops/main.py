import arithmeticops

a=int(input("enter a: "))
b=int(input("enter b: "))
choice=str(input("enter your choice: "))
if(choice=='add'):
 value = arithmeticops.add(a,b)

elif(choice=='sub'):
    value1= arithmeticops.sub(a,b)
   
elif(choice=='mul'):
    value2 = arithmeticops.mul(a,b)
   
elif(choice=='div'):
    value3 = arithmeticops.div(a,b)
    
elif(choice=='mod'):
    value4 = arithmeticops.mod(a,b)
    

