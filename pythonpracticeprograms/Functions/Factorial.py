def factorial(n):
    fact=1
    for i in range(1,n+1):
        fact = fact*i
    print(f"the factorial no.of {n} is:",fact)

    
n=int(input("enter the number: "))
factorial(n)