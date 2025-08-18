#Print the multiplication table of a number entered by the user.
n = int(input("enter the number to create multiplication table:"))
for i in range(1,11):
    print(f"{n} x {i} = {n*i}")
 