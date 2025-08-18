str = input("enter the string:")
reversed=""
for char in str:
    reversed = char + reversed
print("reversed string is:",reversed)
   
