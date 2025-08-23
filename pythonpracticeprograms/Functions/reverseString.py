def reversed_str(str):
    reversed=""
    for char in str:
        reversed = char + reversed
    print("reversed string is:",reversed)
    return reversed

str=input("enter the string:")
reversed_str(str)
   
